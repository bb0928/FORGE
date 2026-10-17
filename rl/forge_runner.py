"""ForgeRunner: Multi-gpu runner for FORGE residual policy.

Extends rsl-rl's OnPolicyRunner with:
1. Curriculum learning (upper body residual scaling 0 -> 0.8)
2. Entropy annealing (0.01 -> 0.001)
3. Base policy action composition

Multi-gpu: Uses rsl-rl's built-in DDP support via MjlabOnPolicyRunner.
"""

from __future__ import annotations

import os
import time
import torch

from mjlab.rl.runner import MjlabOnPolicyRunner


class ForgeRunner(MjlabOnPolicyRunner):
    """Runner for FORGE residual training with curriculum and adaptation.

    Key differences from standard runner:
    - Action curriculum for upper body residual
    - Entropy annealing schedule
    - Base policy composition: final = base + residual * scale * curriculum
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # base_mode=True trains a standalone base policy (no frozen base, no curriculum):
        # the actor output IS the full-body action. Used by train_forge_base.py to produce
        # a checkpoint whose critic can be loaded into the residual policy. In base mode the
        # whole body is driven by the actor, so the upper-body curriculum is forced to 1.0.
        self.base_mode = bool(train_cfg.get("base_mode", False))

        # Curriculum config. Both base and residual ramp the curriculum from 0; it gates
        # the FK wrist-target amplitude (task difficulty). base covers the SIMPLE range
        # (0 -> curriculum_max, default 0.5); residual covers the HARD range (warm-started,
        # 0.5 -> 1.0). curriculum_max / curriculum_min read from cfg.
        self.curriculum_min = train_cfg.get("curriculum_min", 0.0)
        self.curriculum_ratio = self.curriculum_min
        self.curriculum_max = train_cfg.get("curriculum_max", 0.5 if self.base_mode else 1.0)
        # increment now read from cfg (was hard-coded 0.05). Residual lowers it to 0.02 so each
        # curriculum step is a smaller jump in wrist-target difficulty -> the value function can
        # absorb the return-distribution shift (0.05 jumps caused value loss to spike to 12 and
        # reward to regress around iter368/curriculum0.5 in residual_v14base).
        self.curriculum_increment = train_cfg.get("curriculum_increment", 0.05)
        # FORGE trigger: episodic mean of the SINGLE tracking_x_vel term must exceed
        # 0.8 * its reward weight. Because both the episode-sum and the weight carry the
        # same dt factor, dt cancels and the condition reduces to:
        #   sum_over_episode(step_reward_xvel) / max_episode_length > 0.8 * weight
        # where step_reward_xvel = raw(0..1) * weight (no dt). So the threshold is
        # 0.8 * weight. With weight 2.0 -> 1.6. (Old code compared TOTAL episode reward
        # mean against 0.5, wrong metric -> curriculum stuck at 0 forever.)
        self.curriculum_track_term = "track_linear_velocity"
        # Must match the track_linear_velocity reward weight in the cfg (threshold is
        # 0.8*weight on per-walk-step mean). Reward weight raised to 4.0, so set 4.0.
        self.curriculum_track_weight = train_cfg.get("curriculum_track_weight", 4.0)
        self.curriculum_threshold_frac = 0.8

        # Entropy annealing config (read from cfg so residual can set 0 -> no entropy
        # pressure, letting split std anneal toward what maximizes reward).
        self.entropy_start = train_cfg.get("entropy_start", 0.01)
        self.entropy_end = train_cfg.get("entropy_end", 0.001)
        self.entropy_decay_iters = train_cfg.get("entropy_decay_iters", 15000)
        # V33 std L2 reg strength (curriculum-gated). Read from cfg so V34+ tune in one place.
        self.std_l2_max = train_cfg.get("std_l2_max", 0.03)
        # V35 two-stage std-reg gate (replaces curr<0.1 gate). Latches ON when the policy
        # is demonstrably capable: iter>std_l2_gate_iter AND mean ep_len>std_l2_gate_eplen
        # AND per-walk-step track>std_l2_gate_track. Once latched it stays on (no flip-flop).
        # Stage 1 (curr=0): constant base reg breaks the "dark-ages" plateau. Stage 2: as the
        # curriculum rises, reg ramps base->max. After std_l2_hardstop iters, hold at max.
        self.std_l2_base = train_cfg.get("std_l2_base", 0.01)
        self.std_l2_hard_open = train_cfg.get("std_l2_hard_open", 500)
        self.std_l2_gate_iter = train_cfg.get("std_l2_gate_iter", 500)
        self.std_l2_gate_eplen = train_cfg.get("std_l2_gate_eplen", 980.0)
        self.std_l2_gate_track = train_cfg.get("std_l2_gate_track", 1.0)
        self.std_l2_hardstop = train_cfg.get("std_l2_hardstop", 3000)
        self._reg_gate_latched = False
        self._reg_track_est = 0.0

        # Action composition.
        # IMPORTANT: the mjlab env's JointPositionAction term already applies the per-joint
        # scale and default offset internally on env.step():
        #     joint_target = default_pos + action * G1_ACTION_SCALE
        # So the runner must pass the RAW summed action (base + residual) WITHOUT any extra
        # scaling. The previous code multiplied residual by 0.25 here, which (a) double-scaled
        # on top of the env's G1_ACTION_SCALE and (b) applied a coefficient only to residual,
        # breaking the strict base+residual semantics. We now hand the env exactly base+residual.
        self.num_actions = env.num_actions
        # waist_yaw(12) belongs to UPPER (steering/manip); waist_roll(13)/pitch(14) stay
        # in LOWER (balance). Aligned with FORGE fix_waist=False split.
        self.upper_dof_indices = [12] + list(range(15, self.num_actions))

        # Resolve the column index of the tracking term inside RewardManager._step_reward.
        self._track_term_idx = self._resolve_reward_term_index(self.curriculum_track_term)

        # Episode tracking for curriculum: accumulate tracking_x_vel ONLY on walk-mode steps,
        # averaged over the walk-step COUNT (not full episode length). Rationale: mode is
        # resampled every 3-8s, so a 20s episode mixes walk/manip segments. In manip mode
        # vel_command=0 and the robot stands -> tracking_x_vel is trivially ~max (2.0), which
        # would inflate the curriculum metric and advance it prematurely (FORGE has this same
        # optimistic bias; this is our deliberate improvement). By measuring per-walk-step
        # tracking we get "how well does it walk WHEN walking", independent of manip fraction.
        n = env.num_envs
        self._ep_walk_track_sum = torch.zeros(n, device=device)
        self._ep_walk_steps = torch.zeros(n, device=device)
        # Require a minimum number of walk steps in an episode before it can vote on
        # curriculum (avoids noise from episodes that barely walked).
        self._min_walk_steps = 25
        # FORGE only checks/advances the curriculum once per episode period
        # (common_step_counter % max_episode_length == 0) and bumps at most +increment.
        # We replicate that gating: advance at most once every `_curriculum_check_period`
        # env-steps, by +increment, so ramp is gradual (otherwise advancing on every `done`
        # batch over 4096 envs slams ratio 0->1 in a few dozen iters).
        # check_period: env-steps between curriculum advances. cfg can override (residual sets
        # it larger, e.g. 2000, to give the value function more iters to catch up between jumps).
        _cfg_period = train_cfg.get("curriculum_check_period", None)
        if _cfg_period is not None:
            self._curriculum_check_period = int(_cfg_period)
        else:
            try:
                self._curriculum_check_period = int(env.unwrapped.max_episode_length)
            except Exception:
                self._curriculum_check_period = 1000
        self._steps_since_curriculum_check = 0

        print(f"ForgeRunner: curriculum={self.curriculum_ratio:.2f} (max {self.curriculum_max}), "
              f"entropy={self.entropy_start}->{self.entropy_end}, std_l2_max={self.std_l2_max}, "
              f"track_term={self.curriculum_track_term}[idx={self._track_term_idx}] "
              f"threshold={self.curriculum_threshold_frac * self.curriculum_track_weight:.3f} (per walk-step), "
              f"upper_dof={self.upper_dof_indices}")

    def _resolve_reward_term_index(self, term_name):
        """Find the column index of a reward term in RewardManager._step_reward."""
        try:
            rm = self.env.unwrapped.reward_manager
            names = list(rm._term_names)
            if term_name in names:
                return names.index(term_name)
            print(f"[WARN] curriculum term '{term_name}' not in reward terms {names}; "
                  f"curriculum disabled.")
        except Exception as e:
            print(f"[WARN] could not resolve reward term index ({e}); curriculum disabled.")
        return None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Override learn to add curriculum + entropy annealing + action composition."""
        from tensordict import TensorDict

        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations()
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=obs["actor"].shape[0])
        obs = obs.to(self.device)

        # Sync the (possibly resumed) curriculum ratio to the command term up front, so the
        # FK target amplitude / squat depth match curriculum from step 0. Both base and
        # residual ramp curriculum now.
        self._push_curriculum_to_command()

        self.alg.train_mode()

        # Sync parameters for multi-gpu
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            self._push_iteration_to_command(it)
            start = time.time()
            local_it = it - start_it

            # Entropy annealing
            progress = min(it / max(self.entropy_decay_iters, 1), 1.0)
            entropy_coef = self.entropy_start + (self.entropy_end - self.entropy_start) * progress
            self.alg.entropy_coef = entropy_coef
            # V33: upper-body std L2 penalty coef, gated by curriculum_ratio.
            # curr<0.1 -> 0 (free explore basics). curr 0.1->1.0 -> lambda 0->std_l2_max.
            # V35 two-stage gate. Estimate current per-walk-step track + mean ep_len.
            _c = self.curriculum_ratio
            try:
                _eplen = float(self.env.episode_length_buf.float().mean().item())
            except Exception:
                _eplen = 0.0
            _valid = self._ep_walk_steps > 0
            if _valid.any():
                _track_now = float((self._ep_walk_track_sum[_valid] / self._ep_walk_steps[_valid]).mean().item())
                # smooth to avoid single-iter noise flipping the gate
                self._reg_track_est = 0.9 * self._reg_track_est + 0.1 * _track_now
            # V35.1: gate on CAPABILITY ONLY (no iter condition -> survives resume where it
            # resets to 0). Latch when the policy can stand+track, then keep reg on.
            # V35.3: gate = track_est>1.0 OR iter>hard_open. Primary signal is track_est
            # (capability), but iter>hard_open is a GUARANTEED fallback so the gate ALWAYS
            # opens even if track somehow stalls. (ep_len check removed in v35.2: it read
            # episode_length_buf.mean() before rollout, polluted by reset envs, never hit 980.)
            if (not self._reg_gate_latched
                    and (self._reg_track_est > self.std_l2_gate_track
                         or it > self.std_l2_hard_open)):
                self._reg_gate_latched = True
            if not self._reg_gate_latched:
                self.alg.std_l2_coef = 0.0                       # not yet capable -> no reg
            else:
                # stage1 constant base (now strong enough to bite early) + stage2 curric ramp base->max
                self.alg.std_l2_coef = self.std_l2_base + (self.std_l2_max - self.std_l2_base) * min(_c, 1.0)

            # Rollout
            with torch.inference_mode():
                for step in range(self.cfg["num_steps_per_env"]):
                    # Get actions from policy (residual in residual-mode; full action in base-mode)
                    actions = self.alg.act(obs)

                    if self.base_mode:
                        # Base training (NEW design): base IS a full-body policy that controls
                        # the upper body to TRACK the FK wrist target (same task as residual,
                        # just the simple curriculum range 0->0.5). The arm motion to reach the
                        # target perturbs the lower body, so the lower body learns balance for
                        # free. No base-action composition; policy output is the full action.
                        # Env applies G1_ACTION_SCALE.
                        final_action = actions
                    else:
                        # Apply curriculum: scale upper-body residual (waist_yaw + arms).
                        actions_curriculum = actions.clone()
                        actions_curriculum[:, self.upper_dof_indices] *= self.curriculum_ratio

                        # Compose: final = base + residual (RAW, no scale here).
                        # The env's JointPositionAction term applies the per-joint G1_ACTION_SCALE
                        # and default offset internally: joint_target = default + final*G1_SCALE.
                        # Base and residual share one identical per-joint scale, and the policy
                        # sees a fixed, non-tunable action->joint mapping (FORGE semantics).
                        base_action = self.env.get_base_action(obs)
                        final_action = base_action + actions_curriculum

                    # Step environment
                    obs, rewards, dones, extras = self.env.step(final_action.to(self.env.device))

                    # Surface curriculum ratio to the logger (both base and residual).
                    try:
                        log_d = self.env.unwrapped.extras.setdefault("log", {})
                        log_d["Curriculum/upper_target_ratio"] = float(self.curriculum_ratio)
                        # V35c4: StdL2 reg diagnostics go to WANDB ONLY (not the console table,
                        # which printed them every iter and spammed the log). Visualize as curves.
                        if self.cfg.get("logger") == "wandb":
                            try:
                                import wandb as _wb
                                if _wb.run is not None:
                                    _wb.log({
                                        "StdL2/coef": float(getattr(self.alg, "std_l2_coef", 0.0)),
                                        "StdL2/gate_latched": float(self._reg_gate_latched),
                                        "StdL2/track_est": float(self._reg_track_est),
                                    }, commit=False)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Handle dict obs
                    if isinstance(obs, dict):
                        obs = TensorDict(obs, batch_size=obs["actor"].shape[0])
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Process step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)

                    # Track per-walk-step tracking_x_vel for curriculum (both base & residual:
                    # advance the FK-target-amplitude curriculum when walking is good enough).
                    if self._track_term_idx is not None:
                        cmd = self.env.unwrapped.command_manager.get_term("twist")
                        is_walk = cmd.is_walk_mode.to(self.device).float()  # (n,)
                        step_track = self.env.unwrapped.reward_manager._step_reward[:, self._track_term_idx].to(self.device)
                        self._ep_walk_track_sum += step_track * is_walk
                        self._ep_walk_steps += is_walk
                        done_ids = torch.where(dones)[0]

                        # Advance curriculum at most once per check period (FORGE-style gradual
                        # ramp), evaluated over the envs that just finished an episode.
                        self._steps_since_curriculum_check += 1
                        if (self._steps_since_curriculum_check >= self._curriculum_check_period
                                and len(done_ids) > 0):
                            self._update_curriculum(done_ids)
                            self._steps_since_curriculum_check = 0

                        # Reset per-env accumulators on their own done (independent of advance).
                        if len(done_ids) > 0:
                            self._ep_walk_track_sum[done_ids] = 0
                            self._ep_walk_steps[done_ids] = 0

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )

            # Save
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        # Final save
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def _update_curriculum(self, done_ids):
        """Advance action curriculum when WALKING quality is good enough.

        Walk-only variant of FORGE's update_action_curriculum: we average tracking_x_vel
        over the walk-mode steps only (step_reward = raw*weight, no dt), so the metric is
        weight * mean_raw_over_walk_steps, compared against 0.8 * weight. Episodes with
        fewer than _min_walk_steps walk steps don't vote (too noisy). After advancing, the
        new ratio is pushed to the command term so squat depth ramps with it.
        """
        if self._track_term_idx is None:
            return
        valid = self._ep_walk_steps[done_ids] >= self._min_walk_steps
        if not valid.any():
            return

        ids = done_ids[valid]
        # Per-walk-step mean tracking reward = weight * mean_raw (independent of manip fraction).
        per_walk_step = (self._ep_walk_track_sum[ids] / self._ep_walk_steps[ids]).mean().item()
        threshold = self.curriculum_threshold_frac * self.curriculum_track_weight

        if per_walk_step > threshold:
            self.curriculum_ratio = min(
                self.curriculum_ratio + self.curriculum_increment,
                self.curriculum_max
            )
            self._push_curriculum_to_command()

    def _push_curriculum_to_command(self):
        """Propagate curriculum_ratio to the command term so squat depth ramps with it."""
        try:
            cmd = self.env.unwrapped.command_manager.get_term("twist")
            if hasattr(cmd, "set_curriculum_ratio"):
                cmd.set_curriculum_ratio(self.curriculum_ratio)
        except Exception:
            pass

    def _push_iteration_to_command(self, iteration: int):
        try:
            cmd = self.env.unwrapped.command_manager.get_term("twist")
            if hasattr(cmd, "set_iteration"):
                cmd.set_iteration(iteration)
        except Exception:
            pass

    def save(self, path, infos=None):
        """Save checkpoint including curriculum state."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = {**(infos or {}), "curriculum_ratio": self.curriculum_ratio}
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        """Load checkpoint including curriculum state."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict.get("infos", {})
        if "curriculum_ratio" in infos:
            self.curriculum_ratio = infos["curriculum_ratio"]
            self._push_curriculum_to_command()
            # V35c4: a resumed run already stands+tracks -> latch the std-reg gate immediately
            # (don't make it re-earn capability from track_est=0 after the iter counter resets).
            if self.curriculum_ratio > 0.0:
                self._reg_gate_latched = True
        return infos
