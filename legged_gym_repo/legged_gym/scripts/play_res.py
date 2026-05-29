

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import onnxruntime as ort

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

# ? >>> Control
control_range = {
    "x_vel": (-0.8, 1.0),
    "y_vel": (-0.5, 0.5),
    "yaw_vel": (-0.8, 0.8),
    "height": (0.34, 0.74),
    "freq": (1.0, 2.0),
}

try:

    from legged_gym.utils.gamepad import GamepadController

    gamepad_controller = GamepadController(control_range)
except:

    gamepad_controller = None
    print("Gamepad controller not available, using keyboard controls instead.")

try:
    from pynput import keyboard
    _keyboard_available = True
except Exception:
    keyboard = None
    _keyboard_available = False

control = {"x_vel": 0.0, "y_vel": 0.0, "yaw_vel": 0.0, "height": 0.74, "freq": 1.0, "exit_flag": False}
step_size = 0.1
height_step = 0.01
freq_step = 0.01
key_pressed = {
    "w": False,
    "s": False,
    "a": False,
    "d": False,
    "e": False,
    "r": False,
    "x": False,
    "z": False,
    "f": False,
    "g": False,
}


def _build_command_layout(commands_dim_len_dict):
    """Map command names to tensor slices to stay in sync with updated state space."""
    layout = {}
    start = 0
    for name, length in commands_dim_len_dict.items():
        if length > 0:
            layout[name] = slice(start, start + length)
        else:
            layout[name] = None
        start += length
    return layout


def _get_scalar_column(layout, key):
    slc = layout.get(key)
    if slc is None or not isinstance(slc, slice):
        return None
    if slc.stop - slc.start != 1:
        return None
    return slc.start


def _get_slice(layout, key):
    """Return a slice for a multi-dim command or None if missing."""
    slc = layout.get(key)
    if slc is None or not isinstance(slc, slice):
        return None
    return slc


def on_press(key):
    try:
        key_char = key.char
        if key_char in key_pressed:
            key_pressed[key_char] = True
        if key_char == "q":
            control["exit_flag"] = True
    except AttributeError:
        pass


def on_release(key):
    try:
        key_char = key.char
        if key_char in key_pressed:
            key_pressed[key_char] = False
    except AttributeError:
        pass


def handle_keyboard():
    if not _keyboard_available:
        return
    if key_pressed["w"]:
        control["x_vel"] += step_size
        control["x_vel"] = min(control["x_vel"], control_range["x_vel"][1])
    if key_pressed["s"]:
        control["x_vel"] -= step_size
        control["x_vel"] = max(control["x_vel"], control_range["x_vel"][0])
    if key_pressed["a"]:
        control["y_vel"] += step_size
        control["y_vel"] = min(control["y_vel"], control_range["y_vel"][1])
    if key_pressed["d"]:
        control["y_vel"] -= step_size
        control["y_vel"] = max(control["y_vel"], control_range["y_vel"][0])
    if key_pressed["e"]:
        control["yaw_vel"] += step_size
        control["yaw_vel"] = min(control["yaw_vel"], control_range["yaw_vel"][1])
    if key_pressed["r"]:
        control["yaw_vel"] -= step_size
        control["yaw_vel"] = max(control["yaw_vel"], control_range["yaw_vel"][0])
    if key_pressed["x"]:
        control["height"] += height_step
        control["height"] = min(control["height"], control_range["height"][1])
    if key_pressed["z"]:
        control["height"] -= height_step
        control["height"] = max(control["height"], control_range["height"][0])
    if key_pressed["f"]:
        control["freq"] += freq_step
        control["freq"] = min(control["freq"], control_range["freq"][1])
    if key_pressed["g"]:
        control["freq"] -= freq_step
        control["freq"] = max(control["freq"], control_range["freq"][0])
    if not key_pressed["w"] and not key_pressed["s"]:
        control["x_vel"] = 0.0
    if not key_pressed["a"] and not key_pressed["d"]:
        control["y_vel"] = 0.0
    if not key_pressed["e"] and not key_pressed["r"]:
        control["yaw_vel"] = 0.0


# ? <<< Control


def load_policy():
    body = torch.jit.load("", map_location="cuda:0")

    def policy(obs):
        action = body.forward(obs)
        return action

    return policy


def load_onnx_policy():
    model = ort.InferenceSession("")

    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")

    return run_inference


def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74, freq=1.0):
    listener = None
    # Only start keyboard listener if available and not headless
    if _keyboard_available and not args.headless:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # # -------------------- modify cfgs to match 116-dim checkpoint --------------------
    # # Remove wrist_quat_command (8 dims) and dof_residual (14 dims) to get 116 dims
    #     del env_cfg.env.one_step_obs_dims['wrist_quat_command']
    #     del env_cfg.env.one_step_obs_dims['dof_residual']
    
    # # Update total observation count
    # env_cfg.env.num_one_step_observations = sum(env_cfg.env.one_step_obs_dims.values())
    # env_cfg.env.num_observations = env_cfg.env.num_actor_history * env_cfg.env.num_one_step_observations
    
    # # Adaptation encoder in old checkpoint uses 1 frame (116 dims), not 6 frames (696 dims)
    # train_cfg.policy.adaptation_history_length = 1
    
    # # -------------------- end --------------------
    
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.domain_rand.obs_delay_feet_enable = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.asset.self_collision = 0
    env_cfg.env.upper_teleop = False
    freq_control = env_cfg.commands.freq_control
    
    print("\n=== Config Information ===")
    if hasattr(train_cfg.policy, 'adaptation_encoder_dims'):
        print(f"adaptation_encoder_dims: {train_cfg.policy.adaptation_encoder_dims}")
    if hasattr(train_cfg.policy, 'adaptation_decoder_dims'):
        print(f"adaptation_decoder_dims: {train_cfg.policy.adaptation_decoder_dims}")
    if hasattr(train_cfg.policy, 'adaptation_groups'):
        print(f"adaptation_groups: {train_cfg.policy.adaptation_groups}")
    print("=========================\n")
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    freq_control = getattr(env.cfg.commands, "freq_control", freq_control)

    command_layout = _build_command_layout(env.cfg.commands.commands_dim_len_dict)
    lin_x_idx = _get_scalar_column(command_layout, "lin_vel_x")
    lin_y_idx = _get_scalar_column(command_layout, "lin_vel_y")
    yaw_idx = _get_scalar_column(command_layout, "ang_vel_yaw")
    height_idx = _get_scalar_column(command_layout, "height")
    freq_idx = _get_scalar_column(command_layout, "frequency")
    heading_idx = _get_scalar_column(command_layout, "heading")
    wrist_force_indices = [
        _get_scalar_column(command_layout, "wrist_force_x"),
        _get_scalar_column(command_layout, "wrist_force_y"),
        _get_scalar_column(command_layout, "wrist_force_z"),
    ]

    # slices for multi-dim commands (used to drive pos masks)
    feet_slice = _get_slice(command_layout, "feet_pos")
    wrist_pose_slice = _get_slice(command_layout, "wrist_pos")
    head_slice = _get_slice(command_layout, "head_pos")

    ratio_mappings = [
        (lin_x_idx, control_range["x_vel"]),
        (lin_y_idx, control_range["y_vel"]),
        (yaw_idx, control_range["yaw_vel"]),
        (height_idx, control_range["height"]),
    ]

    def update_command_ratio():
        for idx, bounds in ratio_mappings:
            if idx is None:
                continue
            cmd_vals = env.commands[:, idx]
            pos_cap = bounds[1]
            neg_cap = bounds[0]
            pos_term = (cmd_vals / pos_cap) * (cmd_vals > 0) if pos_cap != 0 else torch.zeros_like(cmd_vals)
            neg_term = (cmd_vals / neg_cap) * (cmd_vals < 0) if neg_cap != 0 else torch.zeros_like(cmd_vals)
            env.command_ratio[:, idx] = pos_term + neg_term

    def apply_user_commands(x_value, y_value, yaw_value, height_value, freq_value):
        if lin_x_idx is not None:
            env.commands[:, lin_x_idx] = x_value
        if lin_y_idx is not None:
            env.commands[:, lin_y_idx] = y_value
        if yaw_idx is not None:
            env.commands[:, yaw_idx] = yaw_value
        if height_idx is not None:
            env.commands[:, height_idx] = height_value
        if heading_idx is not None:
            env.commands[:, heading_idx] = 0.0
        if freq_control and freq_idx is not None and freq_value is not None:
            env.commands[:, freq_idx] = freq_value
        update_command_ratio()

    apply_user_commands(x_vel, y_vel, yaw_vel, height, freq if freq_control else None)
    
    print("\n=== Environment Observation Dimensions ===")
    print(f"Actor obs dims: {env.num_obs} (one_step: {env.num_one_step_obs}, history: {env.actor_history_length})")
    print(f"Critic obs dims: {env.num_privileged_obs} (one_step: {env.num_one_step_privileged_obs}, history: {env.critic_history_length})")
    print(f"Actor obs groups ({len(env.one_step_obs_dims)}): {list(env.one_step_obs_dims.keys())}")
    print(f"Critic obs groups ({len(env.one_step_privileged_obs_dims)}): {list(env.one_step_privileged_obs_dims.keys())}")
    print(f"Number of actions: {env.num_actions}")
    print("==========================================\n")
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)  # Use this to load from trained pt file


    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        # Reset environment to get obs_batch for sanity check
        env.reset_idx(torch.arange(env.num_envs).to(env.device))
        obs_batch = env.get_observations()  # Returns [obs_buf, obs_mask_buf] concatenated
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, obs_batch=obs_batch)
        print("Exported policy as jit script to: ", path)
    print(policy)
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    # get initial observations before entering control loop
    obs = env.get_observations()



    while not control["exit_flag"]:
        handle_keyboard()
        env.action_curriculum_ratio = 1.0
        
        with torch.no_grad():
            policy_output = policy(obs)
            # Handle both tuple (actions, force_residual_pred) and single actions for backward compatibility
            if isinstance(policy_output, tuple) and len(policy_output) == 2:
                actions, force_residual_pred = policy_output
            else:
                actions = policy_output
                force_residual_pred = None
        
        # # ------------------ DEBUG DUMP: POLICY I/O START ------------------
            
        #         "step": env.common_step_counter
        #     }
            
        #         pickle.dump(policy_io_data, f)
        # # ------------------ DEBUG DUMP: POLICY I/O END ------------------

        # )

        obs, _, _, _, _, _, _ = env.step(actions, force_residual_pred=force_residual_pred)


        if hasattr(env, "is_wrist_pos") and hasattr(env, "is_feet_pos") and hasattr(env, "is_head_pos"):
            is_wrist_pos = env.is_wrist_pos.squeeze(1).detach().cpu().numpy()
            is_feet_pos = env.is_feet_pos.squeeze(1).detach().cpu().numpy()
            is_head_pos = env.is_head_pos.squeeze(1).detach().cpu().numpy()
            is_wrist_force = getattr(env, "is_wrist_force", torch.zeros_like(env.is_wrist_pos)).squeeze(1).detach().cpu().numpy()
            # is_height can be shape [N] (bool) or [N,1]; handle both
            is_height_buf = env.is_height
            if is_height_buf.dim() > 1:
                is_height_buf = is_height_buf.squeeze(1)
            is_height = is_height_buf.detach().cpu().numpy()
            upper_policy_mask = env.upper_policy_mask.squeeze(1).detach().cpu().numpy() if hasattr(env, "upper_policy_mask") else None
            
            current_modes = []
            for i in range(env.num_envs):
                if is_wrist_pos[i]:
                    mode = "wrist_pos"
                elif is_feet_pos[i]:
                    mode = "feet_pos"
                elif is_head_pos[i]:
                    mode = "head_pos"
                elif is_wrist_force[i]:
                    mode = "wrist_force"
                elif is_height[i]:
                    mode = "height"
                else:
                    mode = "base_vel"
                current_modes.append(mode)
            
            unique_modes = set(current_modes)


        

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=0, y_vel=0.0, yaw_vel=0.0, height=0.74, freq=1.0)

