# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
import random
import math
from isaacgym import gymapi
from isaacgym import gymutil
import torch.nn.functional as F
from typing import Tuple, List
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


# def class_to_dict(obj, depth=-1) -> dict:
#     if not hasattr(obj, "__dict__"):
#         return obj
#     result = {}
#     for key in dir(obj):
#         if key.startswith("_"):
#             continue
#         element = []
#         val = getattr(obj, key)
#         if depth == -1:
#             if isinstance(val, list):
#                 for item in val:
#                     element.append(class_to_dict(item))
#             else:
#                 element = class_to_dict(val)
#             result[key] = element
#         else:
#             # import pdb; pdb.set_trace()
#             if depth == 1:
#                 result[key] = val
#             else:
#                 if isinstance(val, list):
#                     for item in val:
#                         element.append(class_to_dict(item, depth-1))
#                 else:
#                     element = class_to_dict(val, depth-1)
#                 result[key] = element
#     return result

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.command_sample_strategy is not None:
            env_cfg.commands.command_sample_strategy = args.command_sample_strategy
        # import pdb; pdb.set_trace()
        if args.tracking_base_height_judge_return0 is not None:
            env_cfg.rewards.tracking_base_height_judge_return0 = args.tracking_base_height_judge_return0
        if args.tracking_waist_action_judge_return0 is not None:
            env_cfg.rewards.tracking_waist_action_judge_return0 = args.tracking_waist_action_judge_return0
        if args.wrist_pos_sample_theta_min is not None and args.wrist_pos_sample_theta_max is not None:
            env_cfg.commands.ranges.wrist_pos_sample_theta_range = [args.wrist_pos_sample_theta_min*np.pi, args.wrist_pos_sample_theta_max*np.pi]
        if args.head_pos_sample_theta_min is not None and args.head_pos_sample_theta_max is not None:
            env_cfg.commands.ranges.head_pos_sample_theta_range = [args.head_pos_sample_theta_min*np.pi, args.head_pos_sample_theta_max*np.pi]
        if args.use_domain is not None:
            env_cfg.domain_rand.use_random = args.use_domain
        if args.base_model_path is not None and getattr(env_cfg.env, "base_model_env", None) is not None:
            env_cfg.env.base_model_env.model_path = args.base_model_path
        if args.use_upper_action is not None and getattr(env_cfg, "use_upper_action", None) is not None:
            env_cfg.use_upper_action = args.use_upper_action
        
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if args.init_critic_from_base_model is not None:
            cfg_train.policy.init_critic_from_base_model = args.init_critic_from_base_model
        if args.policy_class_name is not None:
            cfg_train.runner.policy_class_name = args.policy_class_name
        if args.init_std_upper is not None:
            cfg_train.init_std_upper = args.init_std_upper
    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "aliengo",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {
            "name": "--resume_path",
            "type": str,
            "help": "Path to the checkpoint to resume from. Overrides config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {"name": "--load_model_from_pretrained", "action": "store_true", "default": False, "help": "load pretrained model"},
        {
            "name": "--pretrained_path",
            "type": str,
            "default": "",
            "help": "Path to the pretrained checkpoint to load from. Overrides config file if provided.",
        },
        {
            "name": "--command_sample_strategy",
            "type": str,
            "default": None,
            "help": "Command sample strategy. Overrides config file if provided.",
        },
        {"name": "--tracking_base_height_judge_return0", 
         "action": "store_true", 
         "default": False,
         "help": "tracking base height judge return 0"
        },
        {"name": "--tracking_waist_action_judge_return0", 
         "action": "store_true", 
         "default": False,
         "help": "tracking waist action judge return 0"
        },
        {
            "name": "--wrist_pos_sample_theta_min",
            "type": float,
            "default": 0.5,
            "help": "wrist pos sample theta range. Overrides config file if provided.",
        },
        {
            "name": "--wrist_pos_sample_theta_max",
            "type": float,
            "default": 1,
            "help": "wrist pos sample theta range. Overrides config file if provided.",
        },
        {
            "name": "--head_pos_sample_theta_min",
            "type": float,
            "default": 0.5,
            "help": "head pos sample theta range. Overrides config file if provided.",
        },
        {
            "name": "--head_pos_sample_theta_max",
            "type": float,
            "default": 1,
            "help": "head pos sample theta range. Overrides config file if provided.",
        },
        {
            "name": "--use_domain",
            "action": "store_true",
            "default": False,
            "help": "Use domain randomization.",
        },
        {
            "name": "--base_model_path",
            "type": str,
            "default": "",
            "help": "Path to the base model checkpoint to load from. Overrides config file if provided.",
        },
        {
            "name": "--init_critic_from_base_model",
            "action": "store_true",
            "default": False,
            "help": "Init critic from base model.",
        },
        {
            "name": "--use_upper_action",
            "action": "store_true",
            "default": False,
            "help": "Whether to use upper action in 29dof model.",
        },
        {
            "name": "--policy_class_name",
            "type": str,
            "default": None,
            "help": "Policy class name to use.",
        },
        {
            "name": "--init_std_upper",
            "action": "store_true",
            "default": False,
            "help": "Whether to init upper joint std.",
        }
    ]
    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    # args.sim_device_id = args.compute_device_id
    args.sim_device = args.rl_device
    # if args.sim_device=='cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path, obs_batch=None):
    """
    Export policy as JIT script.
    
    Args:
        actor_critic: The actor-critic model to export
        path: Output directory path
        obs_batch: Optional observation batch for sanity check before export.
                  If provided, will compare actor_critic.act_inference vs exporter output.
    """
    if hasattr(actor_critic, "base_model"):
        # Residual model with base_model
        exporter = PolicyExporterResidual(actor_critic)
        
        # Sanity check before export if obs_batch is provided
        if obs_batch is not None:
            print("Running sanity check before JIT export...")
            # Move exporter to same device as obs_batch for sanity check
            device = obs_batch.device
            exporter = exporter.to(device)
            if not sanity_check_export(actor_critic, exporter, obs_batch, name="JIT sanity check", atol=1e-5):
                print("WARNING: Sanity check failed! Exporter output differs from actor_critic.")
                print("Proceeding with export anyway, but results may be incorrect.")
            else:
                print("Sanity check passed! Exporter matches actor_critic.")
            # Move back to CPU for export (JIT export requires CPU)
            exporter = exporter.to("cpu")
        
        exporter.export(path)
    else:
        # Standard model
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


@torch.no_grad()
def sanity_check_export(actor_critic, exporter, obs_batch, name="jit check", atol=1e-5):
    """
    Compare actor_critic.act_inference(...) versus exporter(...) on the same env observations.
    Call this before scripting the exporter to validate obs/mask alignment.
    """
    actor_critic.eval()
    exporter.eval()

    # First check: Verify parameter independence
    print(f"[{name}] Checking parameter independence...")
    ref_params = {name: param for name, param in actor_critic.named_parameters()}
    exp_params = {name: param for name, param in exporter.named_parameters()}
    
    # Check for shared parameters by comparing data pointers
    shared_params = []
    for ref_name, ref_param in ref_params.items():
        # Try to find corresponding parameter in exporter
        # Note: exporter has different structure, so we check by traversing modules
        for exp_name, exp_param in exp_params.items():
            if ref_param.data_ptr() == exp_param.data_ptr():
                shared_params.append((ref_name, exp_name))
    
    if shared_params:
        print(f"  WARNING: Found {len(shared_params)} shared parameters!")
        for ref_name, exp_name in shared_params[:5]:  # Show first 5
            print(f"    {ref_name} <-> {exp_name}")
    else:
        print(f"  ✓ No shared parameters detected. Models are independent.")
    
    # Second check: Verify computation path independence
    # Add small perturbation to exporter parameters and see if output changes
    print(f"[{name}] Testing computation independence...")
    original_params = {}
    for name, param in exporter.named_parameters():
        original_params[name] = param.data.clone()
    
    # Perturb one parameter slightly
    first_param_name = list(exporter.named_parameters())[0][0]
    first_param = dict(exporter.named_parameters())[first_param_name]
    perturbation = torch.randn_like(first_param) * 1e-6
    first_param.data.add_(perturbation)
    
    a_exp_perturbed_output = exporter(obs_batch)
    # Handle tuple return (actions, force_residual_pred)
    if isinstance(a_exp_perturbed_output, tuple) and len(a_exp_perturbed_output) == 2:
        a_exp_perturbed, _ = a_exp_perturbed_output
    else:
        a_exp_perturbed = a_exp_perturbed_output
    
    # Restore original parameter
    first_param.data.copy_(original_params[first_param_name])
    
    # Check if perturbation changed output
    a_exp_restored_output = exporter(obs_batch)
    # Handle tuple return (actions, force_residual_pred)
    if isinstance(a_exp_restored_output, tuple) and len(a_exp_restored_output) == 2:
        a_exp_restored, _ = a_exp_restored_output
    else:
        a_exp_restored = a_exp_restored_output
    
    diff_perturbed = (a_exp_perturbed - a_exp_restored).abs().max().item()
    if diff_perturbed < 1e-8:
        print(f"  WARNING: Perturbing exporter parameters did NOT change output!")
        print(f"  This suggests exporter might be using actor_critic's computation path.")
    else:
        print(f"  ✓ Perturbation test passed. Exporter computation is independent.")
        print(f"    Perturbation effect: {diff_perturbed:.3e}")
    
    # Main comparison
    a_ref_output = actor_critic.act_inference(obs_batch)
    # Handle both tuple (actions, force_residual_pred) and single actions for backward compatibility
    if isinstance(a_ref_output, tuple) and len(a_ref_output) == 2:
        a_ref, _ = a_ref_output  # Extract actions, ignore force_residual_pred for comparison
    else:
        a_ref = a_ref_output
    
    a_exp_output = exporter(obs_batch)
    # Handle tuple return (actions, force_residual_pred)
    if isinstance(a_exp_output, tuple) and len(a_exp_output) == 2:
        a_exp, _ = a_exp_output  # Extract actions, ignore force_residual_pred for comparison
    else:
        a_exp = a_exp_output

    # Check if outputs are exactly the same
    is_exact_match = torch.equal(a_ref, a_exp)
    if is_exact_match:
        print(f"[{name}] WARNING: Outputs are EXACTLY identical (torch.equal=True).")
        print(f"  This is unusual for independent models due to floating-point precision.")
        print(f"  Possible explanations:")
        print(f"    1. Both models use identical computation graphs (good)")
        print(f"    2. Hidden parameter sharing or computation reuse (bad)")
        print(f"    3. PyTorch's deterministic mode producing identical results (acceptable)")
    else:
        print(f"[{name}] Outputs differ (as expected for independent models).")
    
    diff = (a_ref - a_exp).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    std_abs = diff.std().item()
    print(f"[{name}] Statistics: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, std_abs={std_abs:.3e}")
    print(f"  Output shapes: ref={a_ref.shape}, exp={a_exp.shape}")
    print(f"  Output ranges: ref=[{a_ref.min().item():.6f}, {a_ref.max().item():.6f}], exp=[{a_exp.min().item():.6f}, {a_exp.max().item():.6f}]")

    if max_abs > atol:
        b, d = torch.nonzero(diff == max_abs, as_tuple=True)
        if b.numel() > 0:
            b = int(b[0].item())
            d = int(d[0].item())
            print(f"  worst at batch={b}, dim={d}, ref={a_ref[b,d].item():.6f}, exp={a_exp[b,d].item():.6f}")
        return False
    return True


@torch.no_grad()
def sanity_check_onnx_export(jit_model, onnx_model_path, obs_batch, name="ONNX check", atol=1e-4):
    """
    Compare JIT model vs ONNX model outputs on the same observations.
    Requires onnxruntime to be installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"[{name}] WARNING: onnxruntime not installed. Skipping ONNX validation.")
        print(f"  Install with: pip install onnxruntime")
        return True
    
    jit_model.eval()
    
    # Get JIT output
    a_jit = jit_model(obs_batch)
    
    # Load ONNX model and run inference
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
        ort_inputs = {ort_session.get_inputs()[0].name: obs_batch.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        a_onnx = torch.from_numpy(ort_outputs[0])
        
        # Move to same device for comparison
        if a_jit.is_cuda:
            a_onnx = a_onnx.to(a_jit.device)
        
        diff = (a_jit - a_onnx).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        std_abs = diff.std().item()
        
        print(f"[{name}] Statistics: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, std_abs={std_abs:.3e}")
        print(f"  Output shapes: jit={a_jit.shape}, onnx={a_onnx.shape}")
        print(f"  Output ranges: jit=[{a_jit.min().item():.6f}, {a_jit.max().item():.6f}], onnx=[{a_onnx.min().item():.6f}, {a_onnx.max().item():.6f}]")
        
        if max_abs > atol:
            b, d = torch.nonzero(diff == max_abs, as_tuple=True)
            if b.numel() > 0:
                b = int(b[0].item())
                d = int(d[0].item())
                print(f"  worst at batch={b}, dim={d}, jit={a_jit[b,d].item():.6f}, onnx={a_onnx[b,d].item():.6f}")
            return False
        
        print(f"[{name}] ✓ ONNX export validation passed!")
        return True
        
    except Exception as e:
        print(f"[{name}] ERROR: Failed to run ONNX model: {e}")
        return False


# ----------------------------------------------------------------------
# JIT-Compatible Modules for Export
# ----------------------------------------------------------------------

def _replace_multihead_attention_with_onnx_compatible(module):
    """
    Recursively replace all nn.MultiheadAttention with ONNXCompatibleMultiheadAttention.
    This is necessary because nn.MultiheadAttention has AMP checks that cause ONNX export issues.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            # Get parameters from original attention
            embed_dim = child.embed_dim
            num_heads = child.num_heads
            batch_first = child.batch_first
            
            # Create ONNX-compatible replacement
            onnx_attn = ONNXCompatibleMultiheadAttention(embed_dim, num_heads, batch_first)
            
            # Copy weights from original attention
            # nn.MultiheadAttention stores weights in in_proj_weight and out_proj.weight
            # We need to split in_proj_weight into q, k, v
            if hasattr(child, 'in_proj_weight') and child.in_proj_weight is not None:
                in_proj = child.in_proj_weight  # (3*embed_dim, embed_dim)
                q_proj_w, k_proj_w, v_proj_w = in_proj.chunk(3, dim=0)
                onnx_attn.q_proj.weight.data.copy_(q_proj_w)
                onnx_attn.k_proj.weight.data.copy_(k_proj_w)
                onnx_attn.v_proj.weight.data.copy_(v_proj_w)
            
            # Copy biases if they exist (CRITICAL: this was missing and caused sanity check failure!)
            if hasattr(child, 'in_proj_bias') and child.in_proj_bias is not None:
                in_proj_bias = child.in_proj_bias  # (3*embed_dim,)
                q_proj_b, k_proj_b, v_proj_b = in_proj_bias.chunk(3, dim=0)
                # Replace bias parameters (they exist because we set bias=True)
                onnx_attn.q_proj.bias.data.copy_(q_proj_b)
                onnx_attn.k_proj.bias.data.copy_(k_proj_b)
                onnx_attn.v_proj.bias.data.copy_(v_proj_b)
            else:
                # No bias in original, zero out biases
                onnx_attn.q_proj.bias.data.zero_()
                onnx_attn.k_proj.bias.data.zero_()
                onnx_attn.v_proj.bias.data.zero_()
            
            if hasattr(child, 'out_proj') and hasattr(child.out_proj, 'weight'):
                onnx_attn.out_proj.weight.data.copy_(child.out_proj.weight.data)
                if hasattr(child.out_proj, 'bias') and child.out_proj.bias is not None:
                    onnx_attn.out_proj.bias.data.copy_(child.out_proj.bias)
                else:
                    onnx_attn.out_proj.bias.data.zero_()
            
            # Replace the module
            setattr(module, name, onnx_attn)
        else:
            # Recursively process children
            _replace_multihead_attention_with_onnx_compatible(child)


class ONNXCompatibleMultiheadAttention(nn.Module):
    """
    ONNX-compatible MultiheadAttention implementation.
    Avoids using nn.MultiheadAttention which has AMP checks (is_autocast_enabled)
    that cause UNKNOWN_SCALAR errors during ONNX export.
    """
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        # Linear projections for Q, K, V (bias will be added dynamically if needed)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(self.head_dim)))
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False, 
                attn_mask: Optional[torch.Tensor] = None):
        # Input shape: (B, T, D) if batch_first=True
        # ONNX FIX: Manual attention computation without AMP checks
        # JIT FIX: Add type annotations for all parameters
        B, T, D = query.shape
        
        # Project to Q, K, V (with bias if available)
        Q = self.q_proj(query)  # (B, T, D)
        K = self.k_proj(key)    # (B, T, D)
        V = self.v_proj(value)  # (B, T, D)
        
        # Reshape for multi-head: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) -> (B, 1, 1, T) for broadcasting
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values: (B, num_heads, T, T) @ (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        if need_weights:
            return output, attn_weights
        return output, None



class _ObsEncoderSepMLPExport(torch.nn.Module):
    """
    JIT-friendly version of ObsEncoderSepMLP.
    Replicates the logic: separate MLP for each history step, then attention pooling.
    FIXED: Uses enumerate() to iterate ModuleList, satisfying JIT constraints.
    """
    def __init__(self, mlp_modules, impute, one_step_obs_dims_indexes, history_length):
        super().__init__()
        # Convert ModuleDict (which JIT hates) to ModuleList
        self.mlp_modules = torch.nn.ModuleList(mlp_modules)
        
        # Buffers for constants
        self.register_buffer("one_step_obs_dims_indexes", one_step_obs_dims_indexes.clone())
        self.impute = torch.nn.Parameter(impute.clone(), requires_grad=False)
        self.history_length = history_length

    def forward(self, obs: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        # obs: [B, T * D] (flattened history)
        # obs_mask: [B, T * Num_Groups] (flattened mask)
        
        B_size = obs.shape[0]
        
        # 1. Reshape to [B*T, -1] for broadcasting
        obs = obs.reshape(B_size * self.history_length, -1)
        obs_mask = obs_mask.reshape(B_size * self.history_length, -1)
        
        # Debug: Check dimensions match
        expected_obs_dim = self.one_step_obs_dims_indexes.sum().item()
        actual_obs_dim = obs.shape[1]
        expected_mask_dim = self.one_step_obs_dims_indexes.shape[0]
        actual_mask_dim = obs_mask.shape[1]
        
        # Validate dimensions match expectations
        if actual_obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"ObsEncoder dimension mismatch: obs has {actual_obs_dim} dims but "
                f"one_step_obs_dims_indexes sum is {expected_obs_dim}. "
                f"This suggests the input obs doesn't match the encoder's expected dimensions."
            )
        if actual_mask_dim != expected_mask_dim:
            raise RuntimeError(
                f"ObsEncoder mask dimension mismatch: obs_mask has {actual_mask_dim} dims but "
                f"expected {expected_mask_dim} (num groups)."
            )

        # 2. Expand mask: [B*T, Num_Groups] -> [B*T, D] by repeating each group mask
        # self.one_step_obs_dims_indexes stores per-feature group index counts (same as training ObsEncoderSepMLP)
        expanded_obs_mask = obs_mask.repeat_interleave(self.one_step_obs_dims_indexes, dim=1)

        # 3. Apply Imputation
        # self.impute is [T, D] -> expand to [B, T, D] -> reshape to [B*T, D]
        impute_expanded = self.impute.unsqueeze(0).expand(B_size, -1, -1).reshape(B_size * self.history_length, -1)
        
        # Ensure impute_expanded matches obs dimension (safety check)
        if impute_expanded.shape[1] != obs.shape[1]:
            raise RuntimeError(
                f"Impute dimension mismatch: impute_expanded has {impute_expanded.shape[1]} dims "
                f"but obs has {obs.shape[1]} dims. impute.shape={self.impute.shape}"
            )
        
        # Logic: obs * mask + (1-mask) * impute
        obs = obs * expanded_obs_mask + (1.0 - expanded_obs_mask) * impute_expanded

        # 4. Reshape back to [B, T, D] to process each timestep
        obs = obs.reshape(B_size, self.history_length, -1)

        # 5. Process each timestep with its corresponding MLP
        # ONNX FIX: Always unroll to avoid dynamic if/for loops (no prim_if in graph)
        # Assume history_length is fixed at initialization time (typically 6)
        # Directly unroll all MLP calls to create a static graph
        encode_0 = self.mlp_modules[0](obs[:, 0, :])
        encode_1 = self.mlp_modules[1](obs[:, 1, :])
        encode_2 = self.mlp_modules[2](obs[:, 2, :])
        encode_3 = self.mlp_modules[3](obs[:, 3, :])
        encode_4 = self.mlp_modules[4](obs[:, 4, :])
        encode_5 = self.mlp_modules[5](obs[:, 5, :])
        encode = torch.stack([encode_0, encode_1, encode_2, encode_3, encode_4, encode_5], dim=1) 

        # 6. Attention / Softmax Aggregation
        # Last dim is the attention score
        score = encode[:, :, -1]  # [B, T]
        score = F.softmax(score, dim=-1).unsqueeze(-1)  # [B, T, 1]
        
        # Rest dims are the latent vector
        val = encode[:, :, :-1]  # [B, T, latent_dim]
        
        # Weighted sum
        fused = (val * score).sum(dim=1)  # [B, latent_dim]
        
        # Add sequence dim for Transformer [B, 1, latent_dim]
        return fused.unsqueeze(1)


class PolicyExporterResidual(torch.nn.Module):
    """
    JIT-Safe Policy exporter for MaskedActorCriticV2Residual.
    Strictly aligns with the 'act_inference' logic.
    """
    def __init__(self, actor_critic):
        super().__init__()
        
        # --- 1. CONFIGURATION ---
        self.actor_history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.one_step_obs_dims = actor_critic.one_step_obs_dims # Dict
        
        # Base Model Config
        self.base_model_actor_history_length = actor_critic.base_model.actor_history_length
        self.base_model_one_step_obs_dims = actor_critic.base_model.one_step_obs_dims

        # Indices
        self.register_buffer("lower_dof_indices", actor_critic.lower_dof_indices.clone())
        
        # --- 2. RESIDUAL MODEL COMPONENTS ---
        # Encoder: Convert to JIT-friendly version
        if hasattr(actor_critic.obs_encoder, "mlp_modules"):
            mlp_list = []
            for h in range(actor_critic.obs_encoder.history_length):
                mlp_list.append(copy.deepcopy(actor_critic.obs_encoder.mlp_modules[f"mlp_{h}"]))
            
            self.obs_encoder = _ObsEncoderSepMLPExport(
                mlp_list,
                actor_critic.obs_encoder.impute,
                actor_critic.obs_encoder.one_step_obs_dims_indexes,
                actor_critic.obs_encoder.history_length,
            )
        else:
            # If using MLP or other encoder types, ensure they are JIT compatible or wrap similarly
            self.obs_encoder = copy.deepcopy(actor_critic.obs_encoder)

        self.actor_transformer = copy.deepcopy(actor_critic.actor_transformer)
        # ONNX FIX: Replace MultiheadAttention with ONNX-compatible version
        _replace_multihead_attention_with_onnx_compatible(self.actor_transformer)
        self.actor = copy.deepcopy(actor_critic.actor)

        # --- 3. BASE MODEL COMPONENTS ---
        # Base Encoder
        if hasattr(actor_critic.base_model, "obs_encoder"):
            base_enc = actor_critic.base_model.obs_encoder
            if hasattr(base_enc, "mlp_modules"):
                mlp_list = []
                for h in range(base_enc.history_length):
                    mlp_list.append(copy.deepcopy(base_enc.mlp_modules[f"mlp_{h}"]))
                self.base_obs_encoder = _ObsEncoderSepMLPExport(
                    mlp_list,
                    base_enc.impute,
                    base_enc.one_step_obs_dims_indexes,
                    base_enc.history_length,
                )
            else:
                self.base_obs_encoder = copy.deepcopy(base_enc)
        else:
            self.base_obs_encoder = None
            
        self.base_actor_transformer = copy.deepcopy(actor_critic.base_model.actor_transformer)
        # ONNX FIX: Replace MultiheadAttention with ONNX-compatible version
        _replace_multihead_attention_with_onnx_compatible(self.base_actor_transformer)
        self.base_actor = copy.deepcopy(actor_critic.base_model.actor)

        # --- 4. PRE-CALCULATE BASE MODEL MAPPING (The Critical Part) ---
        # The training code extracts base model observations from residual observations dynamically.
        # We must pre-calculate indices to do this in JIT without dictionary iteration.

        # A. Calculate offsets for Residual Model (Source)
        residual_offsets = {}
        curr = 0
        for k, v in self.one_step_obs_dims.items():
            residual_offsets[k] = (curr, curr + v)
            curr += v
        
        # B. Build Index Mapping for Observation History
        # We process history as [B, T, D]. We need to gather D_base from D_residual.
        base_obs_gather_indices_list = []
        
        for obs_name in self.base_model_one_step_obs_dims.keys():
            if obs_name not in residual_offsets:
                continue # Should not happen if config is correct
                
            start, end = residual_offsets[obs_name]
            indices = torch.arange(start, end, dtype=torch.long)
            
            # **CRITICAL**: Handle 'action_actual' special case.
            # In training code: source_block = source_block[..., self.lower_dof_indices]
            if obs_name == "action_actual":
                # 'indices' currently points to the full action vector in the residual obs.
                # We only want the lower body subset.
                # lower_dof_indices are relative to the action vector start.
                subset_indices = indices[self.lower_dof_indices.cpu()]
                base_obs_gather_indices_list.append(subset_indices)
            else:
                base_obs_gather_indices_list.append(indices)
        
        if len(base_obs_gather_indices_list) > 0:
            self.register_buffer("base_obs_gather_indices", torch.cat(base_obs_gather_indices_list))
        else:
            self.register_buffer("base_obs_gather_indices", torch.zeros(0, dtype=torch.long))

        # C. Build Index Mapping for Mask
        # Residual Mask: [B, T, Num_Residual_Groups]
        # Base Mask: [B, T, Num_Base_Groups]
        residual_keys = list(self.one_step_obs_dims.keys())
        base_keys = list(self.base_model_one_step_obs_dims.keys())
        
        base_mask_gather_indices_list = []
        for k in base_keys:
            if k in residual_keys:
                idx = residual_keys.index(k)
                base_mask_gather_indices_list.append(idx)
        
        self.register_buffer("base_mask_gather_indices", torch.tensor(base_mask_gather_indices_list, dtype=torch.long))

        # --- 5. ADAPTATION MODULE ---
        self.use_adaptation = getattr(actor_critic, "use_adaptation", False)
        if self.use_adaptation:
            self.adaptation_encoder_module = copy.deepcopy(actor_critic.adaptation_encoder_module)
            # CRITICAL: Also copy adaptation_decoder_module to generate force_residual_pred
            self.adaptation_decoder_module = copy.deepcopy(actor_critic.adaptation_decoder_module)
            # Store adaptation_history_length from actor_critic
            self.adaptation_history_length = getattr(actor_critic, "adaptation_history_length", 1)
            # Need lengths to manually apply masking in JIT/ONNX (registered as buffer for ONNX compatibility)
            self.register_buffer("obs_dim_lengths", torch.tensor(list(self.one_step_obs_dims.values()), dtype=torch.long))
            # Get adaptation_latent_dim from the module output
            # Create a dummy input to infer output dimension (must be on same device as module)
            # CRITICAL: Adaptation encoder expects multi-frame input: num_one_step_obs * adaptation_history_length
            module_device = next(self.adaptation_encoder_module.parameters()).device if list(self.adaptation_encoder_module.parameters()) else torch.device("cpu")
            adaptation_input_dim = sum(self.one_step_obs_dims.values()) * self.adaptation_history_length
            dummy_obs = torch.zeros(1, adaptation_input_dim, device=module_device)
            with torch.no_grad():
                dummy_output = self.adaptation_encoder_module(dummy_obs)
                adaptation_latent_dim = dummy_output.shape[-1]
            
            # Extract wrist_forces pred_slice from adaptation_target_layout for force_residual_pred extraction
            # This matches the logic in act_inference() method
            wrist_forces_pred_slice = None
            adaptation_target_layout = getattr(actor_critic, "adaptation_target_layout", [])
            for layout in adaptation_target_layout:
                if layout.get("name") == "wrist_forces":
                    wrist_forces_pred_slice = layout.get("pred_slice")
                    break
            
            if wrist_forces_pred_slice is not None:
                # Store as buffer for JIT/ONNX compatibility
                self.register_buffer("wrist_forces_pred_start", torch.tensor(wrist_forces_pred_slice[0], dtype=torch.long))
                self.register_buffer("wrist_forces_pred_end", torch.tensor(wrist_forces_pred_slice[1], dtype=torch.long))
                self.has_wrist_forces = True
            else:
                # If wrist_forces not in adaptation groups, set to zero
                self.register_buffer("wrist_forces_pred_start", torch.tensor(0, dtype=torch.long))
                self.register_buffer("wrist_forces_pred_end", torch.tensor(0, dtype=torch.long))
                self.has_wrist_forces = False
        else:
            # ONNX FIX: Create a dummy module that outputs zeros instead of None
            # This allows forward() to always call the same code path (no dynamic if)
            adaptation_latent_dim = getattr(actor_critic, "adaptation_latent_dim", 0)
            if adaptation_latent_dim == 0:
                # If dimension unknown, try to infer from actor input
                # actor input = transformer_latent_dim + adaptation_latent_dim
                # Try to get it from actor's first layer if possible
                try:
                    actor_first_layer = actor_critic.actor[0] if hasattr(actor_critic.actor, '__getitem__') else None
                    if actor_first_layer is not None and hasattr(actor_first_layer, 'in_features'):
                        transformer_latent_dim = getattr(actor_critic, "transformer_latent_dim", 256)
                        adaptation_latent_dim = actor_first_layer.in_features - transformer_latent_dim
                except:
                    adaptation_latent_dim = 128  # fallback default
            
            # Create a dummy module that always outputs zeros
            class ZeroAdaptationEncoder(nn.Module):
                def __init__(self, output_dim):
                    super().__init__()
                    self.output_dim = output_dim
                def forward(self, x):
                    return torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
            
            class ZeroAdaptationDecoder(nn.Module):
                def __init__(self, output_dim):
                    super().__init__()
                    self.output_dim = output_dim
                def forward(self, x):
                    return torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
            
            self.adaptation_encoder_module = ZeroAdaptationEncoder(adaptation_latent_dim)
            # Create dummy decoder that outputs zeros (for force_residual_pred)
            # Default adaptation_num_outputs if not available (wrist_forces=6, wrist_pos=6, base_lin_vel=3 = 15)
            adaptation_num_outputs = getattr(actor_critic, "adaptation_num_outputs", 15)
            self.adaptation_decoder_module = ZeroAdaptationDecoder(adaptation_num_outputs)
            # Still need obs_dim_lengths for _apply_adaptation_mask
            self.register_buffer("obs_dim_lengths", torch.tensor(list(self.one_step_obs_dims.values()), dtype=torch.long))
            # No wrist_forces in adaptation, set to zero
            self.register_buffer("wrist_forces_pred_start", torch.tensor(0, dtype=torch.long))
            self.register_buffer("wrist_forces_pred_end", torch.tensor(0, dtype=torch.long))
            self.has_wrist_forces = False
        
        # Register adaptation_latent_dim as buffer for ONNX compatibility (static graph)
        self.register_buffer("adaptation_latent_dim", torch.tensor(adaptation_latent_dim, dtype=torch.long))

    def _split_actor_obs(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split observation batch into history and mask."""
        num_obs_groups = len(self.one_step_obs_dims)
        mask_len = self.actor_history_length * num_obs_groups
        
        # Standard splitting
        obs_mask = obs_batch[:, -mask_len:]
        obs_history = obs_batch[:, :-mask_len]
        return obs_history, obs_mask
    
    def _get_base_model_inputs(self, obs_history_flat: torch.Tensor, obs_mask_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract base model inputs using pre-calculated gather indices.
        Replicates 'get_base_model_obs_history' from training code.
        """
        B = obs_history_flat.shape[0]
        
        # 1. View as [B, T, D]
        obs_history_3d = obs_history_flat.view(B, self.actor_history_length, -1)
        
        # 2. Gather specific features for base model -> [B, T, D_base]
        # This implicitly handles the 'action_actual' subsetting because we pre-calculated the indices.
        base_obs_3d = torch.index_select(obs_history_3d, 2, self.base_obs_gather_indices)
        
        # 3. View mask as [B, T, Num_Groups]
        obs_mask_3d = obs_mask_flat.view(B, self.actor_history_length, -1)
        
        # 4. Gather specific masks for base model
        base_mask_3d = torch.index_select(obs_mask_3d, 2, self.base_mask_gather_indices)
        
        # 5. Flatten back to [B, T*D] and [B, T*Num_Groups] for Encoder input
        base_obs_flat = base_obs_3d.reshape(B, -1)
        base_mask_flat = base_mask_3d.reshape(B, -1)
        
        return base_obs_flat, base_mask_flat

    def _apply_adaptation_mask(self, current_obs: torch.Tensor, current_mask: torch.Tensor) -> torch.Tensor:
        """
        ONNX-safe implementation of _apply_actor_mask for adaptation module.
        FIXED: No Python loops, no .item() calls - pure tensor operations.
        """
        # ONNX FIX: Expand group mask to feature mask without loops
        # current_mask: [B, num_groups]
        # obs_dim_lengths: [num_groups] - lengths of each group
        # We need to expand each group mask to its corresponding feature dimensions
        
        # Create expanded mask: [B, num_groups] -> [B, total_features]
        # Use repeat_interleave with pre-computed repeats tensor
        expanded_mask = current_mask.repeat_interleave(self.obs_dim_lengths, dim=1)
        
        # Apply mask element-wise
        masked_obs = current_obs * expanded_mask
        
        return masked_obs

    def _build_adaptation_input(self, obs_history_flat: torch.Tensor, obs_mask_flat: torch.Tensor) -> torch.Tensor:
        """
        Match training-time adaptation logic in MaskedActorCriticV2Residual._run_adaptation_modules:
        - take last N frames (N = min(adaptation_history_length, actor_history_length))
        - apply group mask per frame
        - concatenate masked frames into [B, N * num_one_step_obs]
        """
        B = obs_history_flat.shape[0]
        T = self.actor_history_length
        G = len(self.one_step_obs_dims)
        # reshape
        obs_seq = obs_history_flat.view(B, T, -1)          # [B, T, D]
        mask_seq = obs_mask_flat.view(B, T, G)            # [B, T, G]
        n = self.adaptation_history_length
        if n > T:
            n = T
        # take last n frames
        obs_tail = obs_seq[:, T - n : T, :]               # [B, n, D]
        mask_tail = mask_seq[:, T - n : T, :]             # [B, n, G]
        # apply mask per frame
        masked_frames: List[torch.Tensor] = []
        for t in range(n):
            masked_frames.append(self._apply_adaptation_mask(obs_tail[:, t, :], mask_tail[:, t, :]))
        return torch.cat(masked_frames, dim=1)            # [B, n * D]

    def forward(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Split Input
        obs_history_flat, obs_mask_flat = self._split_actor_obs(obs_batch)
        
        # 2. Base Model Inference
        # Extract inputs matching base model's expectations
        base_obs_flat, base_mask_flat = self._get_base_model_inputs(obs_history_flat, obs_mask_flat)
        
        # Base Forward Pass
        base_embedding = self.base_obs_encoder(base_obs_flat, base_mask_flat)
        # ONNX FIX: TransformerEncoder returns (B, embed_dim), not (B, 1, embed_dim), so no squeeze needed
        base_token = self.base_actor_transformer(base_embedding)
        base_action = self.base_actor(base_token)
        
        # 3. Residual Model Inference
        # Residual Forward Pass (uses full history and mask)
        obs_embedding = self.obs_encoder(obs_history_flat, obs_mask_flat)
        # ONNX FIX: TransformerEncoder returns (B, embed_dim), not (B, 1, embed_dim), so no squeeze needed
        actor_token = self.actor_transformer(obs_embedding)
        
        # 4. Adaptation Logic (ONNX FIX: Always compute same path, no dynamic if)
        # Build multi-frame adaptation input using _build_adaptation_input
        # This method handles taking last N frames and applying mask per frame
        # Always apply mask and compute adaptation_latent (no if branch)
        # If use_adaptation=False, adaptation_encoder_module is a ZeroAdaptationEncoder that outputs zeros
        masked_obs = self._build_adaptation_input(obs_history_flat, obs_mask_flat)
        adaptation_latent = self.adaptation_encoder_module(masked_obs)
        
        # CRITICAL: Compute adaptation_predictions for force_residual_pred extraction
        # This matches the logic in act_inference() method
        adaptation_predictions = self.adaptation_decoder_module(adaptation_latent)
        
        # Extract wrist_forces from adaptation_predictions (force_residual_pred)
        # This matches the logic in act_inference() lines 800-824
        # JIT FIX: Use tensor indexing directly (JIT can handle this)
        B = adaptation_predictions.shape[0]
        # Extract slice using tensor indices (JIT-compatible)
        # If start==end, this will return empty tensor, we'll handle that below
        force_residual_pred_slice = adaptation_predictions[:, self.wrist_forces_pred_start:self.wrist_forces_pred_end]
        
        # Ensure output is always [B, 6] (wrist_forces dimension)
        # Use conditional assignment that JIT can optimize
        slice_dim = force_residual_pred_slice.shape[1]
        # Create zero tensor as fallback
        force_residual_pred_zero = torch.zeros(B, 6, device=adaptation_predictions.device, dtype=adaptation_predictions.dtype)
        
        # If slice has correct dimension (6), use it; otherwise use zero
        # JIT can handle this conditional if we use torch.where or direct assignment
        if slice_dim == 6:
            force_residual_pred = force_residual_pred_slice
        elif slice_dim == 0:
            force_residual_pred = force_residual_pred_zero
        else:
            # Handle edge cases: pad or truncate to 6
            if slice_dim > 6:
                force_residual_pred = force_residual_pred_slice[:, :6]
            else:
                # Pad with zeros
                padding = torch.zeros(B, 6 - slice_dim, device=adaptation_predictions.device, dtype=adaptation_predictions.dtype)
                force_residual_pred = torch.cat([force_residual_pred_slice, padding], dim=1)
        
        # Always concatenate (static graph)
        actor_input = torch.cat((actor_token, adaptation_latent), dim=-1)
            
        # 5. Get Residual Action
        residual_action = self.actor(actor_input)
        
        # 6. Combine Actions
        # Strategy: Final = Residual. 
        # Then Add Base to Lower indices.
        # ONNX FIX: Use manual expansion instead of scatter/index_put to avoid type inference issues
        # shape of base_action: [B, num_lower]
        # shape of residual_action: [B, num_total]
        
        # Ensure base_action has explicit type information matching residual_action
        # This is critical for ONNX type inference
        base_action_typed = base_action.to(dtype=residual_action.dtype, device=residual_action.device)
        
        # Create full-size tensor initialized to zero with explicit dtype/device
        B = residual_action.shape[0]
        num_total = residual_action.shape[1]
        base_action_full = torch.zeros(B, num_total, dtype=residual_action.dtype, device=residual_action.device)
        
        # ONNX FIX: Use advanced indexing with explicit type casting
        # Create index arrays: lower_dof_indices shape [num_lower]
        # We need to set base_action_full[:, lower_dof_indices] = base_action
        # But ONNX may have issues with advanced indexing, so use expand + scatter with explicit types
        
        # Expand indices for batch: [num_lower] -> [B, num_lower]
        indices_expanded = self.lower_dof_indices.unsqueeze(0).expand(B, -1)
        
        # Ensure all tensors have explicit, matching types
        # Use scatter but ensure base_action_typed has the same scalar type as residual_action
        # Clone to ensure we have a fresh tensor with proper type info
        base_action_cloned = base_action_typed.clone()
        
        # Use scatter with explicit source tensor that has known type
        # The key is ensuring the source tensor (base_action_cloned) has explicit type
        base_action_full = base_action_full.scatter(1, indices_expanded, base_action_cloned)
        
        # Add residual and base (both non-inplace)
        final_action = residual_action + base_action_full
        
        # Return tuple (actions, force_residual_pred) to match act_inference() signature
        # JIT supports returning tuples
        return final_action, force_residual_pred

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        #path = os.path.join(path, "3291456fnl-15000.pt")
        path = os.path.join(path, "0.pt")
        # Ensure CPU
        self.to("cpu")
        # Scripting   
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        print(f"Exported residual policy as JIT script to: {path}")