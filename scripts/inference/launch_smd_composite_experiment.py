"""
MIT License

Copyright (c) 2025 Jinhao Liang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.
import argparse
import os
import pickle
from datetime import datetime
import time
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from typing import Tuple, List
import concurrent.futures
import multiprocessing as mp


# Project includes.
from smd.common.experiments import MultiAgentPlanningExperimentConfig
from smd.common.experiments.experiment_utils import *
from smd.config.smd_params import SMDParams as params
from inference_multi_agent import run_multi_agent_trial
from launch_multi_agent_experiment import run_multi_agent_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run multi-agent planning experiments.")
    parser.add_argument("--start_index", type=int, default=0, help="The starting index for samples.")
    parser.add_argument("--end_index", type=int, default=1, help="The ending index for samples (exclusive).")
    parser.add_argument("--save_path", type=str, default="results_test", help="The base directory to save results.")
    parser.add_argument("--agents_max_speeds", type=float, default=0.05, help="Max speed per agent for projection.")
    parser.add_argument("--rho", type=float, default=5.0, help="Initial ALM rho.")
    parser.add_argument("--rho_factor", type=float, default=1.05, help="Multiplicative factor to increase rho each ALM iter.")
    parser.add_argument("--alm_iteration", type=int, default=100, help="Max ALM iterations.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Convergence tolerance for ALM.")
    parser.add_argument("--projection_step", type=int, nargs='+', default=[15, 5], help="Projection steps used during sampling.")
    parser.add_argument("--runtime_limit", type=int, default=100000, help="Runtime limit for the experiment.")
    parser.add_argument("--map_name", type=str, default='instances_simple', help="Runtime limit for the experiment.")
    parser.add_argument("--experiment_instance_names", type=str, nargs='+', 
                            default=[
                                "EnvEmptyNoWait2DRobotCompositeThreePlanarDiskRandom",
                                # "EnvEmptyNoWait2DRobotCompositeSixPlanarDiskRandom",
                                # "EnvEmptyNoWait2DRobotCompositeNinePlanarDiskRandom"
                            ],
                            help="List of experiment instance names to run.")

    
    args = parser.parse_args()

    proj_params = {
        'agents_max_speeds': args.agents_max_speeds,
        'rho': args.rho,
        'rho_factor': args.rho_factor,
        'alm_iteration': args.alm_iteration,
        'tolerance': args.tolerance,
        'projection_step': args.projection_step,
    }

    stagger_start_time_dt = 0
    runtime_limit = args.runtime_limit
    num_trials_per_combination = 1
    render_animation = True

    

    for sample_idx in range(args.start_index, args.end_index):
        for instance_name in args.experiment_instance_names:
            experiment_config = MultiAgentPlanningExperimentConfig()

            experiment_config.instance_idx = sample_idx
            experiment_config.map_name = args.map_name
            experiment_config.res_base_dir = args.save_path

            experiment_config.num_agents_l = []
            if "two" in instance_name.lower():
                experiment_config.num_agents_l = [2]
            elif "three" in instance_name.lower():
                experiment_config.num_agents_l = [3]
                init_traj4proj = pickle.load(open(f'../../init4proj_data/{map_name}_init4proj_agent_3.pkl', 'rb'))
            elif "six" in instance_name.lower():
                experiment_config.num_agents_l = [6]
                init_traj4proj = pickle.load(open(f'../../init4proj_data/{map_name}_init4proj_agent_6.pkl', 'rb'))
            elif "nine" in instance_name.lower():
                experiment_config.num_agents_l = [9]
                init_traj4proj = pickle.load(open(f'../../init4proj_data/{map_name}_init4proj_agent_9.pkl', 'rb'))

            

            experiment_config.instance_name = instance_name
            experiment_config.init_traj4proj = init_traj4proj[sample_idx]
            experiment_config.proj_params = proj_params

            experiment_config.stagger_start_time_dt = stagger_start_time_dt
            experiment_config.multi_agent_planner_class_l = ["SMDComposite"]
            experiment_config.single_agent_planner_class = "SMDEnsemble"
            experiment_config.runtime_limit = runtime_limit
            experiment_config.num_trials_per_combination = num_trials_per_combination
            experiment_config.render_animation = render_animation
        
            run_multi_agent_experiment(experiment_config)
