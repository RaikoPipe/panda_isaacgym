# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .reach_avoid_collision_env import ReachAvoidCollisionEnv, ReachAvoidCollisionEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Reach-Franka-Avoid-Collision-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.manipulation:ReachAvoidCollisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ReachAvoidCollisionEnvCfg,
        #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        #"rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_tqc_cfg.yaml",
    },
)
