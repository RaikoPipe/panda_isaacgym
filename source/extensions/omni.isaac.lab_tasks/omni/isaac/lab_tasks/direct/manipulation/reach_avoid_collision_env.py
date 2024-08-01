# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Sequence

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom
from gymnasium import spaces
import numpy as np

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sim.spawners.shapes import spawn_sphere
from omni.isaac.lab.envs.common import VecEnvObs, VecEnvStepReturn

from omni.isaac.lab.envs import mdp

import gymnasium as gym


@configclass
class ReachAvoidCollisionEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    num_actions = 9
    num_observations = 23
    num_states = 0
    num_goal_observations = 7

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # obstacles
    obstacles = {}

    for i in range(3):
        obstacles[f"obstacle_{i}"] = sim_utils.MeshSphereCfg(
            radius = 0.05,
            visible=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
        )



    # cabinet
    # cabinet = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/Cabinet",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0, 0.4),
    #         rot=(0.1, 0.0, 0.0, 0.0),
    #         joint_pos={
    #             "door_left_joint": 0.0,
    #             "door_right_joint": 0.0,
    #             "drawer_bottom_joint": 0.0,
    #             "drawer_top_joint": 0.0,
    #         },
    #     ),
    #     actuators={
    #         "drawers": ImplicitActuatorCfg(
    #             joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
    #             effort_limit=87.0,
    #             velocity_limit=100.0,
    #             stiffness=10.0,
    #             damping=1.0,
    #         ),
    #         "doors": ImplicitActuatorCfg(
    #             joint_names_expr=["door_left_joint", "door_right_joint"],
    #             effort_limit=87.0,
    #             velocity_limit=100.0,
    #             stiffness=10.0,
    #             damping=2.5,
    #         ),
    #     },
    # )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 2.0
    rot_reward_scale = 0.5
    around_handle_reward_scale = 0.0
    open_reward_scale = 7.5
    action_penalty_scale = 0.01
    finger_dist_reward_scale = 0.0
    finger_close_reward_scale = 10.0


class ReachAvoidCollisionEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ReachAvoidCollisionEnvCfg

    def __init__(self, cfg: ReachAvoidCollisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()

        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))
        # todo: replace drawer with goal pose

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )


        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.goals = torch.zeros((self.num_envs, self.cfg.num_goal_observations), device=self.device)



    def _setup_scene(self):
        # extract scene entities
        self._robot = Articulation(self.cfg.robot)
        #self._cabinet = Articulation(self.cfg.cabinet)
        # spawn obstacles
        self._obstacles = {name: spawn_sphere(prim_path=f"/World/Sphere/{name}", cfg=cfg) for name, cfg in self.cfg.obstacles.items()}
        self.scene.articulations["robot"] = self._robot
        #self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_gym_env_spaces(self):
        """Override to set custom observation space"""
        # observation, info = self.reset()

        self.single_observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.num_observations,),
                    dtype=np.float32),
                desired_goal=spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.num_goal_observations,),
                    dtype=np.float32),
                achieved_goal=spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.num_goal_observations,),
                    dtype=np.float32),
            )
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.num_actions,))

        # batch the spaces for the vectorized environment
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # todo: adjust termination condition: truncated on timeout and obstacle collision; terminated on success
        terminated = self._robot.data.joint_pos[:, 3] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        rewards = self._compute_rewards()
        rewards = torch.tensor(rewards, dtype=torch.float32)

        return rewards

    def reset(self):
        observation, extras = super().reset()

        extras = {"is_success": self._is_success(observation["achieved_goal"], observation["desired_goal"])}

        return observation, extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action.clone())
        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["observation"] = self._observation_noise_model.apply(self.obs_buf["observation"])

        # todo: adjust extras

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras






    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # set new goal for the environment
        self._set_new_goals(env_ids)

        # cabinet state
        # zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        # self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        # self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # get task and robot observations

        # robot observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        # task observations
        # todo: add collision information as task obs

        dof_pos_scaled = (
                2.0
                * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
                / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
                - 1.0
        )

        robot_grasp_pose = self._get_achieved_goals()
        to_target = self.goals - robot_grasp_pose

        # obs = torch.cat(
        #     (
        #         dof_pos_scaled,
        #         self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
        #         to_target,
        #     ),
        #     dim=-1,
        # )

        obs = torch.cat(
            (joint_pos,
             joint_vel,
             to_target),
            dim=-1,
        )

        return {"observation": torch.clamp(obs, -5.0, 5.0),
                "desired_goal": self.goals,
                "achieved_goal": robot_grasp_pose}


    # auxiliary methods

    def _set_new_goals(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Set new goals for the given environment indices"""
        self.goals[env_ids] = self._sample_goal()

    def _sample_goal(self) -> torch.Tensor:
        """Sample random goal pose"""
        # todo: get random position and quaternion
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

    def _get_achieved_goals(self) -> torch.Tensor:
        return torch.cat((self.robot_grasp_pos, self.robot_grasp_rot), dim=-1)

    def _get_goals(self) -> torch.Tensor:
        return self.goals

    def _is_success(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """Compute the success of the achieved goal with respect to the desired goal"""

        # Separate positions and quaternions
        pose_positions = achieved_goal[:, :3]  # shape (N, 3)
        goal_positions = desired_goal[:, :3]  # shape (N, 3)

        # Calculate the L2 norm for positions
        position_distances = torch.norm(pose_positions - goal_positions, dim=1)  # shape (N,)

        pose_quaternions = achieved_goal[:, 3:]  # shape (N, 4)
        goal_quaternions = desired_goal[:, 3:]  # shape (N, 4)

        # Calculate the L2 norm for quaternions
        quaternion_distances = torch.norm(pose_quaternions - goal_quaternions, dim=1)  # shape (N,)

        # Overall L2 norm distance
        total_distances = position_distances + quaternion_distances

        return torch.where(total_distances < 0.05, torch.ones_like(total_distances), torch.zeros_like(total_distances))


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids]
        )

    def _compute_rewards(self):
        return self._is_success(self._get_achieved_goals(), self._get_goals())

    def _compute_rewards_old(
            self,
            actions,
            cabinet_dof_pos,
            franka_grasp_pos,
            drawer_grasp_pos,
            franka_grasp_rot,
            drawer_grasp_rot,
            franka_lfinger_pos,
            franka_rfinger_pos,
            gripper_forward_axis,
            drawer_inward_axis,
            gripper_up_axis,
            drawer_up_axis,
            num_envs,
            dist_reward_scale,
            rot_reward_scale,
            around_handle_reward_scale,
            open_reward_scale,
            finger_dist_reward_scale,
            action_penalty_scale,
            joint_positions,
            finger_close_reward_scale,
    ):
        """Only kept for reference"""
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2], around_handle_reward + 0.5, around_handle_reward
            ),
            around_handle_reward,
        )
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward,
            ),
            finger_dist_reward,
        )

        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward
        )

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions ** 2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = (
                dist_reward_scale * dist_reward
                + rot_reward_scale * rot_reward
                + around_handle_reward_scale * around_handle_reward
                + open_reward_scale * open_reward
                + finger_dist_reward_scale * finger_dist_reward
                - action_penalty_scale * action_penalty
                + finger_close_reward * finger_close_reward_scale
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "around_handle_reward": (around_handle_reward_scale * around_handle_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "finger_dist_reward": (finger_dist_reward_scale * finger_dist_reward).mean(),
            "action_penalty": (action_penalty_scale * action_penalty).mean(),
            "finger_close_reward": (finger_close_reward * finger_close_reward_scale).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        return rewards

    def _compute_grasp_transforms(
            self,
            hand_rot,
            hand_pos,
            franka_local_grasp_rot,
            franka_local_grasp_pos,
    ):
        # todo: this function is probably for translating the grasp pose from the robot's local frame to the global frame
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos


def _get_randomized_position_near_robot(seed):
    # randomize the position of the robot
    random = torch.Generator(device="cuda")
    random.manual_seed(seed)
    position = torch.rand((3,), generator=random) * 0.1
    position[0] = position[0] - 0.05
    position[1] = position[1] - 0.05
    position[2] = position[2] + 0.05
    return position
