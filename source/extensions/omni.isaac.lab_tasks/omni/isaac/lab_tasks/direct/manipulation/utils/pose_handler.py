import numpy as np
import torch

from pose_handler_config import PoseHandlerConfig

from omni.isaac.utils.math_utils import quat_from_euler_xyz, quat_unique

class PoseHandler:
    def __init__(self,
                 env_ids,
                 config: PoseHandlerConfig,
                 device='cuda'):


        self.env_ids = env_ids
        self.device = device
        self.config = config

        self.target_pose = torch.zeros(0, 7, device=self.device)

    def resample_pose(self):
        self._resample_position_from_hollow_sphere()
        self._resample_orientation()

    def _resample_position_from_hollow_sphere(self):

        random = torch.empty(self.env_ids, device=self.device)

        theta = random.uniform_(0, torch.pi)
        phi = random.uniform_(0, 2 * torch.pi)

        r = torch.sqrt(random.uniform_(self.config.range_radius[0] ** 3, self.config.range_radius[1] ** 3))

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        # add to target pose
        self.target_pose[:, 0] = x
        self.target_pose[:, 1] = y
        self.target_pose[:, 2] = z
    def _resample_orientation(self):
        # euler_angles = torch.zeros_like(self.pose_command_b[:, :3])
        random = torch.empty(self.env_ids, device=self.device)
        euler_x = random.uniform_(*self.config.range_euler['roll'])
        euler_y = random.uniform_(*self.config.range_euler['pitch'])
        euler_z = random.uniform_(*self.config.range_euler['yaw'])

        quat = quat_from_euler_xyz(euler_x[:, 0], euler_y[:, 1], euler_z[:, 2])
        # make sure the quaternion has real part as positive
        self.target_pose[:, 3:] = quat_unique(quat) if self.config.make_quat_unique else quat

