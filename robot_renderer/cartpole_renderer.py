import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch
from robot_renderer.base_renderer import BaseRoboRenderer
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, look_at_view_transform
)


class CartpoleRenderer(BaseRoboRenderer):
    def __init__(self, img_height=84, img_width=84, device="cuda"):
        super().__init__(os.path.join(proj_dir, "assets/cartpole.xml"), img_height=img_height, img_width=img_width, device=device)

    def get_camera(self, qpos, camera_id):
        batch_size = qpos.shape[0]

        R, T = look_at_view_transform(
            eye=torch.tile(torch.tensor([0, -2.8, 1.2], dtype=torch.float32, device=self.device), (batch_size, 1)),
            at=torch.tile(torch.tensor([0, 0, 1.2], dtype=torch.float32, device=self.device), (batch_size, 1)),
            up=torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device), (batch_size, 1)),
            device=self.device)
                
        return FoVPerspectiveCameras(znear=0.01, zfar=100, fov=90, R=R, T=T, device=self.device)

    def get_raster_settings(self, camera_id):
        if camera_id == 0:
            # img size for training 
            return RasterizationSettings(image_size=84)
        else:
            # img size for rendering
            return RasterizationSettings(image_size=256)

    def get_renderer(self, qpos, camera_id):

        cameras = self.get_camera(qpos, camera_id)
        raster_settings = self.get_raster_settings(camera_id)

        return MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings),
                shader=SoftPhongShader(cameras=cameras, device=self.device)
            )