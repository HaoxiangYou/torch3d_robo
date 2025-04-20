import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch
from robot_renderer.base_renderer import BaseRoboRenderer
from utils.background_utils import create_sky_plane, create_floor
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, look_at_view_transform
)

class AntRenderer(BaseRoboRenderer):
    def __init__(self, img_height=84, img_width=84, device="cuda"):
        super().__init__(os.path.join(proj_dir, "assets/xmls/ant.xml"), img_height=img_height, img_width=img_width, device=device)

    def build_background(self):
        self.raw_meshes["background"] = [create_floor(x_dim=200, y_dim=40, device=self.device, center=[50, 0, 0]),
                                        create_sky_plane(center=[50, 20, 0], z_dim=100, x_dim=200, device=self.device)]

    def get_camera(self, qpos, camera_id):
        qpos = torch.atleast_2d(qpos)
        batch_size = qpos.shape[0]
        R, T = look_at_view_transform(
            eye=torch.tile(torch.tensor([0.5, -2, 1], dtype=torch.float32, device=self.device), (batch_size, 1)) + qpos[:,:3],
            at=torch.tile(torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device), (batch_size, 1)) + qpos[:, :3],
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
        
    def get_lights(self, qpos):
        qpos = torch.atleast_2d(qpos)
        batch_size = qpos.shape[0]
        
        light_position = qpos[:, :3] + torch.tensor([[0.0, 0.0, 6.0]], device=self.device).expand(batch_size, -1)

        # Define the light parameters
        ambient_color = torch.tensor([[0.4, 0.4, 0.4]], device=self.device).expand(batch_size, -1)   # Ambient light color
        diffuse_color = torch.tensor([[0.8, 0.8, 0.8]], device=self.device).expand(batch_size, -1)   # Diffuse light color
        specular_color = torch.tensor([[0.3, 0.3, 0.3]], device=self.device) .expand(batch_size, -1) # Specular highlights

        # Create the PointLight object in PyTorch3D
        return PointLights(
            device=self.device,
            location=light_position,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color
        )
        
    def get_transform(self, qpos):
        qpos = torch.atleast_2d(qpos)
        return self.chain.forward_kinematics(qpos[:, 7:], root_pos=qpos[:, :3], root_quat=qpos[:, 3:7])

    def get_renderer(self, qpos, camera_id):
        cameras = self.get_camera(qpos, camera_id)
        raster_settings = self.get_raster_settings(camera_id)
        lights = self.get_lights(qpos)

        return MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings),
                shader=SoftPhongShader(cameras=cameras, device=self.device, lights=lights)
            )