import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
from kinematics.load_kinematic_trees_from_mjcf import build_chain_from_mjcf
from kinematics.load_meshes_from_kinematic_trees import build_meshes_from_chain
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader
)

def fuse_meshes_per_scene_from_batched_objs(meshes: list[Meshes]) -> Meshes:
    fused_scenes = []
    batch_size = len(meshes[0].verts_list())
    for i in range(batch_size):
        meshes_per_scene = [mesh[i] for mesh in meshes] 

        fused_scene = join_meshes_as_scene(meshes_per_scene)

        fused_scenes.append(fused_scene)

    return join_meshes_as_batch(fused_scenes)

class BaseRoboRenderer:
    def __init__(self, xml_path, img_height=84, img_width=84, device="cuda"):
        self.device = device
        self.chain = build_chain_from_mjcf(xml_path)
        self.raw_meshes = build_meshes_from_chain(self.chain)
        self.img_height = img_height,
        self.img_width = img_width
        self.to(self.device)

    def to(self, device):
        self.chain.to(device=device)
        for body_name, meshes in self.raw_meshes.items():
            meshes_on_new_devices = []
            for mesh in meshes:
                meshes_on_new_devices.append(mesh.to(device))
            self.raw_meshes[body_name] = meshes_on_new_devices

    def build_scene_meshes(self, qpos):
        batch_size = qpos.shape[0]
        meshes = []
        body_transforms = self.chain.forward_kinematics(qpos)
        for body_name, transform in body_transforms.items():
            if body_name in self.raw_meshes:
                raw_meshes = self.raw_meshes[body_name]
                for raw_mesh in raw_meshes:
                    verts = raw_mesh.verts_packed().clone()
                    # transform the mesh to associated direction
                    verts = [v for v in transform.transform_points(verts)]
                    faces = [raw_mesh.faces_packed().clone() for _ in range(batch_size)]
                    textures = raw_mesh.textures.extend(batch_size)
                    meshes.append(Meshes(verts=verts, faces=faces, textures=textures))
        
        return fuse_meshes_per_scene_from_batched_objs(meshes)
    
    def get_renderer(self, qpos, camera_id=0):
        raise NotImplementedError

    def render(self, qpos, camera_id=0):
        renderer = self.get_renderer(qpos=qpos, camera_id=camera_id)
        meshes = self.build_scene_meshes(qpos)
        images = renderer(meshes)
        return images
