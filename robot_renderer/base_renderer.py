import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
from kinematics.load_kinematic_trees_from_mjcf import build_chain_from_mjcf
from kinematics.load_meshes_from_kinematic_trees import build_meshes_from_chain
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import torch

def fuse_meshes(meshes_list: list[Meshes]) -> Meshes:
    """
    Fusing M batched Meshes (N each)
    into N Meshes, each with M parts fused.
    Args:
        meshes_list: list of M Meshes with batch size N
    Returns:
        Meshes with batch size N, each containing M sub-meshes fused
    """
    batch_size = len(meshes_list[0])
    num_objs = len(meshes_list)

    # Step 1: Collect all verts/faces/textures in M x N order
    verts_all = []
    faces_all = []
    textures_all = []

    for mesh_idx, mesh in enumerate(meshes_list):
        verts_list = mesh.verts_list()  # list of N tensors
        faces_list = mesh.faces_list()
        tex_list = mesh.textures.verts_features_list()

        for i in range(batch_size):
            verts_all.append(verts_list[i])
            faces_all.append(faces_list[i])
            textures_all.append(tex_list[i])

    # Step 2: Build big tensors and track offsets
    scene_verts = []
    scene_faces = []
    scene_textures = []

    for i in range(batch_size):
        v_offset = 0
        verts_scene = []
        faces_scene = []
        textures_scene = []

        for j in range(num_objs):
            idx = i + j * batch_size
            verts = verts_all[idx]
            faces = faces_all[idx] + v_offset
            tex = textures_all[idx]

            verts_scene.append(verts)
            faces_scene.append(faces)
            textures_scene.append(tex)

            v_offset += verts.shape[0]

        scene_verts.append(torch.cat(verts_scene, dim=0))
        scene_faces.append(torch.cat(faces_scene, dim=0))
        scene_textures.append(torch.cat(textures_scene, dim=0))

    textures = TexturesVertex(verts_features=scene_textures)
    return Meshes(verts=scene_verts, faces=scene_faces, textures=textures)

class BaseRoboRenderer:
    def __init__(self, xml_path, img_height=84, img_width=84, device="cuda"):
        self.device = device
        self.chain = build_chain_from_mjcf(xml_path)
        self.raw_meshes = build_meshes_from_chain(self.chain)
        self.build_background()
        self.img_height = img_height,
        self.img_width = img_width
        self.to(self.device)

    def build_background(self):
        pass

    def to(self, device):
        self.chain.to(device=device)
        for body_name, meshes in self.raw_meshes.items():
            meshes_on_new_devices = []
            for mesh in meshes:
                meshes_on_new_devices.append(mesh.to(device))
            self.raw_meshes[body_name] = meshes_on_new_devices

    def build_scene_meshes(self, transforms):
        batch_size = len(next(iter(transforms.values())))
        meshes = []
        if "background" in self.raw_meshes:
            background_meshes = self.raw_meshes["background"]
            for background_mesh in background_meshes:
                meshes.append(background_mesh.extend(batch_size))
        for body_name, transform in transforms.items():
            if body_name in self.raw_meshes:
                raw_meshes = self.raw_meshes[body_name]
                for raw_mesh in raw_meshes:
                    verts = raw_mesh.verts_packed().clone()
                    # transform the mesh to associated direction
                    verts = [v for v in transform.transform_points(verts)]
                    faces = [raw_mesh.faces_packed().clone() for _ in range(batch_size)]
                    textures = raw_mesh.textures.extend(batch_size)
                    meshes.append(Meshes(verts=verts, faces=faces, textures=textures))
        meshes = fuse_meshes(meshes)
        return meshes
    
    def get_renderer(self, qpos, camera_id=0):
        raise NotImplementedError
    
    def get_transform(self, qpos):
        raise NotImplementedError

    def render(self, qpos, camera_id=0):
        qpos = torch.atleast_2d(qpos)
        renderer = self.get_renderer(qpos=qpos, camera_id=camera_id)
        transforms = self.get_transform(qpos)
        meshes = self.build_scene_meshes(transforms)
        images = renderer(meshes)
        return torch.clamp(images, 0.0, 1.0)
