import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)

import mujoco
import torch
from kinematics.chain import Chain
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import trimesh

# Types we can convert to meshes
GEOM_TYPE_MAP = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
}

def build_pytorch3d_mesh(geom):        
    geom_type = GEOM_TYPE_MAP.get(geom.type_id, None)
    size = geom.size
    rgb = torch.tensor(geom.rgba[:3], dtype=torch.float32)
    if geom_type == "box":
        half_extents = size[:3]
        trmesh = trimesh.creation.box(extents=2 * half_extents)
    elif geom_type == "capsule":
        radius, half_height = size[0], size[1]
        trmesh = trimesh.creation.capsule(radius=radius, height=2 * half_height, count=[16, 8])
    elif geom_type == "sphere":
        trmesh = trimesh.creation.icosphere(radius=size[0])
    elif geom_type == "cylinder":
        radius, half_height = size[0], size[1]
        trmesh = trimesh.creation.cylinder(radius=radius, height=2 * half_height)
    else:
        raise ValueError("Unsupported geom!")
    
    verts = torch.tensor(trmesh.vertices, dtype=torch.float32)
    verts = geom.offset.transform_points(verts)[0]
    faces = torch.tensor(trmesh.faces, dtype=torch.int64)
    verts_features = rgb.unsqueeze(0).repeat(verts.shape[0], 1) 
    verts_features = verts_features.unsqueeze(0)

    return Meshes(verts=[verts], 
                faces=[faces], 
                textures=TexturesVertex(verts_features=verts_features))

def build_meshes_from_chain(chain: Chain):
    frames_name = chain.get_frame_names()
    body_to_mesh = {}
    for frame_name in frames_name:
        geoms = []
        frame = chain.find_frame(frame_name)
        for geom in frame.geoms:
            geoms.append(build_pytorch3d_mesh(geom))
        body_to_mesh[frame_name] = geoms
    
    return body_to_mesh    