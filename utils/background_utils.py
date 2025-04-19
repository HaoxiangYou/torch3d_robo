import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

def create_floor(x_dim=10, y_dim=10, center=[0.0, 0.0, 0.0], tile_size=1.0, color1=[.2, .3, .4], color2=[.7, .8, .9], device='cpu'):
    verts_list = []
    faces_list = []
    color_list = []

    vertex_count = 0
    cx_offset, cy_offset, cz_offset = center

    for i in range(x_dim):
        for j in range(y_dim):
            # Alternate colors in a checkerboard pattern
            color = color1 if (i + j) % 2 == 0 else color2

            # Compute tile center with grid center offset
            cx = (i - x_dim / 2 + 0.5) * tile_size + cx_offset
            cy = (j - y_dim / 2 + 0.5) * tile_size + cy_offset
            cz = cz_offset  # Grid lies in XY plane at Z=cz_offset

            # Define tile vertices
            s = tile_size / 2
            verts = torch.tensor([
                [cx - s, cy - s, cz],
                [cx + s, cy - s, cz],
                [cx + s, cy + s, cz],
                [cx - s, cy + s, cz]
            ], dtype=torch.float32, device=device)

            faces = torch.tensor([
                [2, 1, 0],
                [3, 2, 0]
            ], dtype=torch.int64, device=device) + vertex_count

            colors = torch.tensor([color] * 4, dtype=torch.float32, device=device)

            verts_list.append(verts)
            faces_list.append(faces)
            color_list.append(colors)

            vertex_count += 4

    # Combine everything
    all_verts = torch.cat(verts_list, dim=0)
    all_faces = torch.cat(faces_list, dim=0)
    all_colors = torch.cat(color_list, dim=0).unsqueeze(0)

    textures = TexturesVertex(verts_features=all_colors)
    return Meshes(verts=[all_verts], faces=[all_faces], textures=textures)

def create_sky_plane(center=[0, 0, 0], x_dim=2.0, z_dim=2.0, 
                     color_top=[0.4, 0.6, 0.8], color_bottom=[0.0, 0.0, 0.0], 
                     star_density=1, star_size=0.01, device='cpu'):
    cx, cy, cz = center
    hx, hz = x_dim / 2, z_dim / 2

    # Sky plane vertices (in XZ plane at Y=cy)
    verts = torch.tensor([
        [cx - hx, cy, cz - hz],  # bottom-left
        [cx + hx, cy, cz - hz],  # bottom-right
        [cx + hx, cy, cz + hz],  # top-right
        [cx - hx, cy, cz + hz],  # top-left
    ], dtype=torch.float32, device=device)

    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int64, device=device)

    # Color gradient along Z axis
    base_colors = []
    for vert in verts:
        t = (vert[2] - (cz - hz)) / (2 * hz)
        color = [(1 - t) * color_bottom[k] + t * color_top[k] for k in range(3)]
        base_colors.append(color)
    base_colors = torch.tensor(base_colors, dtype=torch.float32, device=device)

    all_verts = [verts]
    all_faces = [faces]
    all_colors = [base_colors]

    vertex_offset = 4

    # Compute number of stars from density
    plane_area = x_dim * z_dim
    num_stars = int(star_density * plane_area)

    # Generate star centers
    star_x = np.random.uniform(cx - hx, cx + hx, size=(num_stars,))
    star_z = np.random.uniform(cz - hz, cz + hz, size=(num_stars,))
    s = star_size / 2

    # Create quads for all stars at once (each star has 4 vertices)
    # Order: bottom-left, bottom-right, top-right, top-left
    base_offsets = torch.tensor([
        [-s, 0, -s],
        [ s, 0, -s],
        [ s, 0,  s],
        [-s, 0,  s]
    ], dtype=torch.float32, device=device)  # shape: (4, 3)

    # Expand star centers
    centers = torch.tensor(np.stack([star_x, np.full_like(star_x, cy), star_z], axis=1),
                        dtype=torch.float32, device=device)  # (N, 3)
    star_verts = centers[:, None, :] + base_offsets[None, :, :]  # (N, 4, 3)
    star_verts = star_verts.reshape(-1, 3)  # (N*4, 3)

    # Faces per star (relative indices)
    face_template = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int64, device=device)  # (2, 3)

    # Offset face indices for each star
    offsets = (torch.arange(num_stars, device=device) * 4).unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
    star_faces = face_template.unsqueeze(0) + offsets  # (N, 2, 3)
    star_faces = star_faces.reshape(-1, 3)

    # Star colors
    star_colors = torch.ones((num_stars * 4, 3), dtype=torch.float32, device=device)

    # Append to base geometry
    all_verts.append(star_verts)
    all_faces.append(star_faces + vertex_offset)
    all_colors.append(star_colors)

    vertex_offset += num_stars * 4

    all_verts = torch.cat(all_verts, dim=0)
    all_faces = torch.cat(all_faces, dim=0)
    all_colors = torch.cat(all_colors, dim=0).unsqueeze(0)

    textures = TexturesVertex(verts_features=all_colors)
    return Meshes(verts=[all_verts], faces=[all_faces], textures=textures)
