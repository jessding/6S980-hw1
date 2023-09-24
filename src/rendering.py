from jaxtyping import Float, install_import_hook
from torch import Tensor, ones, squeeze, unsqueeze

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.geometry import (
        homogenize_points,
        homogenize_vectors,
        project,
        transform_cam2world,
        transform_rigid,
        transform_world2cam,
    )


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    batch = extrinsics.shape[0]
    vertices = unsqueeze(vertices, 0).expand(
        batch, -1, -1
    )  # 1, 34834, 3 -> batch=16, 34834, 3
    vertices_num = vertices.shape[1]
    extrinsics = unsqueeze(extrinsics, 1).expand(
        -1, vertices_num, -1, -1
    )  # torch.Size([16, 34834, 4, 4])
    intrinsics = unsqueeze(intrinsics, 1).expand(
        -1, vertices_num, -1, -1
    )  # torch.Size([16, 34834, 3, 3])
    canvas = ones(batch, *resolution)  # torch.Size([16, 256, 256])
    homoVerts = homogenize_points(vertices)  # torch.Size([16, 34834, 4])
    cam = transform_world2cam(homoVerts, extrinsics)
    pxs = project(cam, intrinsics)
    px_row = (pxs[..., 0] * resolution[0]).int()
    px_col = (pxs[..., 1] * resolution[1]).int()
    for b in range(batch):
        canvas[b, px_row[b], px_col[b]] = 0

    return canvas.transpose(-1, -2)
