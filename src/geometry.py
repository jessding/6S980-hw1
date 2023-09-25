from jaxtyping import Float
from torch import Tensor, inverse, ones, squeeze, unsqueeze, zeros


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    one = ones((*points.shape[:-1], points.shape[-1] + 1))
    one[..., :-1] = points
    return one


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""
    zero = zeros((*points.shape[:-1], points.shape[-1] + 1))
    zero[..., :-1] = points
    return zero


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    return squeeze(transform @ unsqueeze(xyz, -1), axis=-1)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """
    # print((inverse(cam2world) @ unsqueeze(xyz, -1)).dtype)
    return squeeze(inverse(cam2world).double() @ unsqueeze(xyz, -1).double(), axis=-1)
    # return einsum(inverse(cam2world), xyz, "... i j, ... j -> ... i")


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """
    return squeeze(cam2world @ unsqueeze(xyz, -1), axis=-1)
    # return einsum(cam2world, xyz, "... i j, ... j -> ... i")


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    xyz = xyz[..., :-1]
    uvw = squeeze(intrinsics @ unsqueeze(xyz, -1), axis=-1)
    return uvw[..., :-1] / unsqueeze(uvw[..., -1], axis=-1)
