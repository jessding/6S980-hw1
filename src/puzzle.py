import json
from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float, install_import_hook
from numpy import asarray
from PIL import Image
from torch import Tensor, float64, inverse, stack, tensor

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.provided_code import get_bunny

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.rendering import render_point_cloud


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""
    returned = {}
    with open(path / "metadata.json") as metadata:
        data = json.load(metadata)
        returned["extrinsics"] = tensor(asarray(data["extrinsics"]))
        returned["intrinsics"] = tensor(asarray(data["intrinsics"]))
    imgs = []
    for i in range(32):
        i = str(i).zfill(2)
        img_path = path / "images" / f"{i}.png"
        img = tensor(asarray(Image.open(img_path)))
        imgs += [img]
    returned["images"] = stack(imgs)
    return returned


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """
    mat = [
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]
    mat = tensor(mat).to(dtype=float64)
    extrinsics = inverse(dataset["extrinsics"])
    intrinsics = dataset["intrinsics"]
    rot = extrinsics[..., :3, :3]
    rot = rot @ mat
    extrinsics[..., :3, :3] = rot
    vertices, _ = get_bunny()
    vertices = vertices.to(dtype=float64)
    canvas = render_point_cloud(vertices, extrinsics, intrinsics)
    dataset["images"] = canvas
    dataset["extrinsics"] = extrinsics

    return dataset


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    return "w2c"


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    return "-y"


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    return "z"


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    return "-x"


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    s = """I bashed to find what the exact rotation matrix was. There's two possible 
    options for the sign (+/-) of each of the three 1s in the identity matrix, and 3!=6 
    options for the permutation of the rows from the identity rotation matrix, and 2 
    options for whether I was already in c2w or w2c format, for a total of 
    2^3 * 6 * 2 = 96 total possibilities. I first started bashing on permutation and 
    w2c/c2w until I found a combination that produced valid outputs and didn't error, 
    then I visually inspected the output and compared them with my puzzle dataset images
    to figure out what the sign of each row was."""

    return s
