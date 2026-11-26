"""Write the paint step's OBJ and texture maps as one PBR GLB.

The creator's own GLB writers cannot run here: ``DifferentiableRenderer.mesh_utils.convert_obj_to_glb``
needs bpy (no aarch64 build), and ``convert_utils.create_glb_with_pbr_materials`` embeds the images as
``data:`` URIs holding JPEG bytes under a PNG media type, which the GLB validation refuses. This module
keeps the creator's material mapping (``convert_utils.combine_metallic_roughness``: R = 255,
G = roughness, B = metallic) and lets trimesh embed the images in the binary chunk. Both textures are
written as PNG, so nothing is compressed a second time.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


def metallic_roughness_image(
    metallic: Image.Image, roughness: Image.Image
) -> Image.Image:
    """The glTF metallic-roughness image: R = 255, G = roughness, B = metallic."""
    metallic = metallic.convert("L")
    roughness = roughness.convert("L")
    if roughness.size != metallic.size:
        roughness = roughness.resize(metallic.size)
    combined = np.zeros((metallic.height, metallic.width, 3), dtype=np.uint8)
    combined[:, :, 0] = 255
    combined[:, :, 1] = np.asarray(roughness)
    combined[:, :, 2] = np.asarray(metallic)
    return Image.fromarray(combined)


# A PNG ends with its IEND chunk (length 0, type, CRC); trimesh follows it with 0..3 alignment spaces.
_PNG_END = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def exclude_image_alignment(glb: bytes) -> bytes:
    """The GLB with each PNG view ending at its IEND chunk instead of at trimesh's alignment spaces.

    trimesh counts the 0..3 spaces that align the next view as part of the image, so a validator that
    checks the PNG end finds the file longer than the image. Only that alignment is excluded: the BIN
    chunk and every offset stay as they are.
    """
    json_length = struct.unpack_from("<I", glb, 12)[0]
    document = json.loads(glb[20 : 20 + json_length])
    binary_chunk = glb[20 + json_length :]
    binary = binary_chunk[8:]
    for image in document.get("images", []):
        view = document["bufferViews"][image["bufferView"]]
        start, length = view.get("byteOffset", 0), view["byteLength"]
        payload = binary[start : start + length]
        for padding in range(1, 4):
            if payload.endswith(_PNG_END + b" " * padding):
                view["byteLength"] = length - padding
                break
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    return (
        struct.pack("<4sII", b"glTF", 2, 20 + len(encoded) + len(binary_chunk))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + binary_chunk
    )


def write_pbr_glb(
    obj_path: Path,
    albedo_path: Path,
    metallic_path: Path,
    roughness_path: Path,
    output_path: Path,
) -> None:
    mesh = trimesh.load(obj_path, process=False, force="mesh")
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise ValueError("the painted mesh has no per-vertex UV coordinates")
    with Image.open(albedo_path) as albedo, Image.open(metallic_path) as metallic:
        with Image.open(roughness_path) as roughness:
            material = PBRMaterial(
                name="PBR_Material",
                baseColorTexture=albedo.convert("RGB"),
                metallicRoughnessTexture=metallic_roughness_image(metallic, roughness),
                metallicFactor=1.0,
                roughnessFactor=1.0,
            )
    mesh.visual = TextureVisuals(uv=uv, material=material)
    output_path.write_bytes(
        exclude_image_alignment(mesh.export(file_type="glb"))
    )
