"""Offline SkinTokens/TokenRig adapter with a pure glTF rig exporter."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from glb_validation import normalize_glb_json_padding, validate_mesh_glb
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    MAT4,
    SCALAR,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    VEC3,
    VEC4,
    Accessor,
    Attributes,
    Buffer,
    BufferView,
    Mesh,
    Node,
    Primitive,
    Scene,
    Skin,
)
from pygltflib import (
    Asset as GltfAsset,
)

INPUTS = Path("/inputs")
CHECKPOINT = Path("/models/target/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt")


def one_input_mesh() -> Path:
    candidates = sorted(
        path
        for path in INPUTS.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".glb"
    )
    if len(candidates) != 1:
        raise SystemExit("SkinTokens requires exactly one GLB input")
    candidate = candidates[0]
    try:
        validate_mesh_glb(candidate, profile="geometry")
    except ValueError as exc:
        raise SystemExit(f"SkinTokens input is not a valid GLB mesh: {exc}") from exc
    return candidate


def append_blob(blob: bytearray, value: bytes) -> tuple[int, int]:
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(value)
    return offset, len(value)


def export_rigged_glb(asset: object, output: Path) -> None:
    vertices = np.asarray(asset.vertices, dtype=np.float32)
    faces = np.asarray(asset.faces, dtype=np.uint32).reshape(-1)
    parents = np.asarray(asset.parents, dtype=np.int32)
    matrices = np.asarray(asset.matrix_local, dtype=np.float32)
    skin = np.asarray(asset.skin, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise SystemExit("TokenRig returned invalid vertices")
    if skin.shape != (len(vertices), len(parents)) or matrices.shape != (len(parents), 4, 4):
        raise SystemExit("TokenRig returned an invalid skeleton or skin")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces.reshape(-1, 3), process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    top = np.argpartition(skin, -min(4, skin.shape[1]), axis=1)[:, -4:]
    weights = np.take_along_axis(skin, top, axis=1)
    order = np.argsort(weights, axis=1)[:, ::-1]
    joints = np.take_along_axis(top, order, axis=1).astype(np.uint16)
    weights = np.take_along_axis(weights, order, axis=1)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    weights = weights.astype(np.float32)
    inverse_bind = np.linalg.inv(matrices).transpose(0, 2, 1).astype(np.float32)

    blob = bytearray()
    views: list[BufferView] = []
    accessors: list[Accessor] = []

    def accessor(array: np.ndarray, component: int, kind: str, target: int | None = None) -> int:
        contiguous = np.ascontiguousarray(array)
        offset, length = append_blob(blob, contiguous.tobytes())
        view = BufferView(buffer=0, byteOffset=offset, byteLength=length, target=target)
        views.append(view)
        count = len(contiguous)
        item = Accessor(bufferView=len(views) - 1, byteOffset=0, componentType=component, count=count, type=kind)
        if kind == VEC3:
            item.min = contiguous.min(axis=0).astype(float).tolist()
            item.max = contiguous.max(axis=0).astype(float).tolist()
        accessors.append(item)
        return len(accessors) - 1

    position_accessor = accessor(vertices, FLOAT, VEC3, ARRAY_BUFFER)
    normal_accessor = accessor(normals, FLOAT, VEC3, ARRAY_BUFFER)
    joint_accessor = accessor(joints, UNSIGNED_SHORT, VEC4, ARRAY_BUFFER)
    weight_accessor = accessor(weights, FLOAT, VEC4, ARRAY_BUFFER)
    index_accessor = accessor(faces, UNSIGNED_INT, SCALAR, ELEMENT_ARRAY_BUFFER)
    bind_accessor = accessor(inverse_bind, FLOAT, MAT4)

    joint_names = asset.joint_names or [f"joint_{index}" for index in range(len(parents))]
    nodes = [Node(name="Rigged mesh", mesh=0, skin=0)]
    for index, parent in enumerate(parents):
        local = matrices[index] if parent < 0 else np.linalg.inv(matrices[parent]) @ matrices[index]
        children = (np.flatnonzero(parents == index) + 1).astype(int).tolist()
        nodes.append(
            Node(
                name=str(joint_names[index]),
                matrix=local.T.reshape(-1).astype(float).tolist(),
                children=children or None,
            )
        )
    roots = (np.flatnonzero(parents < 0) + 1).astype(int).tolist()
    primitive = Primitive(
        attributes=Attributes(
            POSITION=position_accessor,
            NORMAL=normal_accessor,
            JOINTS_0=joint_accessor,
            WEIGHTS_0=weight_accessor,
        ),
        indices=index_accessor,
    )
    gltf = GLTF2(
        asset=GltfAsset(version="2.0", generator="Vonk Forge SkinTokens adapter"),
        scenes=[Scene(nodes=[0, *roots])],
        scene=0,
        nodes=nodes,
        meshes=[Mesh(primitives=[primitive])],
        skins=[Skin(inverseBindMatrices=bind_accessor, joints=list(range(1, len(parents) + 1)), skeleton=roots[0] if roots else None)],
        buffers=[Buffer(byteLength=len(blob))],
        bufferViews=views,
        accessors=accessors,
    )
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != "/opt/vonk/source/run.py" or args.output_mime != "model/gltf-binary":
        raise SystemExit("unexpected signed adapter contract")
    torch.manual_seed(args.seed)
    loaded = trimesh.load(one_input_mesh(), force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise SystemExit("input GLB does not contain a mesh")

    from src.data.dataset import DatasetConfig, RigDatasetModule
    from src.data.transform import Transform
    from src.model.tokenrig import TokenRigResult
    from src.server.spec import get_model
    from src.tokenizer.parse import get_tokenizer

    with tempfile.TemporaryDirectory(prefix="vonk-skintokens-") as temporary:
        npz = Path(temporary) / "input.npz"
        np.savez(npz, vertices=np.asarray(loaded.vertices), faces=np.asarray(loaded.faces))
        model = get_model(str(CHECKPOINT))
        assert model.tokenizer_config is not None
        tokenizer = get_tokenizer(**model.tokenizer_config)
        transform = Transform.parse(**model.transform_config["predict_transform"])
        configuration = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            datapath={"data_name": None, "loader": "npz", "filepaths": {"articulation": [str(npz)]}},
        ).split_by_cls()
        module = RigDatasetModule(
            predict_dataset_config=configuration,
            predict_transform=transform,
            tokenizer=tokenizer,
            process_fn=model._process_fn,
        )
        batch = next(iter(module.predict_dataloader()["articulation"]))
        batch = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        batch.pop("skeleton_tokens", None)
        batch.pop("skeleton_mask", None)
        batch["generate_kwargs"] = {
            "max_length": 2048,
            "top_k": 5,
            "top_p": 0.95,
            "temperature": 1.0,
            "repetition_penalty": 2.0,
            "num_return_sequences": 1,
            "num_beams": 10,
            "do_sample": True,
        }
        results: list[TokenRigResult] = model.predict_step(batch, skeleton_tokens=None, make_asset=True)["results"]
        result = results[0].asset
        if result is None:
            raise SystemExit("TokenRig did not return a rigged asset")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output_dir / ".output.tmp.glb"
        export_rigged_glb(result, temporary_output)
        normalize_glb_json_padding(temporary_output)
        try:
            validate_mesh_glb(temporary_output, profile="skinned")
        except ValueError as exc:
            temporary_output.unlink(missing_ok=True)
            raise SystemExit(f"SkinTokens produced an invalid GLB artifact: {exc}") from exc
        os.replace(temporary_output, args.output_dir / "output.glb")


if __name__ == "__main__":
    main()
