"""The creator's BPT steps (``main.py``) on the mounted checkpoint, for one mesh.

run.py starts this in its own process group with the creator's source on ``PYTHONPATH`` and a
working directory holding the ``miche`` link its point-cloud encoder reads its config from. It does
what ``main.py`` does for ``--input_type mesh``: normalise the mesh to the unit cube, sample 50,000
surface points with normals and keep 4,096 at random, let the 500M model generate the artist-style
vertex codes at the given temperature (top-k 50, top-p 0.95, as the creator hard-codes), decode them
and build the mesh. The result stays in the creator's normalised space; run.py restores the scale.
The creator's ``main.py`` parses its arguments at import, so its steps are repeated here.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

from model.data_utils import to_mesh
from model.model import MeshTransformer
from model.serializaiton import BPT_deserialize
from utils import apply_normalize, joint_filter, sample_pc

# The point cloud the model is conditioned on (config pc_num) and the creator's filter settings.
PC_NUM = 4096
FILTER_KWARGS = {"k": 50, "p": 0.95}


def build_model(config_path: Path, checkpoint: Path):
    with open(config_path, "r") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    model = MeshTransformer(
        dim=config["dim"],
        attn_depth=config["depth"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        mode=config["mode"],
        num_discrete_coors=2 ** int(config["quant_bit"]),
        block_size=config["block_size"],
        offset_size=config["offset_size"],
        conditioned_on_pc=config["conditioned_on_pc"],
        use_special_block=config["use_special_block"],
        encoder_name=config["encoder_name"],
        encoder_freeze=config["encoder_freeze"],
    )
    model.load(str(checkpoint))
    return model.eval().half().cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # The creator seeds nothing; the recipe's seed makes a run repeatable.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = build_model(args.config, args.checkpoint)
    mesh = apply_normalize(trimesh.load(str(args.mesh), force="mesh"))
    pc_normal = sample_pc(mesh, pc_num=PC_NUM, with_normal=True)
    with torch.no_grad():
        codes = model.generate(
            batch_size=1,
            temperature=args.temperature,
            pc=torch.from_numpy(pc_normal).unsqueeze(0).cuda().half(),
            filter_logits_fn=joint_filter,
            filter_kwargs=FILTER_KWARGS,
            return_codes=True,
        )
    code = codes[0]
    code = code[code != model.pad_id].cpu().numpy()
    # The creator swaps a failed decode for np.zeros(3, 3), which is itself a TypeError. A mesh that
    # cannot be decoded is no artifact, so the job fails with the decoder's own error.
    try:
        vertices = BPT_deserialize(
            code,
            block_size=model.block_size,
            offset_size=model.offset_size,
            use_special_block=model.use_special_block,
        )
    except Exception as error:
        raise SystemExit(f"BPT could not decode the generated codes: {error!r}") from error
    faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
    result = to_mesh(vertices, faces, transpose=False, post_process=True)
    np.savez(args.output, vertices=result.vertices, faces=result.faces)


if __name__ == "__main__":
    main()
