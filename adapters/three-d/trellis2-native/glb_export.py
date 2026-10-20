"""Keep trimesh's alignment bytes outside embedded WebP image buffer views."""

from __future__ import annotations

from pathlib import Path


def export_webp_glb(mesh, path: Path) -> None:
    lengths: dict[int, int] = {}

    def inspect_images(buffers, tree) -> None:
        blobs = list(buffers.values())
        for image in tree.get("images", []):
            if image.get("mimeType") != "image/webp":
                continue
            index = image["bufferView"]
            blob = blobs[index]
            if blob[:4] != b"RIFF" or blob[8:12] != b"WEBP":
                continue  # Validation still rejects malformed exporter output.
            length = int.from_bytes(blob[4:8], "little") + 8
            padding = len(blob) - length
            # trimesh 4.10.1 _buffer_append pads with spaces; _build_views
            # incorrectly includes those bytes in the image's byteLength.
            if 0 <= padding <= 3 and blob[length:] == b" " * padding:
                lengths[index] = length

    def trim_views(tree) -> None:
        for index, length in lengths.items():
            tree["bufferViews"][index]["byteLength"] = length

    mesh.export(
        path,
        extension_webp=True,
        buffer_postprocessor=inspect_images,
        tree_postprocessor=trim_views,
    )
