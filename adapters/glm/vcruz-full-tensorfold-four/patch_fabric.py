"""Keep NCCL selectors intact while b12x receives the selected plain device."""
from pathlib import Path
import sys


def patch(root):
    path = Path(root) / "src/tensorfold/families/glm_moe_dsa/cuda/roce.py"
    source = path.read_text()
    before = 'hca = os.environ.get("NCCL_IB_HCA")'
    after = 'hca = os.environ["B12X_ROCE_HCA"]  # Vonk: retain exact NCCL selectors'
    if source.count(before) != 1:
        raise ValueError("pinned TensorFold fabric binding site changed")
    path.write_text(source.replace(before, after))


if __name__ == "__main__":
    patch(sys.argv[1])
