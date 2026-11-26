"""Build-time import smoke: no GPU, no weights.

Run with the creator's source trees on PYTHONPATH and the adapter's modules in the working
directory. A missing wheel, shared library or native build fails the image build here, not the
hardware start (MeshRender hides a failed mesh_inpaint_processor import until first use).
"""

import torchvision_fix

# basicsr imports the removed torchvision.transforms.functional_tensor; the fix comes first.
torchvision_fix.apply_fix()

import custom_rasterizer  # noqa: E402, F401
import custom_rasterizer_kernel  # noqa: E402, F401
import DifferentiableRenderer.mesh_inpaint_processor  # noqa: E402, F401
import open3d  # noqa: E402, F401
import pygltflib  # noqa: E402, F401
import pymeshlab  # noqa: E402, F401
import run  # noqa: E402, F401
import trimesh  # noqa: E402, F401
import worker  # noqa: E402, F401
import xatlas  # noqa: E402, F401
from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402, F401
from realesrgan import RealESRGANer  # noqa: E402, F401
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest  # noqa: E402, F401
