"""Build-time import smoke: no GPU, no weights.

Run with the creator's source tree on PYTHONPATH and the adapter's modules in the working directory.
A missing wheel or shared library fails the image build here, not the hardware start.
"""

import av  # noqa: F401
import open_clip  # noqa: F401
import run  # noqa: F401
import worker  # noqa: F401
from mmaudio.eval_utils import all_model_cfg, generate, load_video  # noqa: F401
from mmaudio.model.flow_matching import FlowMatching  # noqa: F401
from mmaudio.model.networks import get_my_mmaudio  # noqa: F401
from mmaudio.model.utils.features_utils import FeaturesUtils  # noqa: F401
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest  # noqa: F401
