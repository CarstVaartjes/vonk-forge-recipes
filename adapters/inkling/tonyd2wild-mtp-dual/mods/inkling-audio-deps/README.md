# Inkling audio dependencies

The base Spark image does not include vLLM's optional audio packages. Image
input works without them, but WAV input fails during API preprocessing with
`ImportError: Please install vllm[audio] for audio support`.

This mod installs pinned ARM64 wheels for PyAV, SciPy, SoundFile, and SoXR. It
does not reinstall vLLM or Torch. Validate with `scripts/test-multimodal.py`
using a mono 16 kHz WAV file.
