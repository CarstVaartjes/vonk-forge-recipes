"""Validate the actual pinned ComfyUI consumer without loading model weights."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile

sys.argv += ["--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import nodes
import execution
import folder_paths

async def main():
    await nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)
    document = json.loads(Path('/opt/vonk/source/workflows/fasth3.json').read_text())
    with tempfile.TemporaryDirectory() as directory:
        for model in document['models']:
            path = Path(directory) / model['category'] / model['filename']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            folder_paths.add_model_folder_path(model['category'], str(path.parent))
        prompt = document['prompt']
        prompt['6']['inputs']['prompt'] = 'A sunset over the sea with soft ocean sounds.'
        prompt['8']['inputs']['noise_seed'] = 0
        result = await execution.validate_prompt('vonk-h3-build-smoke', prompt, None)
        assert result[0], result
        assert '16' in result[2], result
    print('Pinned TensorFold H3 workflow accepted by ComfyUI')

asyncio.run(main())
