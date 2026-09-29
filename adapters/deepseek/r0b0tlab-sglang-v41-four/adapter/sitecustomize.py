"""Enable pinned serving hooks explicitly; python -S health/sandbox bypasses them."""
import os
import sys

try:
    adaptive = os.environ.get('DSV41_ADAPTIVE_CHUNK','0')
    engram = os.environ.get('DSV41_ENGRAM_FILE_STORE','0')
    kslice = os.environ.get('DSV41_TORCH_INDEXER_KSLICE','0')
    if adaptive not in ('0','1') or engram not in ('0','1') or kslice not in ('0','1'):
        raise ValueError('DSV41 hook switches must be 0 or 1')
    if adaptive == '1':
        from sglang_patch.prefill_chunk_sizer import install as install_chunk
        install_chunk()
    elif kslice == '1':
        from sglang_patch.torch_indexer_budget import install as install_kslice
        install_kslice()
    if engram == '1':
        model_path=os.environ.get('DSV41_MODEL_PATH','')
        if not model_path:
            raise RuntimeError('DSV41_ENGRAM_FILE_STORE requires DSV41_MODEL_PATH')
        from sglang_patch.engram_file_store import install as install_engram
        install_engram(model_path)
except Exception as error:
    # Python normally reports sitecustomize exceptions then keeps running;
    # that would silently serve without the memory/correctness hooks.
    sys.stderr.write('DSV41_ADAPTER_REFUSED: '+repr(error)+'\n')
    sys.stderr.flush()
    os._exit(78)
