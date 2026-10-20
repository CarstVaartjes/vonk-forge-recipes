# MiniCPM single-GPU vision load repair

Checked 2026-10-20 against the pinned sources and current upstream heads.

The 2026-10-20 21:55Z sweep evidence reported ExLlamaV3 `1.6.1.post1`,
`Loading vision modules 0/38`, and a `conv2d` exception on input `[1, 2048]`.
This is a deterministic loader integration error, not evidence of an invalid
EXL3 quant or insufficient memory.

- [The quant card](https://huggingface.co/vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw/blob/855cb7601710064aeb7452b6b7e45e60a40be76b/README.md)
  describes 8-bit vision weights and native image smoke results, while warning
  that compatibility depends on runtime architecture and multimodal support.
  These are the publisher's results, not a Vonk hardware test.
- [Pinned TabbyAPI `model.py:392-394`](https://github.com/vcruz305/tabbyAPI/blob/880e6b8bbdeda83090eb6cb5eabdbf8623fc4426/backends/exllamav3/model.py#L392-L394)
  logs that GPU splitting is disabled on one GPU. Nevertheless,
  [lines 1018-1022](https://github.com/vcruz305/tabbyAPI/blob/880e6b8bbdeda83090eb6cb5eabdbf8623fc4426/backends/exllamav3/model.py#L1018-L1022)
  pass reserve/split parameters without `device` to the vision loader.
- [Pinned ExLlamaV3 `model_ls.py:64-69`](https://github.com/vcruz305/exllamav3/blob/475461e9a1a288e7c10eb472056239d3fb78b625/exllamav3/model/model_ls.py#L64-L69)
  defaults to a `(1, chunk_size)` integer dummy state. MiniCPM's
  [vision class](https://github.com/vcruz305/exllamav3/blob/475461e9a1a288e7c10eb472056239d3fb78b625/exllamav3/architecture/minicpmv4_7_vision.py#L20-L26)
  does not override it or disable measuring forwards. Its first module is a 2D
  convolution. The autosplit measuring forward at `model_ls.py:212` therefore
  feeds text-shaped data to `modules/conv.py:90`.
- The vision model's real `forward` orchestrates image reconstruction, positions,
  window merging and final merging. Merely changing the dummy tensor shape
  would not turn its module list into that execution graph.
- [ExLlamaV3 `model.py:592-600`](https://github.com/vcruz305/exllamav3/blob/475461e9a1a288e7c10eb472056239d3fb78b625/exllamav3/model/model.py#L592-L600)
  supports `device=torch.device("cuda:0")`: `_load_single` loads module weights
  without measuring forwards. Its API forbids simultaneous reserve/split inputs.

No config switch exposed by this pinned TabbyAPI selects that vision load path:
`gpu_split` is ignored with one GPU, and `gpu_split_auto: false` still calls
`load_gen` without `device`. Disabling vision would hide the broken capability.

The ExLlamaV3 fork head `d968d96dbe56072317329d573bd06e1403eb949b` retains
the same MiniCPM vision class and loader. TabbyAPI's fork head is still the
pinned `880e6b8bbdeda83090eb6cb5eabdbf8623fc4426`. Stock ExLlamaV3 head
`a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a` has no MiniCPM architecture
registration; replacing the specialized fork with the generic version is not
a fix. Stock TabbyAPI head `a270c555db91a5178d72f7dee7bdd34951ac1dad`
also loads vision without an explicit device.

Keep the existing model, engine, server and dependency pins. Apply
`patch-tabby-vision.py` while building the image: on one GPU, load vision onto
`cuda:0` with neither reserve nor split. Text and draft loading retain their
existing budgeting, and the multi-GPU branch retains its prior behavior.
The patch refuses an unexpected source block so future upstream changes must
be reviewed. The source archive remains unchanged and retains its upstream hash.

`tests/test_minicpm_vision_load.py` executes the shipped TabbyAPI load method
and ExLlamaV3 load dispatcher, with CUDA/weight I/O doubled. It injects the
reported failure into the autosplit branch: the original code fails and the
patched code uses the single-device path and continues through draft/text load.
This verifies loader behavior without downloading weights. Image inference,
container build and physical Spark acceptance have not been rerun; this task
does not access production or the running sweep.
