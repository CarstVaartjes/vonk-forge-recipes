"""Image-size bounds for image input (MM=1, SGLang's --enable-multimodal), as an import-time overlay.

    INKLING_MM_MAX_IMAGE_TOKENS=16384     most vision tokens the images of ONE request may have together (one token per
                                          40 x 40 pixel cell, each image at its native size); unset / 0: off
    INKLING_MM_INFLIGHT_TOKENS=16384      most vision tokens of all requests in flight together (from preprocessing to
                                          the end of the request); unset / 0: off. At least the per-request bound

Inkling2's image processor has no size or count limit: each image becomes ceil(height / 40) * ceil(width / 40) tokens
at its native resolution, preprocessing keeps every image of a request in memory at once (19.2 KB of BF16 patches
per token, plus a float32 copy of the image being encoded), and the vision tower encodes every image a prefill chunk
touches in one call. The KV pool and the prefill readers are bounded and charged to the pool plan; the vision side
is bounded here. Before any pixel is decoded, the request's images are sized from their headers (PIL images by their
size); a request above the bound fails with that message and the server keeps running. Images within the bound are
processed exactly as stock (no resize). 16,384 tokens hold two 12 MP phone photos (4032 x 3024: 7,676 tokens each)
or one 5120 x 5120 image.

A request's preprocessed patches stay in host memory until it finishes, so the per-request bound alone does not bound
a server with many image requests in flight (8 requests at 16,320 image tokens each next to thinking c24 and a cold
240K needle took MemAvailable below 3 GiB at MEM_FRACTION 0.86). The in-flight bound admits image requests in the
tokenizer before their images are decoded: a request whose image tokens do not fit next to those in flight waits
(nothing is decoded while it waits) until earlier image requests finish; a lone request is always admitted. Text-only
requests never wait.

Covers Inkling2ImageProcessor.preprocess (the API's image path and the encoder-side path) and the Inkling2
multimodal processor's request entry; precomputed embeddings from a separate encoder server (--language-only with
--encoder-urls) pass through neither and are not bounded.
"""

import asyncio
import threading
import weakref

import functools
import io
import os
import sys

ENV = "INKLING_MM_MAX_IMAGE_TOKENS"
ENV_INFLIGHT = "INKLING_MM_INFLIGHT_TOKENS"
MODULE = "sglang.srt.multimodal.inkling2.image_processing"
PROCESSOR = "sglang.srt.multimodal.inkling2.processor"
TOKENIZER = "sglang.srt.managers.tokenizer_manager"
CELL = 40  # pixels per vision token side (Inkling2ImageProcessor.count_tokens_for_image)
LIMIT = 1 << 20
STATE = {"installed": False, "cap": 0, "inflight": 0}


def cap_from_env(env=None, name=ENV, limit=LIMIT):
    env = os.environ if env is None else env
    raw = env.get(name, "").strip()
    if not raw.isdigit() or int(raw) > limit:
        if raw == "":
            return 0
        raise RuntimeError(f"inkling_mm: {name}={raw!r} must be a whole number of tokens in [0, {limit}] (0 = off)")
    return int(raw)


class Budget:
    """Vision tokens of the requests in flight. acquire() waits (asyncio) until the request fits next to those in
    flight, or nothing is in flight; release() is idempotent per request id and may run from any thread."""

    def __init__(self, cap):
        self.cap, self.used, self.held = cap, 0, {}
        self.lock = threading.Lock()
        self.loop = self.cond = None
        self.peak = self.waited = 0

    async def acquire(self, rid, tokens):
        if tokens <= 0:
            return
        if self.cond is None:
            self.loop, self.cond = asyncio.get_running_loop(), asyncio.Condition()
        async with self.cond:
            if not self.fits(tokens):
                self.waited += 1
            await self.cond.wait_for(lambda: self.fits(tokens))
            with self.lock:
                self.used += tokens
                self.held[rid] = self.held.get(rid, 0) + tokens
                self.peak = max(self.peak, self.used)

    def fits(self, tokens):
        with self.lock:
            return self.used == 0 or self.used + tokens <= self.cap

    def release(self, rid):
        with self.lock:
            tokens = self.held.pop(rid, 0)
            self.used -= tokens
        if tokens and self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._notify()))
        return tokens

    async def _notify(self):
        async with self.cond:
            self.cond.notify_all()


BUDGET = None


def rids_of(obj):
    rid = getattr(obj, "rid", None)
    return [rid] if isinstance(rid, str) else [r for r in (rid or ()) if isinstance(r, str)]


def image_tokens(width, height):
    """Inkling2ImageProcessor.count_tokens_for_image."""
    if width <= 0 or height <= 0:
        return 0
    return -(-width // CELL) * -(-height // CELL)


def size_of(image, load_bytes):
    """(width, height) from the header (a PIL image: its size), without decoding pixels."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.size
    with Image.open(io.BytesIO(load_bytes(image))) as opened:
        return opened.size


def check(images, cap, load_bytes):
    """Raise ValueError when the images of one request are above `cap` vision tokens together."""
    sizes = [size_of(image, load_bytes) for image in images]
    total = sum(image_tokens(w, h) for w, h in sizes)
    if total > cap:
        shown = ", ".join(f"{w} x {h}" for w, h in sizes[:4]) + (", ..." if len(sizes) > 4 else "")
        raise ValueError(f"the request's {len(sizes)} image(s) ({shown} pixels) are {total} vision tokens; this "
                         f"server accepts at most {cap} per request ({ENV}, MM_MAX_IMAGE_TOKENS in scripts/launch.sh); "
                         "downscale or split them")
    return total


def _patch(module):
    cls = getattr(module, "Inkling2ImageProcessor", None)
    load_bytes = getattr(module, "_load_image_bytes", None)
    original = getattr(cls, "preprocess", None)
    counter = getattr(cls, "count_tokens_for_image", None)
    if (load_bytes is None or original is None or getattr(cls, "__call__", None) is not original or counter is None
            or counter(None, 41, 40) != image_tokens(41, 40)):
        raise RuntimeError("inkling_mm: unexpected Inkling2 image processor (preprocess, __call__, _load_image_bytes "
                           "or the token count changed)")
    cap = STATE["cap"]

    @functools.wraps(original)
    def preprocess(self, images, return_tensors="pt"):
        check(images if isinstance(images, (list, tuple)) else [images], cap, load_bytes)
        return original(self, images, return_tensors)

    cls.preprocess = cls.__call__ = preprocess


def _patch_processor(module):
    """Admit a request's images against the in-flight budget before any is decoded (URLs are downloaded first, then
    sized from their headers); the resolved bytes go on to the stock processor, so nothing is fetched twice."""
    cls = getattr(module, "Inkling2MultimodalProcessor", None)
    original = getattr(cls, "process_mm_data_async", None)
    resolve = getattr(module, "_resolve_image", None)
    if original is None or resolve is None or not asyncio.iscoroutinefunction(original):
        raise RuntimeError("inkling_mm: unexpected Inkling2 multimodal processor (process_mm_data_async / _resolve_image)")
    from sglang.srt.multimodal.inkling2 import image_processing

    load_bytes, cap = image_processing._load_image_bytes, STATE["cap"]

    @functools.wraps(original)
    async def process_mm_data_async(self, image_data=None, audio_data=None, video_data=None, input_text="",
                                    request_obj=None, **kwargs):
        images = list(image_data or [])
        rids = rids_of(request_obj)
        if not images or BUDGET is None or len(rids) != 1:
            return await original(self, image_data, audio_data, video_data, input_text, request_obj, **kwargs)

        def size():
            resolved = [resolve(image) for image in images]
            return resolved, sum(image_tokens(*size_of(image, load_bytes)) for image in resolved)

        resolved, tokens = await asyncio.get_running_loop().run_in_executor(self.io_executor, size)
        if cap and tokens > cap:
            check(resolved, cap, load_bytes)  # raises with the sizes
        await BUDGET.acquire(rids[0], tokens)
        try:
            weakref.finalize(request_obj, BUDGET.release, rids[0])  # a request that never reaches the release below
        except TypeError:
            pass
        try:
            return await original(self, resolved, audio_data, video_data, input_text, request_obj, **kwargs)
        except BaseException:
            BUDGET.release(rids[0])
            raise

    cls.process_mm_data_async = process_mm_data_async


def _patch_tokenizer(module):
    cls = getattr(module, "TokenizerManager", None)
    original = getattr(cls, "generate_request", None)
    import inspect

    if original is None or not inspect.isasyncgenfunction(original) or (
            tuple(inspect.signature(original).parameters) != ("self", "obj", "request")):
        raise RuntimeError("inkling_mm: unexpected TokenizerManager.generate_request")

    @functools.wraps(original)
    async def generate_request(self, obj, request=None):
        responses = original(self, obj, request)
        try:
            async for response in responses:
                yield response
        finally:
            try:
                await responses.aclose()
            finally:
                for rid in rids_of(obj):
                    BUDGET.release(rid)

    cls.generate_request = generate_request


def install(env=None):
    global BUDGET
    if STATE["installed"]:
        return STATE["cap"]
    cap = cap_from_env(env)
    inflight = cap_from_env(env, ENV_INFLIGHT, 1 << 24)
    if inflight and cap and inflight < cap:
        raise RuntimeError(f"inkling_mm: {ENV_INFLIGHT}={inflight} is below {ENV}={cap}")
    STATE.update(installed=True, cap=cap, inflight=inflight)
    from inkling_attn._hook import when_imported

    if cap:
        when_imported(MODULE, _patch)
        sys.stderr.write(f"[inkling_mm pid={os.getpid()}] requests with images above {cap} vision tokens are refused\n")
    if inflight:
        BUDGET = Budget(inflight)
        when_imported(PROCESSOR, _patch_processor)
        when_imported(TOKENIZER, _patch_tokenizer)
        sys.stderr.write(f"[inkling_mm pid={os.getpid()}] image requests in flight are admitted up to {inflight} "
                         "vision tokens together\n")
    return cap
