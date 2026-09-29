"""Source-bound adaptive prefill scheduling, not a KV precision change.

Bound query rows T * (history + T). The byte coefficient is a conservative
planning estimate, NOT proof of full-model physical-memory capacity.
Installation defers scheduler patching until normal module import completes.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import logging
import math
import os
import sys

LOG = logging.getLogger(__name__)
SCHEDULER = 'sglang.srt.managers.scheduler'
BACKEND = 'sglang.srt.layers.attention.deepseek_v4_backend'


class BudgetChunkSizer:
    def __init__(self,budget_tokens2,chunk_min,chunk_max,page=256):
        if (isinstance(budget_tokens2,bool) or not isinstance(budget_tokens2,(float,int))
                or not math.isfinite(budget_tokens2) or budget_tokens2<=0
                or any(type(x) is not int or x<=0 for x in (chunk_min,chunk_max,page))
                or chunk_min%page or chunk_max%page or chunk_min>chunk_max):
            raise ValueError('finite positive budget; positive integer page-aligned min<=max required')
        self.budget=int(budget_tokens2)
        self.cmin,self.cmax,self.page=chunk_min,chunk_max,page

    def predict(self,history_len):
        if type(history_len) is not int or history_len<0:
            raise ValueError('history must be a nonnegative integer')
        # Integer arithmetic avoids cancellation at long prefixes.
        t=min(self.cmax,(math.isqrt(history_len*history_len+4*self.budget)-history_len)//2)
        t=t//self.page*self.page
        return max(self.cmin, min(self.cmax, t))


def settings(env=None):
    env=os.environ if env is None else env
    cfg=dict(budget=float(env.get('DSV41_CHUNK_BUDGET_TOKENS2','3e8')),
             cmin=int(env.get('DSV41_CHUNK_MIN','256')),
             cmax=int(env.get('DSV41_CHUNK_MAX','512')),
             score_mib=int(env.get('DSV41_INDEXER_SCORE_BUDGET_MIB','1024')),
             k_chunk_max=int(env.get('DSV41_INDEXER_K_CHUNK_MAX','2048')))
    s=BudgetChunkSizer(cfg['budget'],cfg['cmin'],cfg['cmax'])
    s.predict(0)
    s.predict(1048576)
    if not 1<=cfg['score_mib']<=1024:raise ValueError('indexer copy budget must be 1..1024 MiB')
    if type(cfg['k_chunk_max']) is not int or not 1<=cfg['k_chunk_max']<=16384:
        raise ValueError('indexer K-slice cap must be 1..16384')
    return cfg


def patch_scheduler(module,cfg):
    cls=module.Scheduler
    orig=cls.maybe_init_dynamic_chunk_sizer
    if getattr(orig,'_dsv41_budget',None)==cfg:return
    if getattr(orig,'_dsv41_budget',None) is not None:raise RuntimeError('budget hook configuration drift')
    if list(inspect.signature(orig).parameters)!=['self']:
        raise RuntimeError('SGLang scheduler sizer signature drift')
    def patched(self):
        orig(self)
        if self.dynamic_chunk_sizer is not None:
            raise RuntimeError('DSV41 budget hook cannot replace an existing PP sizer')
        if self.page_size!=256 or self.chunked_prefill_size not in (256,512):
            raise RuntimeError('DSV41 budget requires page256 and static-first chunk256/512')
        cmax=min(cfg['cmax'], int(self.chunked_prefill_size))
        self.dynamic_chunk_sizer=BudgetChunkSizer(cfg['budget'],cfg['cmin'],cmax)
        LOG.warning('DSV41_ADAPTIVE_CHUNK_ACTIVE budget=%s min=%s max=%s first=%s',cfg['budget'],cfg['cmin'],cmax,self.chunked_prefill_size)
    patched._dsv41_budget=dict(cfg)
    cls.maybe_init_dynamic_chunk_sizer=patched


def _apply(name,module,cfg):
    if name==SCHEDULER:patch_scheduler(module,cfg)
    elif name==BACKEND:
        if not hasattr(module,'_TORCH_INDEXER_SCORE_BUDGET_BYTES'):
            raise RuntimeError('SGLang indexer budget constant missing')
        module._TORCH_INDEXER_SCORE_BUDGET_BYTES=cfg['score_mib']<<20
        module._TORCH_INDEXER_K_CHUNK_MAX=cfg['k_chunk_max']
        from sglang_patch.torch_indexer_budget import apply_to_backend
        apply_to_backend(module)


class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self,cfg):self.cfg=cfg
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in (SCHEDULER,BACKEND):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None:raise ImportError('missing pinned module '+fullname)
        base=spec.loader;cfg=self.cfg
        class Loader(importlib.abc.Loader):
            def create_module(self,spec):
                return base.create_module(spec) if hasattr(base,'create_module') else None
            def exec_module(self,module):
                base.exec_module(module)
                _apply(fullname,module,cfg)
        spec.loader=Loader()
        return spec


def install():
    cfg=settings()
    old=next((f for f in sys.meta_path if isinstance(f,_AfterImport)),None)
    if old is not None:
        if old.cfg!=cfg:raise RuntimeError('budget hook reinstalled with different settings')
        return
    for name in (SCHEDULER,BACKEND):
        if name in sys.modules:_apply(name,sys.modules[name],cfg)
    sys.meta_path.insert(0,_AfterImport(cfg))
