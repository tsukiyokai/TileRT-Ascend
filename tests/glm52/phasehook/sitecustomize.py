"""Split a decode round into its phases, with no profiler and no harness edit.

At width 8 `advance()` is one seed forward when the chain needs reseeding,
seven chained draft forwards, one full-model verify, one target-conditioned
draft forward, then `host_tokens` and the commit. Each returns a value the
host reads, so each is synchronous and host wall per call is a fair split.

The phase names and the way a draft call is classified follow `_Draft` in
`tilert/models/glm_5_2_ascend/profiling.py`: a call carrying
`external_indices` is a chain step, one after a verify is the conditioned
pass, and otherwise it is a seed.

A profiler is deliberately not used: it adds host work to a device the host
feeds, so it changes the durations it reports. This costs two clock reads a
call, about twenty-two against a 57 ms round.

Printed at exit on rank 0. `PHASE_HOOK=off` disables it.
"""

import atexit
import builtins
import collections
import os
import sys
import time

DRAFT_MODULE = "tilert.models.glm_5_2_ascend.cann_mtp"
TARGET_MODULE = "tilert.models.glm_5_2_ascend.cann_model"

totals = collections.Counter()
counts = collections.Counter()
verified = [False]


def _record(name, begin):
    totals[name] += time.perf_counter_ns() - begin
    counts[name] += 1


def _wrap_draft(draft_class):
    original = draft_class.forward

    def forward(self, tokens, parent, **kwargs):
        if kwargs.get("external_indices") is not None:
            name = "chain"
        else:
            name = "conditioned" if verified[0] else "seed"
        begin = time.perf_counter_ns()
        try:
            return original(self, tokens, parent, **kwargs)
        finally:
            _record(name, begin)
            if name == "conditioned":
                verified[0] = False

    draft_class.forward = forward


def _wrap_target(model_class):
    for name in ("verify", "host_tokens"):
        original = getattr(model_class, name)

        def timed(self, *args, _name=name, _original=original, **kwargs):
            begin = time.perf_counter_ns()
            try:
                return _original(self, *args, **kwargs)
            finally:
                _record(_name, begin)
                if _name == "verify":
                    verified[0] = True

        setattr(model_class, name, timed)


def _report():
    rounds = counts["verify"]
    if not rounds or os.environ.get("RANK", "0") != "0":
        return
    print(f"[phase] {rounds} rounds", flush=True)
    accounted = 0
    for name, total in totals.most_common():
        accounted += total
        print(f"[phase] {name:12} {total / 1e6 / rounds:8.3f} ms a round  "
              f"{counts[name] / rounds:5.2f} calls  "
              f"{total / 1e6 / counts[name]:8.3f} ms a call", flush=True)
    print(f"[phase] accounted  {accounted / 1e6 / rounds:8.3f} ms a round",
          flush=True)


def _install_when_ready():
    """Wrap each class once its own module has finished executing.

    A name appears in `sys.modules` before its body runs, so the guard is on
    the attribute rather than on the module.
    """
    original_import = builtins.__import__
    pending = {"draft": True, "target": True}

    def patched(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        draft = sys.modules.get(DRAFT_MODULE)
        if pending["draft"] and draft is not None and hasattr(draft, "CannMtp"):
            pending["draft"] = False
            _wrap_draft(draft.CannMtp)
        target = sys.modules.get(TARGET_MODULE)
        if pending["target"] and target is not None and hasattr(target, "CannModel"):
            pending["target"] = False
            _wrap_target(target.CannModel)
        if not any(pending.values()):
            builtins.__import__ = original_import
            print("[phase] hook installed", flush=True)
        return module

    builtins.__import__ = patched


def _chain_other_hooks():
    """Run any other sitecustomize this run needs, by path.

    Python imports the name `sitecustomize` once, so two directories on
    PYTHONPATH each holding one means only the first is ever executed. The
    static-kernel cache fix is installed by another one, and silently losing it
    would change what the measured binary is -- which is exactly the class of
    error this instrument exists to avoid.
    """
    import importlib.util

    for path in os.environ.get("PHASE_HOOK_CHAIN", "").split(os.pathsep):
        if not path or not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("_chained_hook", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"[phase] chained {path}", flush=True)


if os.environ.get("PHASE_HOOK", "on") != "off":
    atexit.register(_report)
    _install_when_ready()
_chain_other_hooks()
