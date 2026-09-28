"""API-stability guard for laya.evals: an accidental rename or a torch import fails here.

Run: python tests/test_evals_api.py
"""
import dataclasses
import inspect
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import evals  # noqa: E402

PASS, FAIL = [], []


def check(name, got, expected):
    ok = got == expected
    (PASS if ok else FAIL).append(name)
    print(("  ok   " if ok else "  FAIL ") + name, "" if ok else "(got %r, expected %r)" % (got, expected))


def check_true(name, condition):
    check(name, bool(condition), True)


# --------------------------------------------------------------- fields
check("Example fields",
      [f.name for f in dataclasses.fields(evals.Example)],
      ["state", "questions", "expected", "tags", "language", "model"])
check("EvalReport fields",
      [f.name for f in dataclasses.fields(evals.EvalReport)],
      ["config", "overall", "slices", "cases"])

# --------------------------------------------------------------- evaluators
check("default evaluator names",
      sorted(e.name for e in evals.default_evaluators()),
      ["choice_accuracy", "mean_confidence", "noul_accuracy", "score_mae"])
check_true("ScoreWithin names its tolerance", evals.ScoreWithin(0.25).name == "score_within_0.25")

# --------------------------------------------------------------- exports / callables
for name in ("Dataset", "Example", "EvalError", "EvalReport", "evaluate", "ece", "assert_regression"):
    check_true("laya.evals.%s exists" % name, hasattr(evals, name))

# --------------------------------------------------------------- evaluate's call signature
# `evaluate` is the harness entry point callers extend from their own scripts, so its parameter
# names and order are the contract. `sort_by_length` is core's #294 forward-pass grouping knob
# reaching a scored run through `laya-evals run --sort-by-length`; it is appended, never inserted,
# because a caller that passed `on_error` or `config` positionally must still mean the same thing.
_sig = inspect.signature(evals.evaluate)
check("evaluate signature", [p.name for p in _sig.parameters.values()],
      ["runner", "dataset", "evaluators", "batch_size", "on_error", "config", "sort_by_length"])
check("evaluate: grouping defaults to off", _sig.parameters["sort_by_length"].default, False)
check_true("evaluate: the knob sits after the batch size it groups inside",
           list(_sig.parameters).index("batch_size")
           < list(_sig.parameters).index("sort_by_length"))

# The two batch entry points the CLI wires up: both have to take the knob by the same name, or a
# `--sort-by-length` run reports `sort_by_length_sent: false` for the surface it ships.
from laya import evals_cli  # noqa: E402

for label, fn in (("RouterRunner.predict_batch", evals_cli.RouterRunner.predict_batch),
                  ("OnnxRunner.predict_batch", evals_cli.OnnxRunner.predict_batch)):
    params = inspect.signature(fn).parameters
    check_true("%s takes the grouping knob" % label, "sort_by_length" in params)
    check("%s defaults it to off" % label, params["sort_by_length"].default, False)

# A signature is not a wiring: a runner that declares the knob and drops it on the way through
# would pass both checks above. `OnnxRunner`'s call is gated in test_evals_onnx.py against a
# recording agent; this is the same witness for the router-shaped runner the CLI uses by default.
class _RecordingRouter:
    def __init__(self):
        self.calls = []

    def predict_batch(self, requests, batch_size=None, sort_by_length=False):
        self.calls.append({"batch_size": batch_size, "sort_by_length": sort_by_length})
        return [{"model": "m", "answers": {}} for _ in requests]


_router = _RecordingRouter()
_runner = evals_cli.RouterRunner(_router)
_runner.predict_batch(["s1", "s2"], {"intent": {"type": "choice"}}, batch_size=2, sort_by_length=True)
check("RouterRunner hands the grouping to the router it wraps",
      _router.calls[-1], {"batch_size": 2, "sort_by_length": True})
_runner.predict_batch(["s1", "s2"], {"intent": {"type": "choice"}}, batch_size=2)
check("RouterRunner does not invent it for a run that did not ask",
      _router.calls[-1], {"batch_size": 2, "sort_by_length": False})

# --------------------------------------------------------------- torch stays out of import
probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; import laya.evals; sys.exit(1 if 'torch' in sys.modules else 0)"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
check("import laya.evals does not import torch", probe.returncode, 0)

# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for name in FAIL:
    print("  FAIL " + name)
if not FAIL:
    print("all eval API checks passed")
sys.exit(1 if FAIL else 0)
