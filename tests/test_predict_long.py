"""predict_long: scan a state longer than the window and aggregate per question.

Weight-free. The real forward path is stubbed (predict_batch returns canned per-window answers),
so this checks only predict_long's own logic: the fits-in-one-window short-circuit, the overlapping
window split, the per-type aggregation (noul = strongest window, choice/score = most-confident
window), and the per-call hook controls it forwards to whichever of those two calls runs. Numerical
behaviour on real weights is exercised in tests/test_local_e2e.py.
"""
import os
import re
import sys
import threading
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya.agent import Agent  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name if got == want else "%s: got %r want %r" % (name, got, want))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: raised %r not %s" % (name, e, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s: %s" % (name, detail))


class _Tok:
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False):
        # token count == character count, so the test controls windowing by string length
        return {"input_ids": list(range(len(text)))}

    def decode(self, ids):
        return "w%d_%d" % (ids[0], ids[-1]) if ids else "w"


def make_agent(batch_result_fn):
    a = Agent.__new__(Agent)
    a.cfg = {"max_len": 100, "head_max_len": 20}   # budget = max(64, 100-20-8) = 72
    a.tok = _Tok()
    a._to_internal = staticmethod(Agent._to_internal).__func__
    a._calls = {"system_one": 0, "batch_states": None, "system_one_kwargs": None,
                "batch_kwargs": None}

    def _system_one(state, questions, lang=None, **controls):
        a._calls["system_one"] += 1
        a._calls["system_one_kwargs"] = controls
        return {"model": "laya-rl-agent", "answers": {"_via": "system_one"}, "usage": {"input_tokens": 1}}

    def _predict_batch(states, questions, batch_size=None, lang=None, **controls):
        a._calls["batch_states"] = list(states)
        a._calls["batch_kwargs"] = controls
        return batch_result_fn(list(states), questions)

    a.system_one = _system_one
    a.predict_batch = _predict_batch
    return a


def make_real_agent():
    """`predict_long` on the real `predict_batch`/`system_one`, with only the three composed
    helpers stubbed -- the harness `tests/test_hooks.py` uses, so no weights are involved.

    Why both agents: the stubbed `predict_batch` above is the right instrument for what
    `predict_long` *forwards*, but it never runs a hook, so anything it says about a hook answering
    rests on a result count alone. Here the hook chain really runs and every forward pass is
    counted, which is the difference between "a hook answered the document" and "a hook replaced
    the window list" -- both change the count.
    """
    a = Agent.__new__(Agent)
    a.cfg = {"max_len": 100, "head_max_len": 20}       # budget = max(64, 100-20-8) = 72
    a.tok = type("Tok", (_Tok,), {"pad_token_id": 0})()
    a._to_internal = staticmethod(Agent._to_internal).__func__
    a.model_id = "convaiinnovations/laya"
    a.hooks = []
    a.hooks_raise = True
    a.hooks_timeout = None
    a._hooks_lock = threading.Lock()
    a._forward_calls, a._encoded = [], []

    def _encode_state(state, ids, internal, **overrides):
        a._encoded.append(state)
        return [{"ids": [1, 2, 3], "markers": [0, 1], "qtype": 2} for _ in ids]

    def _forward(b):
        n = b["input_ids"].shape[0]
        a._forward_calls.append(n)
        return np.zeros((n, 2), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode_answers(logits, act, items, ids, internal, row, **kw):
        conf = 0.4 + 0.05 * row                        # so the last window decides
        return {"dept": {"type": "choice", "choice": "a", "confidence": conf,
                         "answer_confidence": conf},
                "flag": {"type": "noul", "noul": conf, "confidence": conf,
                         "answer_confidence": conf}}

    a._encode_state = _encode_state
    a._forward = _forward
    a._decode_answers = _decode_answers
    return a


Q = {"dept": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
     "flag": {"type": "noul", "instructions": "?"}}

# 1. fits in one window -> delegates to system_one, no windowing
a = make_agent(lambda s, q: [])
short = a.predict_long({"body": "x" * 50}, Q)   # 50 tokens <= budget 72
check("short/delegates to system_one", short["answers"], {"_via": "system_one"})
check("short/no predict_batch call", a._calls["batch_states"], None)
check("short/one window is reported", short["usage"].get("windows", "<absent>"), 1)
check("short/system_one's own usage is kept", short["usage"].get("input_tokens", "<absent>"), 1)

# 1b. the count is added to a copy, because a start hook that skips hands back the caller's own
# payload dict and that object may be cached and reused
payload = {"model": "laya-rl-agent", "answers": {"_via": "system_one"}, "usage": {"input_tokens": 1}}
a = make_agent(lambda s, q: [])
a.system_one = lambda state, questions, lang=None, **controls: payload
via_hook = a.predict_long({"body": "x" * 50}, Q)
check("short/a new result dict comes back", via_hook is payload, False)
check("short/the caller's payload is not written to", payload["usage"], {"input_tokens": 1})

# 2. long state -> overlapping windows, aggregated per question
def canned(states, q):
    # one canned answer per window; the 3rd window is the confident/positive one
    out = []
    for i, _ in enumerate(states):
        conf = 0.9 if i == 2 else 0.4
        ptrue = 0.95 if i == 2 else 0.1
        out.append({"answers": {
            "dept": {"type": "choice", "choice": "b" if i == 2 else "a",
                     "probabilities": {"a": 1 - conf, "b": conf}, "confidence": conf,
                     "answer_confidence": conf, "action": {"act_probability": 1.0}},
            "flag": {"type": "noul", "noul": ptrue, "confidence": max(ptrue, 1 - ptrue),
                     "answer_confidence": max(ptrue, 1 - ptrue), "action": {"act_probability": 1.0}},
        }, "usage": {"input_tokens": 10}})
    return out


a = make_agent(canned)
# 300 tokens, budget 72, stride 36 -> several overlapping windows, last covers the tail
res = a.predict_long({"body": "y" * 300}, Q)
nwin = len(a._calls["batch_states"])
check("long/windows recorded in usage", res["usage"]["windows"], nwin)
check("long/more than one window", nwin > 1, True)
check("long/overlap: stride is half the budget", a._calls["batch_states"][1], "w36_107")
check("long/choice = most-confident window", res["answers"]["dept"]["choice"], "b")
check("long/noul = strongest window", res["answers"]["flag"]["noul"], 0.95)
check("long/usage sums window tokens", res["usage"]["input_tokens"], 10 * nwin)
# the deciding window is named on each answer (window index 2 is the confident/positive one)
check("long/choice names the deciding window", res["answers"]["dept"]["window"]["index"], 2)
check("long/noul names the deciding window", res["answers"]["flag"]["window"]["index"], 2)
check("long/window start is the 3rd overlap offset", res["answers"]["dept"]["window"]["token_start"], 72)
check("long/window carries the count", res["answers"]["flag"]["window"]["count"], nwin)

# 3. only aggregate="auto" is supported
a = make_agent(canned)
check_raises("aggregate/rejects unknown mode", ValueError,
             lambda: a.predict_long({"body": "y" * 300}, Q, aggregate="mean"))

# 4. the per-call hook controls `predict`/`system_one`/`predict_batch` take reach the scan
HOOK_KEYS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")
LONG = {"body": "y" * 300}
sentinel = object()


def sentinel_start(ctx):
    pass


a = make_agent(canned)
res = a.predict_long(LONG, Q, hooks=[sentinel], on_predict_start=sentinel_start,
                     hooks_raise=False, hooks_timeout=2.5)
got = a._calls["batch_kwargs"]
check("hooks/forwarded to the scan", sorted(got), sorted(HOOK_KEYS))
check("hooks/the hook list reaches it", got.get("hooks"), [sentinel])
check("hooks/hooks_raise reaches it", got.get("hooks_raise"), False)
check("hooks/hooks_timeout reaches it", got.get("hooks_timeout"), 2.5)
started = got.get("on_predict_start")
started = started if isinstance(started, list) else ([] if started is None else [started])
check("hooks/the caller's start hook still runs first",
      bool(started) and started[0] is sentinel_start, True)
check("hooks/end stays absent when the caller sets none", got.get("on_predict_end"), None)

# 5. the same controls reach the fits-in-one-window path
a = make_agent(lambda s, q: [])
a.predict_long({"body": "x" * 50}, Q, hooks=[sentinel], hooks_timeout=1.25)
got = a._calls["system_one_kwargs"]
check("hooks/forwarded to system_one", sorted(got), sorted(HOOK_KEYS))
check("hooks/the hook list reaches it there", got.get("hooks"), [sentinel])
check("hooks/hooks_timeout reaches it there", got.get("hooks_timeout"), 1.25)

# 6. what the scan forwards is the window texts, in scan order, not the caller's state object
a = make_agent(canned)
a.predict_long(LONG, Q)
states = a._calls["batch_states"]
check("states/several windows", len(states) > 1, True)
check("states/decoded text, not the caller's dict",
      all(isinstance(s, str) for s in states) and states[0] == "w0_71", True)
check("states/they overlap", a._calls["batch_states"][1], "w36_107")

# 7. forwarding the controls must not move a decision
plain = make_agent(canned).predict_long(LONG, Q)
with_hooks = make_agent(canned).predict_long(LONG, Q, hooks=[sentinel], hooks_timeout=9)
check("hooks/forwarding is decision-neutral", with_hooks, plain)

# 8. a start hook that answers the document, on the engine's own evidence
DOC = {"model": "laya-rl-agent",
       "answers": {"dept": {"type": "choice", "choice": "b", "answer_confidence": 0.9},
                   "flag": {"type": "noul", "noul": 0.2, "answer_confidence": 0.8}},
       "usage": {"input_tokens": 3, "output_tokens": 0}}

# The control first: how many windows this state really splits into, measured from the forward
# passes it costs rather than from the key under test. Two questions, one row each per window.
scan = make_real_agent()
base = scan.predict_long(LONG, Q)
nwin = sum(scan._forward_calls) // 2
check("scan/more than one window", nwin > 1, True)
check("scan/the scan reports the windows it read", base["usage"]["windows"], nwin)
check("scan/the deciding window is the last",
      base["answers"]["dept"]["window"]["index"], nwin - 1)

# The instrument for sections 8-8c: a hook that answers must return, and a hook that replaces the
# window list must raise. Letting either unexpected case propagate would abort the suite and hide
# every check after it, so each call is captured and its type asserted by a named check.
def _attempt(fn):
    """Call `fn`, returning (value, exception)."""
    try:
        return fn(), None
    except BaseException as exc:  # noqa: BLE001
        return None, exc


def _kind(exc):
    return exc.__class__.__name__ if exc else None


def _narrow(keep):
    """A start hook that truncates the window list and lets inference run on what is left."""
    def hook(ctx):
        ctx.states = list(ctx.states)[:keep]
    return hook


a = make_real_agent()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res, skip_exc = _attempt(
        lambda: a.predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.skip([DOC])))
check("skip/one payload for the document is never refused as a count mismatch", _kind(skip_exc), None)
check("skip/no forward pass ran", a._forward_calls, [])
usage, answers = (res or {}).get("usage", {}), (res or {}).get("answers", {})
check("skip/no window was scored", usage.get("windows", "<absent>"), 0)
check("skip/the payload tokens are kept", usage.get("input_tokens", "<absent>"), 3)
check("skip/the hook's answer is returned", answers.get("dept", {}).get("choice"), "b")
check("skip/no deciding window is claimed",
      sorted(k for v in answers.values() for k in v if k == "window"), [])
check("skip/the caller is told", [w.category.__name__ for w in caught], ["RuntimeWarning"])

# 8b. a start hook that replaces the window list is the other reading of the same count, and it is
# not a hook answer: inference ran on the states the hook left behind.
a = make_real_agent()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _, narrow_exc = _attempt(lambda: a.predict_long(LONG, Q, on_predict_start=_narrow(1)))
narrow_err = str(narrow_exc) if narrow_exc else "<no error>"
check("narrow/rejects a list shortened to one state", _kind(narrow_exc), "ValueError")
# Why this check sits next to the rejection: the forward is the proof that this was a rewrite and
# not an answer, and it is what made the two cases indistinguishable before.
check("narrow/a forward did run, so nothing was answered before inference",
      a._forward_calls, [2])
check("narrow/the rewrite is not called a hook answer",
      [w.category.__name__ for w in caught], [])
check_true("narrow/the error names the split it could not attribute",
           ("split into %d windows" % nwin) in narrow_err, narrow_err)
check_true("narrow/the error offers ctx.skip as the way to answer",
           "ctx.skip" in narrow_err, narrow_err)
a = make_real_agent()
check_raises("narrow/rejects a list shortened to two states", ValueError,
             lambda: a.predict_long(LONG, Q, on_predict_start=_narrow(2)))
# the other direction, so the rule is the count rather than a hook that dropped something. The
# error must name the caller's split: `predict_long` hands the hook a copy, so an in-place
# `append` grows the hook's list and leaves the attributed windows alone.
a = make_real_agent()
_, added_exc = _attempt(
    lambda: a.predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.states.append("invented")))
added_err = str(added_exc) if added_exc else "<no error>"
check("grow/rejects a list with a state added", _kind(added_exc), "ValueError")
check_true("grow/the count is the caller's split, not the hook's list",
           ("split into %d windows" % nwin) in added_err, added_err)

# 8c. rewriting the windows themselves stays supported: the count holds, so the attribution does
a = make_real_agent()
res, exc = _attempt(
    lambda: a.predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.states.extend(())))
check("rewrite/a no-op on the list scans as usual", _kind(exc), None)
check("rewrite/a no-op on the list keeps the count",
      ((res or {}).get("usage") or {}).get("windows", "<absent>"), nwin)
a = make_real_agent()
res, exc = _attempt(lambda: a.predict_long(
    LONG, Q, on_predict_start=lambda ctx: setattr(ctx, "states", [s.upper() for s in ctx.states])))
check("rewrite/the hook's text is what was scored", (a._encoded or [""])[0].isupper(), True)
check("rewrite/the same count is the same scan", _kind(exc), None)
check("rewrite/the count is unchanged", ((res or {}).get("usage") or {}).get("windows", "<absent>"), nwin)
check("rewrite/the deciding window is still named",
      ((res or {}).get("answers") or {}).get("dept", {}).get("window", {}).get("index"), nwin - 1)

# 8d. the fits-in-one-window path answers to a hook too, so 0 keeps meaning "no window scored this"
a = make_real_agent()
short_hooked, exc = _attempt(
    lambda: a.predict_long({"body": "x" * 50}, Q, on_predict_start=lambda ctx: ctx.skip([DOC])))
check("short skip/no forward pass", a._forward_calls, [])
check("short skip/a hook answer is not counted as a window read",
      ((short_hooked or {}).get("usage") or {}).get("windows", "<absent>"), 0)
a = make_real_agent()
short_plain, exc = _attempt(lambda: a.predict_long({"body": "x" * 50}, Q))
check("short skip/the same state without a hook is one window",
      ((short_plain or {}).get("usage") or {}).get("windows", "<absent>"), 1)

# 9. a batch call that returns the wrong count for its own states is an error, not a guess
a = make_agent(lambda s, q: [{"answers": {}, "usage": {}}] * 2)
check_raises("count/rejects a result count matching nothing", ValueError,
             lambda: a.predict_long(LONG, Q))

# 10. the count is total, so one question works on every result predict_long can return
short_all = make_agent(lambda s, q: []).predict_long({"body": "x" * 50}, Q)
scan_agent = make_agent(canned)
scanned_all = scan_agent.predict_long(LONG, Q)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hook_all, hook_exc = _attempt(
        lambda: make_real_agent().predict_long(LONG, Q, on_predict_start=lambda ctx: ctx.skip([DOC])))
check("total/a hook answer returns rather than being refused", _kind(hook_exc), None)
check("total/no path leaves the key out",
      [(r or {}).get("usage", {}).get("windows", "<absent>")
       for r in (short_all, scanned_all, hook_all)],
      [1, len(scan_agent._calls["batch_states"]), 0])
check("total/0 separates a hook answer from a single-window one",
      ((hook_all or {}).get("usage", {}).get("windows", "<absent>"),
       short_all["usage"].get("windows", "<absent>")),
      (0, 1))


# 11. the page that teaches this key teaches the value the code actually writes
README = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "README.md"), encoding="utf-8").read()
bullet = README[README.index("A state that already fits one window"):][:600]
check("docs/README pins the single-window value", 'usage["windows"] = 1' in bullet, True)
check("docs/README states the key is total", "The key is total" in bullet, True)
check("docs/README documents all three counts", sorted(set(re.findall(r"`([0-9N])`", bullet))),
      ["0", "1", "N"])
# The two rules this branch added, pinned where they are taught rather than in the code comment
hooks_bullet = README[README.index("Hooks wrap the inference that answers the state"):][:800]
check("docs/README says a hook may not change the window count or order",
      "reorder windows" in hooks_bullet, True)
check("docs/README names ctx.skip as the way to answer", "ctx.skip(...)" in hooks_bullet, True)
API = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs", "hooks", "api.md"), encoding="utf-8").read()
api_para = API[API.index("On `predict_long` the hooks wrap"):][:1200]
check("docs/api.md matches: count and order are fixed",
      "or their" in api_para and "number of states raises" in api_para, True)
check("docs/api.md matches: the hook answer reads as zero windows",
      'usage["windows"]` at `0' in api_para, True)
check("docs/predict_long's docstring says the key is total",
      "always present" in (Agent.predict_long.__doc__ or ""), True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
