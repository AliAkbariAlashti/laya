"""API-stability guard for prediction hooks.

These tests pin the public hook surface (parameter names, kinds, defaults, context fields,
lifecycle events, exports) so a change that would break callers fails here first. If a change
is intentional, update this file in the same commit.

The cache key the examples and docs teach is pinned here too: `ctx.skip()` hands back whatever
the key matched, so what a key covers is part of the contract, not an implementation detail.

Run: python tests/test_hooks_api.py
"""
import contextlib
import dataclasses
import functools
import hashlib
import inspect
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya import Agent, AsyncHook, BaseHook, PredictContext, PredictHook, Router, load  # noqa: E402
from laya.hooks import HOOK_EVENTS, Hook  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402
from laya.router import _question_schema  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def sig(fn):
    return inspect.signature(fn).parameters


def check_param(name, fn, param, default, kind=None):
    params = sig(fn)
    if param not in params:
        FAIL.append("%s/%s: missing parameter" % (name, param))
        return
    p = params[param]
    check("%s/%s default" % (name, param), p.default, default)
    if kind is not None:
        check("%s/%s kind" % (name, param), p.kind, kind)


HOOK_KEYS = {
    "hooks": None,
    "on_predict_start": None,
    "on_predict_end": None,
    "hooks_raise": True,
    "hooks_concurrent": True,
    "hooks_timeout": None,
}

# --------------------------------------------------------------- constructors
for label, fn in (("Agent.__init__", Agent.__init__), ("load", load),
                  ("Router.__init__", Router.__init__), ("ONNXAgent.__init__", ONNXAgent.__init__)):
    for param, default in HOOK_KEYS.items():
        check_param(label, fn, param, default)

# Router keeps lang_guess and explicit per-model revisions too
check_param("Router.__init__", Router.__init__, "lang_guess", None)
check_param("Router.__init__", Router.__init__, "revisions", None)
# ...and the remaining `Agent` options, forwarded to every checkpoint it builds
check_param("Router.__init__", Router.__init__, "agent_kwargs", None)
# ...and the per-checkpoint digests that `revisions` has always had a sibling need for
check_param("Router.__init__", Router.__init__, "sha256_digests", None)

# --------------------------------------------------------------- predict surfaces
for label, fn in (("Agent.predict_batch", Agent.predict_batch),
                  ("Agent.system_one", Agent.system_one),
                  ("Agent.predict_long", Agent.predict_long),
                  ("Router.predict", Router.predict),
                  ("Router.predict_long", Router.predict_long),
                  ("ONNXAgent.system_one", ONNXAgent.system_one)):
    check_param(label, fn, "hooks", None)
    check_param(label, fn, "on_predict_start", None)
    check_param(label, fn, "on_predict_end", None)
    check_param(label, fn, "hooks_raise", None)
    check_param(label, fn, "hooks_timeout", None)

# The scan sizes its windows from the checkpoint budget, so it takes `window`/`stride` instead
# of the per-call token overrides the single-window entries take.
check_param("Agent.predict_long", Agent.predict_long, "window", None)
check_param("Agent.predict_long", Agent.predict_long, "stride", None)
for param in ("max_len", "head_max_len"):
    check("Agent.predict_long/%s not accepted" % param, param in sig(Agent.predict_long), False)

check_param("Agent.predict_batch", Agent.predict_batch, "batch_size", None)
check_param("Agent.predict_batch", Agent.predict_batch, "sort_by_length", False)
check_param("Router.predict_batch", Router.predict_batch, "batch_size", None)
check_param("Router.predict_batch", Router.predict_batch, "sort_by_length", False)

# per-call token-budget overrides
for label, fn in (("Agent.predict_batch", Agent.predict_batch),
                  ("Agent.system_one", Agent.system_one),
                  ("Router.predict", Router.predict),
                  ("ONNXAgent.system_one", ONNXAgent.system_one)):
    check_param(label, fn, "max_len", None)
    check_param(label, fn, "head_max_len", None)

# Router.predict_batch has no call-level budget: a heterogeneous batch sets it per request, and
# the request keys `max_len` / `head_max_len` are read into each request's PredictContext.
check_param("Router.predict_batch", Router.predict_batch, "batch_size", None)
check_param("Router.predict_batch", Router.predict_batch, "hooks_timeout", None)
for param in ("max_len", "head_max_len"):
    check("Router.predict_batch/%s is per-request, not a call argument" % param,
          param in sig(Router.predict_batch), False)

# route() takes per-call hooks so a hook can pin a checkpoint for one call
check_param("Router.route", Router.route, "hooks", None)
check_param("Router.route", Router.route, "hooks_raise", None)
check_param("Router.route", Router.route, "hooks_timeout", None)

# --------------------------------------------------------------- aliases
check_true("Agent.predict is Agent.system_one", Agent.predict is Agent.system_one)
check_true("Router.system_one is Router.predict", Router.system_one is Router.predict)
check_true("ONNXAgent.predict is ONNXAgent.system_one", ONNXAgent.predict is ONNXAgent.system_one)

# --------------------------------------------------------------- context
FIELDS = ["states", "questions", "run_id", "results", "decision", "model", "agent", "router",
          "max_len", "head_max_len", "usage", "started_at", "elapsed_ms", "error"]
check("PredictContext fields", [f.name for f in dataclasses.fields(PredictContext)], FIELDS)
check("PredictContext/states required", PredictContext.__dataclass_fields__["states"].default,
      dataclasses.MISSING)
check("PredictContext/questions required", PredictContext.__dataclass_fields__["questions"].default,
      dataclasses.MISSING)
for optional in ("results", "decision", "model", "agent", "router", "usage", "elapsed_ms", "error"):
    check("PredictContext/%s default None" % optional,
          PredictContext.__dataclass_fields__[optional].default, None)
check_true("PredictContext/skip exists", callable(getattr(PredictContext, "skip", None)))
check_true("PredictContext/run_id has a factory",
           PredictContext.__dataclass_fields__["run_id"].default_factory is not dataclasses.MISSING)

# --------------------------------------------------------------- hook protocol
check("Hook lifecycle events", set(HOOK_EVENTS),
      {"on_predict_start", "on_predict_end", "on_route", "on_load", "on_evict", "on_error"})
for event in HOOK_EVENTS:
    check_true("Hook/%s declared" % event, hasattr(Hook, event))
check_true("PredictHook is callable-typed", callable(PredictHook))

# --------------------------------------------------------------- exports
for name in ("PredictContext", "PredictHook", "Hook", "BaseHook", "AsyncHook"):
    check_true("__all__/%s" % name, name in laya.__all__)
    check_true("laya.%s exists" % name, hasattr(laya, name))
check_true("laya.hooks/run_coroutine_sync exists",
           callable(getattr(__import__("laya.hooks", fromlist=["run_coroutine_sync"]),
                            "run_coroutine_sync", None)))

# BaseHook is the concrete no-op base class; all six events exist and are callable.
for event in HOOK_EVENTS:
    check_true("BaseHook/%s callable" % event, callable(getattr(BaseHook, event, None)))

# process-wide default registry lives in laya.hooks (not the top level)
for helper in ("default_hooks", "set_default_hooks", "add_default_hook", "clear_default_hooks",
               "compose_hooks"):
    check_true("laya.hooks/%s exists" % helper, callable(getattr(__import__("laya.hooks", fromlist=[helper]), helper, None)))
check_true("defaults/not exported at top level", not hasattr(laya, "set_default_hooks"))

# --------------------------------------------------------------- class defaults
for label, cls in (("Agent", Agent), ("Router", Router), ("ONNXAgent", ONNXAgent)):
    check("%s/hooks default" % label, cls.hooks, ())
    check("%s/hooks_raise default" % label, cls.hooks_raise, True)
    check("%s/hooks_concurrent default" % label, cls.hooks_concurrent, True)
    check("%s/hooks_timeout default" % label, cls.hooks_timeout, None)
    check("%s/_hooks_lock default" % label, cls._hooks_lock, None)

# Agent and ONNXAgent carry the checkpoint id for ctx.model; Router has no single model.
for label, cls in (("Agent", Agent), ("ONNXAgent", ONNXAgent)):
    check("%s/model_id default" % label, cls.model_id, None)

# runtime registration surface
for label, cls in (("Agent", Agent), ("Router", Router), ("ONNXAgent", ONNXAgent)):
    for method in ("add_hook", "remove_hook", "hooks_installed"):
        check_true("%s/%s exists" % (label, method), callable(getattr(cls, method, None)))

# The LangChain runnables batch: laya.integrations.langchain's own suite checks what
# batch() returns, so these lines pin only the caller-visible shape. A rename, or losing
# the keyword-only return_exceptions, would break LCEL and LangGraph map-reduce silently.
from laya.integrations.langchain import (  # noqa: E402
    LayaEvaluator,
    LayaGuardrail,
    LayaRouter,
    LayaTriage,
)

for label, cls in (("LayaRouter", LayaRouter), ("LayaGuardrail", LayaGuardrail),
                   ("LayaTriage", LayaTriage), ("LayaEvaluator", LayaEvaluator)):
    for method in ("invoke", "batch", "abatch"):
        check_true("langchain/%s.%s exists" % (label, method),
                   callable(getattr(cls, method, None)))
    check("langchain/%s/batch defaults" % label,
          [(p.name, p.kind.name, p.default) for p in sig(cls.batch).values()],
          [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
           ("inputs", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
           ("config", "POSITIONAL_OR_KEYWORD", None),
           ("return_exceptions", "KEYWORD_ONLY", False),
           ("kwargs", "VAR_KEYWORD", inspect.Parameter.empty)])
    check("langchain/%s/abatch defaults" % label,
          [(p.name, p.kind.name, p.default) for p in sig(cls.abatch).values()],
          [(p.name, p.kind.name, p.default) for p in sig(cls.batch).values()])

# --------------------------------------------------------------- LangChain decision node
# `LayaDecision` is the LCEL form of `laya.decide`, so its constructor has to keep accepting the
# same schema argument and runner overrides. The module imports without langchain-core installed
# (the runnable base falls back to plain object), so these checks hold in every CI lane.
from laya.integrations.langchain import LayaDecision  # noqa: E402

for param, default in (("return_details", False), ("state_key", None), ("agent", None),
                       ("base_url", None), ("api_key", None), ("model", None)):
    check_param("LayaDecision.__init__", LayaDecision.__init__, param, default)
check_param("LayaDecision.__init__", LayaDecision.__init__, "decision_schema",
            inspect.Parameter.empty, inspect.Parameter.POSITIONAL_OR_KEYWORD)
check_true("LayaDecision/invoke exists", callable(getattr(LayaDecision, "invoke", None)))
check_true("LayaDecision/__all__", "LayaDecision" in laya.__all__)
check_true("LayaDecision/laya attribute", hasattr(laya, "LayaDecision"))
check_true("LayaDecision/integrations export",
           "LayaDecision" in __import__("laya.integrations", fromlist=["__all__"]).__all__)


# --------------------------------------------------------------- LangChain token budget
# `max_len`/`head_max_len` are the per-request knobs that decide how many tokens each option of a
# choice question gets. Every runnable has to carry both, under the names the core API uses, or a
# chain has no way to widen a question that overflows the checkpoint's budget.
from laya.integrations.langchain import (  # noqa: E402
    LayaEvaluator,
    LayaGuardrail,
    LayaRouter,
    LayaTriage,
)

for label, cls in (("LayaRouter", LayaRouter), ("LayaGuardrail", LayaGuardrail),
                   ("LayaTriage", LayaTriage), ("LayaEvaluator", LayaEvaluator)):
    # Without langchain-core the runnables are plain objects, so there is no model schema to
    # carry the field; only the constructor contract applies in that lane.
    declared = getattr(cls, "model_fields", None)
    for param in ("max_len", "head_max_len"):
        check_param("%s.__init__" % label, cls.__init__, param, None)
        if declared is not None:
            check_true("%s/%s declared field" % (label, param), param in declared)


# The LangChain runnables have to carry the per-call hook arguments too, or a chain cannot
# install the cache/guard patterns from docs/hooks/patterns.md on a single node. They default to
# None (meaning "inherit the runner"), which is what keeps an unset node byte-identical to today.
# `hooks_concurrent` is deliberately absent: like the core predict surfaces, it is not per-call.
from laya.integrations.langchain import (  # noqa: E402
    LayaEvaluator,
    LayaGuardrail,
    LayaRouter,
    LayaTriage,
)

PER_CALL_HOOK_PARAMS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise",
                        "hooks_timeout")

for label, cls in (("LayaRouter", LayaRouter), ("LayaGuardrail", LayaGuardrail),
                   ("LayaTriage", LayaTriage), ("LayaEvaluator", LayaEvaluator)):
    for param in PER_CALL_HOOK_PARAMS:
        check_param("%s.__init__" % label, cls.__init__, param, None)
        declared = getattr(cls, "model_fields", None)
        if declared is not None:
            check_true("%s/%s declared field" % (label, param), param in declared)
    check_true("%s has no per-call hooks_concurrent" % label,
               "hooks_concurrent" not in inspect.signature(cls.__init__).parameters)

# --------------------------------------------------------------- usage block
# `input_tokens` / `output_tokens` are the fields every client decodes, so they are always
# present. `options` (#538) is additive and conditional: it appears only for a request whose
# options lost their distinct token spans, which is what keeps it out of ordinary responses.
from laya.common import build_sequence, collapsed_options  # noqa: E402

check_param("build_sequence", build_sequence, "return_stats", False)
check("collapsed_options/nothing collapsed is empty",
      collapsed_options(["q"], [{"options": {"options": 3, "options_distinct": 3,
                                             "tokens_per_option": None}}]), {})
check("collapsed_options/a collapsed question is reported",
      collapsed_options(["q"], [{"options": {"options": 58, "options_distinct": 42,
                                             "tokens_per_option": 4}}]),
      {"q": {"total": 58, "distinct": 42, "tokens_per_option": 4}})


# ------------------------------------------------- the cache key the examples and docs teach
# `examples/hooks/cache.py`, and the caching blocks of `docs/hooks/patterns.md`,
# `docs/hooks/examples.md` and `docs/hooks/api.md`, are four copies of one `ctx.skip()` pattern.
# The key is the part that
# can be wrong: a payload sorted by key folds two criteria orders into one cache entry, while
# `_question_schema` keeps them apart because a choice question's option order is positional. The
# second caller then gets the first caller's answer. Measured on the English checkpoint, a question
# whose criteria were only reordered came back at 0.661 confidence from the cache instead of its
# own 0.496 -- enough to open a gate at 0.6 that a fresh pass would have kept shut.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CRITERIA = {"refund": "give me money back", "cancel": "stop the service", "other": "anything else"}
Q_ORDER_A = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": dict(CRITERIA)}}
Q_ORDER_B = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": {"other": CRITERIA["other"], "refund": CRITERIA["refund"],
                                  "cancel": CRITERIA["cancel"]}}}
BATCH = ["I was charged twice for the same invoice.",
         "The app crashes every time I open the export screen.",
         "Where do I change my notification settings?"]


def taught_cache(path):
    """One taught copy of the cache pattern, exec'd out of that file's own text.

    Sliced rather than imported: the example calls `laya.load()` at module scope, which would
    download a checkpoint, and a docs code block is not importable at all. The slice stops at that
    `laya.load(...)` line, or at the closing fence of the block, for the same reason, so none of
    this needs weights.
    """
    with open(os.path.join(REPO, path), encoding="utf-8") as handle:
        text = handle.read()
    start = text.index("CACHE = {")
    stop = len(text)
    for cut in ("\nlaya.load", "\nagent = laya.load", "\n```"):
        found = text.find(cut, start)
        if found != -1:
            stop = min(stop, found)
    namespace = {"json": json, "hashlib": hashlib, "laya": laya}
    exec(text[start:stop], namespace)
    return namespace


def key_of(namespace, ctx, index=0):
    """The copy's key builder, called the way that copy declares it."""
    key = namespace.get("cache_key") or namespace["key"]
    return key(ctx, index) if len(sig(key)) > 1 else key(ctx)


def cache_ctx(questions, model="english", max_len=None, head_max_len=None, states=(BATCH[0],)):
    return PredictContext(states=list(states), questions=questions, model=model,
                          max_len=max_len, head_max_len=head_max_len)


def answers_for(states):
    """Per-state payloads, each labelled with its own state, so a mix-up is visible."""
    return [{"model": "laya-rl-agent", "answers": {"ask": {"type": "choice", "choice": tag}},
             "usage": {"input_tokens": 10 + i}} for i, tag in enumerate(
                 ["billing", "support", "other", "refund", "cancel"][:len(states)])]


TAUGHT = [("examples/hooks/cache.py", taught_cache("examples/hooks/cache.py")),
          ("docs/hooks/patterns.md", taught_cache("docs/hooks/patterns.md")),
          ("docs/hooks/examples.md", taught_cache("docs/hooks/examples.md")),
          # `api.md` is the page that documents `ctx.skip()`, so it is the page a reader copies
          # from -- and it carries the same key, which makes it the copy most able to go stale.
          ("docs/hooks/api.md", taught_cache("docs/hooks/api.md"))]

for path, namespace in TAUGHT:
    key = functools.partial(key_of, namespace)
    read = namespace.get("cache_read") or namespace["read"]
    write = namespace.get("cache_write") or namespace["write"]
    ctx = cache_ctx(Q_ORDER_A)
    check_true("%s/reordered criteria is a new entry" % path, key(ctx) != key(cache_ctx(Q_ORDER_B)))
    check_true("%s/reorders are distinct exactly when the core says so" % path,
               (key(ctx) != key(cache_ctx(Q_ORDER_B)))
               == (_question_schema(Q_ORDER_A) != _question_schema(Q_ORDER_B)))
    check_true("%s/the same call is a hit" % path, key(ctx) == key(cache_ctx(Q_ORDER_A)))
    check_true("%s/another checkpoint is a new entry" % path,
               key(ctx) != key(cache_ctx(Q_ORDER_A, model="multilingual")))
    for field in ("max_len", "head_max_len"):
        check_true("%s/another %s is a new entry" % (path, field),
                   key(ctx) != key(cache_ctx(Q_ORDER_A, **{field: 256})))

    # A hook fires once per call, and a call can carry a whole batch.
    check("%s/key takes the state it describes" % path,
          len(sig(namespace.get("cache_key") or namespace["key"])), 2)
    multi = cache_ctx(Q_ORDER_A, states=BATCH)
    check_true("%s/two states of one call are two entries" % path,
               key(multi, 0) != key(multi, 1))
    namespace["CACHE"].clear()
    filled = cache_ctx(Q_ORDER_A, states=BATCH)
    filled.results = answers_for(BATCH)
    write(filled)
    check("%s/write stores one entry per state" % path, len(namespace["CACHE"]), len(BATCH))
    warm = cache_ctx(Q_ORDER_A, states=list(reversed(BATCH)))
    read(warm)
    check("%s/a warm batch is served back per state, in the caller's order" % path,
          warm.results, list(reversed(filled.results)))
    cold = cache_ctx(Q_ORDER_A, states=BATCH[:2] + ["an uncached state"])
    read(cold)
    check_true("%s/a batch with one uncached state still runs" % path, cold.results is None)
    single = cache_ctx(Q_ORDER_A, states=[BATCH[1]])
    read(single)
    check("%s/a single-state call is one entry" % path, single.results, [filled.results[1]])
    empty = cache_ctx(Q_ORDER_A, states=[])
    try:
        read(empty)
        served = empty.results
    except Exception as exc:                      # a hook that raises on an empty call is a failure
        served = "raised %s" % exc.__class__.__name__
    check("%s/an empty batch skips to an empty list" % path, served, [])
    namespace["CACHE"].clear()

# Three copies, one key: a docs page that drifts from the example fails here, not in someone's
# production cache.
for path, namespace in TAUGHT[1:]:
    check("%s/keyed like the example" % path,
          key_of(namespace, cache_ctx(Q_ORDER_A)),
          key_of(TAUGHT[0][1], cache_ctx(Q_ORDER_A)))
    check("%s/per-state like the example" % path,
          key_of(namespace, cache_ctx(Q_ORDER_A, states=BATCH), 1),
          key_of(TAUGHT[0][1], cache_ctx(Q_ORDER_A, states=BATCH), 1))


# ------------------------------------------- what the cache example's own demo claims
# The example closes by saying its two-state batch "is served", and nothing in its printed output
# distinguishes a served payload from a fresh one -- both are the same answers. So run the demo
# against a stub agent that counts its own forward passes, and check the claim rather than trusting
# the comment. No weights: `laya.load` is the only thing the example takes from the library.
def run_cache_example():
    with open(os.path.join(REPO, "examples/hooks/cache.py"), encoding="utf-8") as handle:
        source = handle.read().replace("\nimport laya\n", "\n")
    forwards = []

    class StubAgent:
        def __init__(self, on_start, on_end):
            self.on_start, self.on_end = on_start, on_end

        def _call(self, states, questions):
            ctx = PredictContext(states=list(states), questions=questions, model="english")
            if self.on_start:
                self.on_start(ctx)
            if ctx.results is None:
                forwards.append(len(states))
                ctx.results = [{"model": "laya-rl-agent",
                                "answers": {"ask": {"type": "choice", "choice": s[:8]}}}
                               for s in states]
                ctx.usage = {"input_tokens": 10 * len(states), "output_tokens": 0}
            if self.on_end:
                self.on_end(ctx)
            return ctx.results

        def system_one(self, state, questions, model=None):
            return self._call([state], questions)[0]

        def predict_batch(self, states, questions, model=None, batch_size=None):
            return self._call(states, questions)

    class StubLaya:
        @staticmethod
        def load(name, on_predict_start=None, on_predict_end=None):
            return StubAgent(on_predict_start, on_predict_end)

    out = io.StringIO()
    namespace = {"laya": StubLaya}
    with contextlib.redirect_stdout(out):
        exec(compile(source, "examples/hooks/cache.py", "exec"), namespace)
    return namespace, forwards, out.getvalue()


example, example_forwards, example_out = run_cache_example()
# Why the exact list: the repeat of STATE and the warm batch are the two served calls, and the
# batch is the only multi-state one. A demo that mixed a cold state into the batch would show
# `[1]` here, which is the claim its comment does not make.
check("examples/hooks/cache.py/the repeat and the batch are the served calls",
      example["SKIPS"], [1, 2])
check("examples/hooks/cache.py/one forward per new entry, none for the batch",
      example_forwards, [1, 1, 1])
check_true("examples/hooks/cache.py/its own output still says two entries",
           "cache entries: 2" in example_out, example_out)
check("examples/hooks/cache.py/the batch answers in the caller's order",
      [r["answers"]["ask"]["choice"] for r in example["served"]],
      [example["OTHER"][:8], example["STATE"][:8]])


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all hook API tests passed")
sys.exit(1 if FAIL else 0)
