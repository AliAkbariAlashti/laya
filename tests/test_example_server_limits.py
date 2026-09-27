"""Regression: examples/server.py must bound a request the way laya.serve does.

`laya/serve.py` refuses a request carrying more than MAX_QUESTIONS questions, a state
over MAX_STATE_CHARS, or a body over its own cap, because Laya encodes the state once
per question -- cost is questions x state size, collated into one tensor.

examples/server.py bounded `states` to 64 and left the rest open: 20 000 questions and
a 5 MB state were both accepted where the shipped server answers 413. The bounds are
read from laya.serve rather than restated, so the two cannot drift.

Scope: the question count and the state size, answered 413 as laya.serve answers them.
`Question.instructions` and `criteria` still carry unbounded text that no per-field
bound can see; capping the request body is the backstop for those and is left out
deliberately -- see the PR description.

Also counts the Router calls `/predict/batch` makes: the endpoint is batch-shaped (one
`questions` map, up to 64 states), so it owes one `Router.predict_batch` per request, not
one `Router.predict` per state. Those checks inject a recording stand-in at `demo.ROUTER`,
which still loads no weights.

Driven over HTTP through TestClient. No weights are loaded; the router stays unbuilt, so
a request that passes validation answers 503, which is the assertion for "accepted".

Run: python tests/test_example_server_limits.py
"""
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LAYA_PRELOAD", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, ("  -- " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail and not cond else ""), flush=True)


def main():
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        # Only the optional serving stack may be missing. An ImportError naming anything
        # else -- in particular `cannot import name MAX_QUESTIONS from laya.serve`, the
        # drift this test exists to catch -- must fail rather than skip.
        if (getattr(exc, "name", None) or "").split(".")[0] not in ("fastapi", "httpx", "starlette"):
            raise
        print("SKIP: fastapi/httpx not installed -- pip install laya[serve] httpx")
        return 0
    try:
        import server as demo
    except ImportError as exc:
        if (getattr(exc, "name", None) or "").split(".")[0] not in ("fastapi", "httpx", "starlette", "multipart"):
            raise
        print("SKIP: examples/server.py needs the serve extra -- pip install laya[serve]")
        return 0

    from laya.serve import MAX_QUESTIONS, MAX_STATE_CHARS

    client = TestClient(demo.app, raise_server_exceptions=False)
    one = {"a": {"type": "noul", "instructions": "x"}}

    def questions(n):
        return {"q%d" % i: {"type": "noul", "instructions": "x"} for i in range(n)}

    def code(**kw):
        return client.post("/predict", **kw).status_code

    # The bounds must come from laya.serve, not a local copy, or the two drift.
    demo_q = getattr(demo, "MAX_QUESTIONS", None)
    demo_s = getattr(demo, "MAX_STATE_CHARS", None)
    ok("the per-request bounds are laya.serve's",
       demo_q == MAX_QUESTIONS and demo_s == MAX_STATE_CHARS,
       "demo %r/%r vs laya.serve %r/%r" % (demo_q, demo_s, MAX_QUESTIONS, MAX_STATE_CHARS))

    # --- too many questions, too large a state: 413, as laya.serve answers ---
    ok("more than MAX_QUESTIONS questions is 413",
       code(json={"state": "hi", "questions": questions(MAX_QUESTIONS + 1)}) == 413)
    # A noul question carries no answer options, so an options-based budget cannot see
    # it; the count is what has to be bounded.
    ok("a noul-only flood is 413 (it carries no answer options)",
       code(json={"state": "hi", "questions": questions(20_000)}) == 413)
    ok("a state over MAX_STATE_CHARS is 413",
       code(json={"state": "A" * (MAX_STATE_CHARS + 1), "questions": one}) == 413)
    ok("an oversized dict state is 413",
       code(json={"state": {"body": "A" * (MAX_STATE_CHARS + 1)}, "questions": one}) == 413)
    ok("an oversized state inside a batch is 413",
       client.post("/predict/batch",
                   json={"states": ["hi", "A" * (MAX_STATE_CHARS + 1)], "questions": one}
                   ).status_code == 413)


    # --- the refusal must not echo the rejected payload back ----------------
    # Declaring these as Field(max_length=...) instead would report 422 *and* include
    # the offending `input` in FastAPI's validation-error body, so refusing a 5 MB
    # state would write 5 MB back to the caller -- a size limit that amplifies.
    big = {"state": "A" * 5_000_000, "questions": one}
    sent = len(json.dumps(big).encode())
    resp = client.post("/predict", json=big)
    ok("a rejected 5 MB state answers 413 without echoing it",
       resp.status_code == 413 and len(resp.content) < 1_000,
       "%s, %d bytes returned for %d sent" % (resp.status_code, len(resp.content), sent))

    # --- a genuine schema error is still 422, not 413 -----------------------
    ok("an unknown question type is 422",
       code(json={"state": "hi", "questions": {"a": {"type": "bogus", "instructions": "x"}}}) == 422)
    ok("a choice question with no criteria is 422",
       code(json={"state": "hi", "questions": {"a": {"type": "choice", "instructions": "x"}}}) == 422)
    ok("an empty state is 422", code(json={"state": "", "questions": one}) == 422)
    ok("zero questions is 422", code(json={"state": "hi", "questions": {}}) == 422)

    # --- the limits are limits, not walls -----------------------------------
    # No router is built, so anything that passes validation answers 503.
    ok("an ordinary request passes validation",
       code(json={"state": {"body": "billed twice, please refund"}, "questions": one}) == 503)
    ok("exactly MAX_QUESTIONS passes",
       code(json={"state": "hi", "questions": questions(MAX_QUESTIONS)}) == 503)
    ok("a state of exactly MAX_STATE_CHARS passes",
       code(json={"state": "A" * MAX_STATE_CHARS, "questions": one}) == 503)
    ok("a list state passes", code(json={"state": ["a", "b"], "questions": one}) == 503)
    ok("a small chunked body passes",
       code(content=(lambda: (yield json.dumps({"state": "hi", "questions": one}).encode()))(),
            headers={"content-type": "application/json"}) == 503)
    # /predict/batch reports per-item failures inside a 200 envelope, so an accepted
    # batch is a 200 here; over the state bound it is refused outright, and 413 rather
    # than 422 because "too many states" is a size violation like the others.
    ok("a 64-state batch is accepted",
       client.post("/predict/batch", json={"states": ["hi"] * 64, "questions": one}).status_code == 200)
    ok("a 65-state batch is still refused by the existing bound",
       client.post("/predict/batch", json={"states": ["hi"] * 65, "questions": one}).status_code == 422)
    ok("the page and health endpoints are unaffected",
       client.get("/").status_code == 200 and client.get("/health").status_code == 200)

    # --- /predict/batch must make ONE Router call, not one per state ---------
    #
    # README teaches `Router.predict_batch` as the way to answer many states with shared forward
    # passes ("routes the full workload first, groups requests by checkpoint ... results are
    # restored to the original request order"), and this endpoint is the batch-shaped surface the
    # README points at for trying Laya without writing code. The handler used to call
    # `Router.predict` once per state anyway, so 64 states were 64 forwards. These drive the real
    # app -- validation, `_questions()`, the handler body, the JSON envelope -- over a recording
    # stand-in at `demo.ROUTER`, so no weights are needed to count the calls.

    import inspect

    from laya.router import Router as CoreRouter

    class RecordingRouter:
        """Answers like the Router does, and remembers how it was asked."""

        def __init__(self, fail_on=None):
            self.predict_calls = []
            self.batch_calls = []
            self.fail_on = fail_on

        def predict(self, state, questions, **kw):
            self.predict_calls.append((state, dict(kw)))
            return self._answer(state, questions)

        def predict_batch(self, requests, **kw):
            self.batch_calls.append((list(requests), dict(kw)))
            return [self._answer(r["state"], r["questions"]) for r in requests]

        def _answer(self, state, questions):
            if self.fail_on and self.fail_on in str(state):
                raise ValueError("simulated failure for " + self.fail_on)
            return {"answers": {"a": {"type": "noul", "choice": False}}, "state": state,
                    "questions": questions}

    states = ["ticket %d" % i for i in range(8)]

    class LegacyRouter(RecordingRouter):
        """A Router-like object whose `predict_batch` predates the batch path entirely."""

        predict_batch = None

    # A Router that predates `predict_batch` must still be usable through the fallback.
    older = LegacyRouter()
    demo.ROUTER = older
    legacy = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one})
    ok("a router without predict_batch still answers every state",
       legacy.status_code == 200 and len(legacy.json()["results"]) == 8
       and not [r for r in legacy.json()["results"] if "error" in r],
       "%s / %s" % (legacy.status_code, json.dumps(legacy.json())[:200]))

    router = RecordingRouter()
    demo.ROUTER = router
    batched = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one})
    body = batched.json()
    ok("an 8-state batch is ONE predict_batch call and zero predict calls",
       len(router.batch_calls) == 1 and not router.predict_calls,
       "predict_batch=%d predict=%d" % (len(router.batch_calls), len(router.predict_calls)))
    ok("the batch envelope is unchanged: count and one result per state, in order",
       body.get("count") == 8 and [r.get("state") for r in body.get("results", [])] == states)
    # Read defensively: an endpoint that never batches has nothing to inspect, and the checks below
    # say so by name instead of letting this script die at the unpack.
    requests_sent, call_kwargs = (router.batch_calls[0] if router.batch_calls else ([], {}))
    ok("each request carries state + questions, and the questions map is the same one",
       len(requests_sent) == 8
       and all(sorted(r) == ["questions", "state"] for r in requests_sent)
       and all(r["questions"] == requests_sent[0]["questions"] for r in requests_sent))
    controls_sent = [k for r in requests_sent for k in r if k in ("model", "task", "lang")]
    ok("unset controls are absent, not sent as null",
       len(requests_sent) == 8 and not call_kwargs and not controls_sent,
       "call kwargs=%r controls=%r" % (call_kwargs, controls_sent))
    # Every key the endpoint puts in a request must be one core actually reads out of it, and the
    # set of those keys is derived from `Router.route_batch`'s own source, so a rename or a new
    # override in core shows up here rather than silently stopping reaching the router.
    import re

    route_src = inspect.getsource(CoreRouter.route_batch)
    read = ({"state", "questions"}
            | set(re.findall(r'request\["(\w+)"\]', route_src))
            | set(re.findall(r'request\.get\("(\w+)"', route_src)))
    sent_keys = {k for r in requests_sent for k in r}
    ok("the request keys the endpoint sends are ones core reads",
       len(requests_sent) == 8 and read >= sent_keys,
       "core reads %r, endpoint sends %r" % (sorted(read), sorted(sent_keys)))
    ok("predict_batch is still a one-positional-list call",
       list(inspect.signature(CoreRouter.predict_batch).parameters)[1] == "requests")

    pinned = RecordingRouter()
    demo.ROUTER = pinned
    TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one,
                                "model": "multilingual", "lang": "de"})
    sent = pinned.batch_calls[0][0] if pinned.batch_calls else []
    ok("a pinned model/lang travels with every request",
       len(sent) == 8
       and all(r["model"] == "multilingual" and r["lang"] == "de" for r in sent)
       and not any("task" in r for r in sent),
       json.dumps(sent[:1])[:200])

    # One state failing must not cost its neighbours their answer: the endpoint's published
    # contract is per-item errors inside a 200.
    partial = RecordingRouter(fail_on="ticket 3")
    demo.ROUTER = partial
    poison = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one}).json()
    errors = [r for r in poison["results"] if "error" in r]
    ok("one failing state yields one error entry, not an empty batch",
       len(poison["results"]) == 8 and [r["index"] for r in errors] == [3]
       and "simulated failure" in errors[0]["error"],
       json.dumps(poison)[:240])
    ok("the failing batch retried per state, so its neighbours still answered",
       len(partial.batch_calls) == 1 and len(partial.predict_calls) == 8)

    # The single-state surface must keep going through predict(), and `/predict/batch` with one
    # state must still batch -- otherwise the two endpoints diverge on where hooks fire.
    single = RecordingRouter()
    demo.ROUTER = single
    one_state = TestClient(demo.app, raise_server_exceptions=False)
    one_state.post("/predict", json={"state": "ticket 0", "questions": one})
    ok("/predict still calls Router.predict once",
       len(single.predict_calls) == 1 and not single.batch_calls)

    demo.ROUTER = None
    not_ready = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one})
    ok("an unready router keeps answering 200 with per-item 503s",
       not_ready.status_code == 200
       and len(not_ready.json()["results"]) == 8
       and all("503" in r.get("error", "") for r in not_ready.json()["results"]),
       json.dumps(not_ready.json())[:200])
    demo.ROUTER = None

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
