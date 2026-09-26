"""Metric math, dataset parsing and regression comparison for laya.evals. No weights.

Run: python -m pytest tests/test_evals.py -q
"""
import json

import pytest

from laya.evals import (
    ChoiceAccuracy,
    Dataset,
    EvalError,
    EvalReport,
    Example,
    MeanConfidence,
    NoulAccuracy,
    ScoreMAE,
    ScoreWithin,
    assert_regression,
    default_evaluators,
    ece,
    evaluate,
)

Q = {"intent": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}}
QSCORE = {"quality": {"type": "score", "instructions": "?", "criteria": ["low", "high"]}}
QNOUL = {"flag": {"type": "noul", "instructions": "?"}}


def choice_answer(label, confidence=0.9):
    return {"type": "choice", "choice": label, "probabilities": {label: confidence},
            "confidence": confidence}


def noul_answer(prob):
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1 - prob)}


class StubRunner:
    """Returns fixed answers per state, so evaluation is deterministic and weight-free."""

    def __init__(self, by_state):
        self.by_state = by_state

    def predict(self, state, questions, model=None):
        return {"model": model or "stub", "answers": self.by_state[state]}


# --------------------------------------------------------------- evaluator math
def test_choice_and_confidence_math():
    evaluator = ChoiceAccuracy()
    assert evaluator.score(choice_answer("a"), "a") == 1.0
    assert evaluator.score(choice_answer("b"), "a") == 0.0
    assert evaluator.score(noul_answer(0.9), "a") is None, "wrong answer type does not apply"
    assert MeanConfidence().score(choice_answer("a", 0.8), "a") == 0.8


def test_noul_and_score_math():
    assert NoulAccuracy().score(noul_answer(0.9), True) == 1.0
    assert NoulAccuracy().score(noul_answer(0.2), True) == 0.0
    assert ScoreMAE().score({"type": "score", "score": 0.4}, 0.7) == pytest.approx(0.3)
    within = ScoreWithin(0.5)
    assert within.score({"type": "score", "score": 0.4}, 0.7) == 1.0
    assert within.name == "score_within_0.5"


def test_calibration_uses_answer_confidence():
    answer = {"type": "choice", "choice": "a", "confidence": 0.2, "answer_confidence": 0.9}
    assert MeanConfidence().score(answer, "a") == pytest.approx(0.9), "calibrated, not entropy"
    report = evaluate(StubRunner({"s": {"intent": answer}}),
                      Dataset([Example("s", Q, {"intent": "a"})]))
    assert report.overall["mean_confidence"] == pytest.approx(0.9)


def test_compare_ignores_latency_by_default():
    report = EvalReport(overall={"choice_accuracy": 0.8, "latency_p50_ms": 12.0})
    baseline = {"overall": {"choice_accuracy": 0.8, "latency_p50_ms": 5.0}}
    ok, deltas = report.compare(baseline)
    assert ok and "latency_p50_ms" not in deltas, "timing noise is not a quality regression"
    bad, deltas = report.compare(baseline, {"latency_p50_ms": 1.0})
    assert not bad and "latency_p50_ms" in deltas


def test_ece_on_known_inputs():
    assert ece([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    assert ece([0.0, 0.0], [False, False]) == pytest.approx(0.0)
    assert ece([], []) is None


# --------------------------------------------------------------- dataset
def test_dataset_from_jsonl(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join([
        json.dumps({"state": "s1", "questions": Q, "expected": {"intent": "a"}, "language": "en"}),
        "# a comment line",
        json.dumps({"state": "s2", "questions": Q, "expected": {"intent": "b"}, "tags": ["t"]}),
    ]), encoding="utf-8")
    dataset = Dataset.from_jsonl(str(path))
    assert len(dataset) == 2
    assert dataset.examples[0].language == "en"
    assert dataset.examples[1].tags == ("t",)


@pytest.mark.parametrize("row, fragment", [
    ({"state": "s", "questions": Q}, "missing 'expected'"),
    ({"state": "s", "questions": Q, "expected": {"nope": "a"}}, "unknown question"),
    ({"state": "s", "questions": [], "expected": {}}, "'questions' must be an object"),
])
def test_dataset_rejects_bad_rows(row, fragment):
    with pytest.raises(EvalError) as exc:
        Example.from_dict(row)
    assert fragment in str(exc.value)


def test_dataset_rejects_malformed_json_and_empty(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(EvalError):
        Dataset.from_jsonl(str(bad))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(EvalError):
        Dataset.from_jsonl(str(empty))


# --------------------------------------------------------------- evaluate
def test_evaluate_overall_and_slices():
    dataset = Dataset([
        Example("s1", Q, {"intent": "a"}, language="en"),
        Example("s2", Q, {"intent": "b"}, language="en"),
        Example("s3", Q, {"intent": "a"}, language="de"),
    ])
    runner = StubRunner({
        "s1": {"intent": {"type": "choice", "choice": "a", "confidence": 1.0}},
        "s2": {"intent": {"type": "choice", "choice": "b", "confidence": 1.0}},
        "s3": {"intent": {"type": "choice", "choice": "b", "confidence": 1.0}},
    })
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()])
    assert report.overall["choice_accuracy"] == pytest.approx(2 / 3)
    assert report.slices["language"]["en"]["choice_accuracy"] == pytest.approx(1.0)
    assert report.slices["language"]["de"]["choice_accuracy"] == pytest.approx(0.0)
    assert report.slices["qid"]["intent"]["choice_accuracy"] == pytest.approx(2 / 3)
    assert "choice_accuracy" in report.to_markdown()


def test_evaluate_batches_same_questions():
    class BatchRunner(StubRunner):
        def __init__(self, by_state):
            super().__init__(by_state)
            self.batches = []

        def predict_batch(self, states, questions, model=None, batch_size=None):
            self.batches.append(list(states))
            return [{"model": "m", "answers": self.by_state[s]} for s in states]

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = BatchRunner({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert runner.batches == [["s1", "s2"]], "identical questions share one forward pass"
    assert report.overall["choice_accuracy"] == 1.0


class RequestsRunner(StubRunner):
    """The `Router` shape: a list of per-request dicts, and no ``model=`` on the call."""

    def __init__(self, by_state):
        super().__init__(by_state)
        self.batches = []
        self.batch_sizes = []

    def predict_batch(self, requests, batch_size=None):
        self.batches.append(list(requests))
        self.batch_sizes.append(batch_size)
        return [{"model": r.get("model") or "m", "answers": self.by_state[r["state"]]}
                for r in requests]


def _labels(report):
    """The decided labels with their correctness: parity at decision level, not as floats."""
    return [(c["answer"]["choice"], c["correct"]) for c in report.cases]


def test_evaluate_drives_a_requests_shaped_batch():
    dataset = Dataset([Example("s1", Q, {"intent": "a"}, model="english"),
                       Example("s2", Q, {"intent": "a"}, model="english")])
    runner = RequestsRunner({"s1": {"intent": choice_answer("a")},
                             "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert len(runner.batches) == 1, "a request-dict batch is a forward pass the harness can run"
    expected = [{"state": "s1", "questions": Q, "model": "english"},
                {"state": "s2", "questions": Q, "model": "english"}]
    assert runner.batches[0] == expected, "each request carries its own state, questions, checkpoint"
    assert runner.batch_sizes == [8], "the requested batch size reaches the runner"
    assert report.overall["choice_accuracy"] == 1.0


def test_requests_shaped_batch_agrees_with_single_predicts():
    """`batch_size` changes how many forward passes a run makes, never what it scores."""
    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"}),
                       Example("s3", Q, {"intent": "b"})])
    answers = {"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("b")},
               "s3": {"intent": choice_answer("b")}}
    batched = evaluate(RequestsRunner(answers), dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    single = evaluate(RequestsRunner(answers), dataset, evaluators=[ChoiceAccuracy()])
    assert _labels(batched) == _labels(single) == [("a", True), ("b", False), ("b", True)]
    assert len(batched.cases) == len(single.cases) == 3


def test_evaluate_scores_a_batch_shape_it_cannot_call():
    """A `predict_batch` in neither documented shape must not fail the run row by row."""
    class UntypedBatch(StubRunner):
        def predict_batch(self, states, questions):
            raise AssertionError("the harness may not call this: no model=, no requests")

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = UntypedBatch({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8, on_error="skip")
    assert report.overall["choice_accuracy"] == 1.0, "scored one predict at a time"
    assert not report.config.get("errored"), "a batch entry point in an unknown shape is not an error"


def test_evaluate_scores_a_runner_with_no_batch_entry_point():
    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = StubRunner({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert report.overall["choice_accuracy"] == 1.0


def test_evaluate_batches_a_pass_through_wrapper_positionally():
    """A wrapper that forwards `*args, **kwargs` takes the positional call, whatever it names."""
    class PassThrough(StubRunner):
        def __init__(self, by_state):
            super().__init__(by_state)
            self.calls = []

        def predict_batch(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return [{"model": "m", "answers": self.by_state[s]} for s in args[0]]

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = PassThrough({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == (["s1", "s2"], Q)
    assert runner.calls[0][1] == {"model": None, "batch_size": 8}
    assert report.overall["choice_accuracy"] == 1.0


# --------------------------------------------------------------- timing (#585)
FORWARD_MS = 100.0
SHARED = 0.6                      # a batch of n costs SHARED * n * FORWARD_MS, as one call
QWIDE = {"intent": {"type": "choice", "instructions": "?",
                    "criteria": {"a": "x", "b": "y", "c": "z"}}}


class Clock:
    """A timer that advances only when the runner predicts, so every figure below is exact."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now / 1000.0


class TimedRunner(StubRunner):
    """Answers from the state alone, so grouping can never change a decision -- only the clock."""

    def __init__(self, by_state):
        super().__init__(by_state)
        self.clock = Clock()
        self.chunks = []
        self.singles = []

    def predict(self, state, questions, model=None):
        self.clock.now += FORWARD_MS
        self.singles.append(state)
        return StubRunner.predict(self, state, questions, model)

    def predict_batch(self, states, questions, model=None, batch_size=None):
        self.clock.now += len(states) * FORWARD_MS * SHARED
        self.chunks.append(len(states))
        return [StubRunner.predict(self, s, questions, model) for s in states]


def _timed_pair(monkeypatch, batch_size):
    """Score three shareable rows and one that cannot join them, on a fake clock.

    Returns (report, runner): the runner's own `chunks` is the witness that the grouping the test
    asserts is the grouping the harness really issued.
    """
    import laya.evals as evals_module

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"}),
                       Example("s3", Q, {"intent": "a"}), Example("s4", QWIDE, {"intent": "a"})])
    answers = {state: {"intent": choice_answer("a")} for state in ("s1", "s2", "s3", "s4")}
    runner = TimedRunner(answers)
    monkeypatch.setattr(evals_module, "time", runner.clock)
    return evals_module.evaluate(runner, dataset, evaluators=[ChoiceAccuracy()],
                                 batch_size=batch_size), runner


def test_batched_latency_is_what_a_request_waited(monkeypatch):
    solo, solo_runner = _timed_pair(monkeypatch, None)
    batched, runner = _timed_pair(monkeypatch, 8)

    assert _labels(batched) == _labels(solo), "identical decisions; only the timing moved"
    assert (solo_runner.chunks, solo_runner.singles) == ([], ["s1", "s2", "s3", "s4"])
    assert (runner.chunks, runner.singles) == ([3], ["s4"]), \
        "one shared call of three, and the row whose questions match nothing left alone"

    assert solo.overall["latency_p50_ms"] == pytest.approx(FORWARD_MS)
    # The chunk of three returns all three requests together at +180 ms, so that is their latency.
    assert batched.overall["latency_p50_ms"] == pytest.approx(3 * FORWARD_MS * SHARED)
    assert batched.overall["latency_p50_ms"] > solo.overall["latency_p50_ms"], \
        "batching trades request latency for throughput; the report has to say so"


def test_the_throughput_share_keeps_its_own_metric(monkeypatch):
    solo, _ = _timed_pair(monkeypatch, None)
    batched, _ = _timed_pair(monkeypatch, 8)

    # Unbatched, the two quantities are the same number, so every report without the flag is
    # unchanged by this fix.
    assert solo.overall["cost_per_decision_p50_ms"] == pytest.approx(solo.overall["latency_p50_ms"])
    assert solo.overall["cost_per_decision_p95_ms"] == pytest.approx(solo.overall["latency_p95_ms"])
    # Batched, the share is the figure the old `latency_p50_ms` published: 180 ms over three rows.
    assert batched.overall["cost_per_decision_p50_ms"] == pytest.approx(FORWARD_MS * SHARED)
    assert batched.overall["cost_per_decision_p95_ms"] == pytest.approx(FORWARD_MS)


def test_a_latency_gate_cannot_pass_a_run_where_nothing_finished_in_time(monkeypatch):
    from laya import evals_cli

    solo, _ = _timed_pair(monkeypatch, None)
    batched, _ = _timed_pair(monkeypatch, 8)
    # 80 ms is below every wait in either run (100 ms alone, 180 ms shared). The old report passed
    # the batched run at 60 ms, which was 1/3 of a call no request could see the end of.
    for name, report in (("unbatched", solo), ("--batch-size 8", batched)):
        assert evals_cli._check_thresholds(report.overall, {}, {"latency_p50_ms": 80.0}), \
            "%s abstains: no request in it was served inside the limit" % name
    for name, report in (("unbatched", solo), ("--batch-size 8", batched)):
        assert not evals_cli._check_thresholds(report.overall, {}, {"latency_p50_ms": 200.0}), \
            "%s passes a bound every request beat" % name
    # The throughput win is still gateable, under the name that measures it.
    assert not evals_cli._check_thresholds(batched.overall, {}, {"cost_per_decision_p50_ms": 80.0})
    assert evals_cli._check_thresholds(solo.overall, {}, {"cost_per_decision_p50_ms": 80.0})


def test_timing_facts_record_what_the_harness_did(monkeypatch):
    solo, _ = _timed_pair(monkeypatch, None)
    batched, _ = _timed_pair(monkeypatch, 8)

    assert solo.config["timing"]["batch_size"] is None
    assert solo.config["timing"]["batch_form"] is None
    assert solo.config["timing"]["rows_grouped"] == 0
    assert solo.config["timing"]["rows_alone"] == 4
    assert solo.config["timing"]["max_chunk"] == 1
    assert batched.config["timing"]["batch_size"] == 8
    assert batched.config["timing"]["batch_form"] == "states"
    assert batched.config["timing"]["chunks"] == 2
    assert batched.config["timing"]["rows_grouped"] == 3
    assert batched.config["timing"]["rows_alone"] == 1
    assert batched.config["timing"]["max_chunk"] == 3
    assert batched.config["timing"]["latency_metric"] != batched.config["timing"]["cost_metric"]

    # `--json` is the artifact a reviewer reads, so the two runs must differ there, not only in
    # the flag they were asked with.
    assert json.dumps(batched.to_json()) != json.dumps(solo.to_json())


def test_a_requested_batch_size_is_not_a_batched_run(monkeypatch):
    """The report records the grouping it achieved, so a runner that cannot batch cannot hide it."""
    import laya.evals as evals_module

    class Untimed(StubRunner):
        pass

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    monkeypatch.setattr(evals_module, "time", Clock())
    report = evals_module.evaluate(Untimed({"s1": {"intent": choice_answer("a")},
                                            "s2": {"intent": choice_answer("a")}}),
                                   dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert report.config["timing"]["batch_form"] is None
    assert report.config["timing"]["rows_grouped"] == 0, "asked to batch, and the report says it did not"
    assert report.config["timing"]["rows_alone"] == 2


def test_compare_leaves_the_cost_metrics_alone():
    report = EvalReport(overall={"choice_accuracy": 0.8, "latency_p50_ms": 12.0,
                                 "cost_per_decision_p50_ms": 4.0})
    baseline = {"overall": {"choice_accuracy": 0.8, "latency_p50_ms": 5.0,
                            "cost_per_decision_p50_ms": 2.0}}
    ok, deltas = report.compare(baseline)
    assert ok and not {"latency_p50_ms", "cost_per_decision_p50_ms"} & set(deltas)
    bad, deltas = report.compare(baseline, {"cost_per_decision_p50_ms": 0.5})
    assert not bad, "a 2 ms drift is outside the 0.5 ms tolerance it named"
    assert "latency_p50_ms" not in deltas, "the metric nobody named stays out of the comparison"
    assert deltas["cost_per_decision_p50_ms"]["diff"] == pytest.approx(2.0)


def test_docs_and_the_harness_name_the_same_timing_metrics(monkeypatch):
    """`docs/evals.md` must list exactly the metrics `evaluate` publishes, with their quantities."""
    import pathlib
    import re

    page = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "evals.md").read_text()
    batched, _ = _timed_pair(monkeypatch, 8)
    published = {metric for metric in batched.overall if metric.endswith("_ms")}

    rows = {}
    for line in page.splitlines():
        if line.startswith("| `"):
            for metric in re.findall(r"`([a-z_0-9]+_ms)`", line):
                rows[metric] = line
    assert set(rows) == published, "no metric documented that is not emitted, and none missed"
    assert "per request" in rows["latency_p50_ms"]
    assert "waited" in rows["latency_p50_ms"], "the row has to say a request waits the whole call"
    assert "divided by the rows it carried" in rows["cost_per_decision_p50_ms"]
    assert "## Batching and timing" in page, "the trade-off the two numbers encode is written down"
    assert "`config.timing`" in page, "the report's own run facts are documented"


def test_evaluate_skips_errors_when_asked():
    class Boom(StubRunner):
        def predict(self, state, questions, model=None):
            if state == "bad":
                raise RuntimeError("no model")
            return super().predict(state, questions, model)

    dataset = Dataset([Example("ok", Q, {"intent": "a"}), Example("bad", Q, {"intent": "a"})])
    runner = Boom({"ok": {"intent": choice_answer("a")}})
    with pytest.raises(RuntimeError):
        evaluate(runner, dataset, on_error="fail")
    report = evaluate(runner, dataset, on_error="skip")
    assert len(report.cases) == 1
    assert report.config["errored"][0]["error"].startswith("RuntimeError")


# --------------------------------------------------------------- compare
def test_compare_and_assert_regression():
    report = EvalReport(overall={"choice_accuracy": 0.8, "ece": 0.10})
    baseline = {"overall": {"choice_accuracy": 0.79, "ece": 0.08}}
    ok, deltas = report.compare(baseline, {"choice_accuracy": 0.02, "ece": 0.03})
    assert ok and deltas["choice_accuracy"]["diff"] == pytest.approx(0.01)
    bad, bad_deltas = report.compare(baseline, {"ece": 0.01})
    assert not bad and bad_deltas["ece"]["diff"] == pytest.approx(0.02)
    with pytest.raises(AssertionError):
        assert_regression(report, baseline, {"ece": 0.01})


def test_default_evaluators_cover_the_three_types():
    names = {e.name for e in default_evaluators()}
    assert {"choice_accuracy", "noul_accuracy", "score_mae", "mean_confidence"} <= names


# --------------------------------------------------------------- CLI
def _write_dataset(tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_cli_validate_and_dispatch(tmp_path):
    from laya import cli, evals_cli

    good = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["validate", good]) == 0
    assert cli.main(["eval", "validate", good]) == 0, "`laya eval` dispatches to laya-evals"

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    assert evals_cli.main(["validate", str(bad)]) == 1


def test_cli_compare_exit_codes(tmp_path):
    from laya import evals_cli

    (tmp_path / "report.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.80}}))
    (tmp_path / "baseline.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.79}}))
    report, baseline = str(tmp_path / "report.json"), str(tmp_path / "baseline.json")
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.02"]) == 0
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.001"]) == 1


def test_cli_rejects_a_malformed_tolerance():
    from laya import evals_cli

    with pytest.raises(EvalError):
        evals_cli._parse_pairs(["choice_accuracy"])
