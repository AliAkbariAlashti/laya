"""Per-call decision controls shared by the framework integrations.

Every integration wrapper ends at one of two places: a local `runner.predict(state, questions,
**overrides)` call, or a POST to a laya-serve node. Which overrides those two accept is a single
rule, and three independent copies of it drift -- this module is the copy the LangChain, CrewAI
and LlamaIndex wrappers import.

The names are listed once, in `PREDICT_CONTROLS` and `HOOK_CONTROLS`, and the signatures below
are checked against them by `tests/test_crewai.py` and `tests/test_llamaindex.py`, which also
assert every wrapper's constructor accepts all of them. A control added here without reaching a
wrapper's `__init__` fails that wrapper's suite rather than being dropped silently.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# The token-budget overrides `Agent.predict` / `Router.predict` take per call, and that
# laya-serve accepts in the request body up to its `LAYA_MAX_TOKEN_BUDGET` ceiling.
PREDICT_CONTROLS = ("max_len", "head_max_len")

# The per-call hook family. These are Python callables and flags that run inside `predict`, so
# they exist only on the local path -- see `reject_remote_hooks`.
HOOK_CONTROLS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")


def budget_kwargs(max_len: Optional[int] = None,
                  head_max_len: Optional[int] = None) -> Dict[str, Any]:
    """The two token budgets, with the unset ones omitted.

    Absent rather than `None`: on the local path a `None` budget would override the checkpoint's
    own default with nothing, and on the remote path an older stand-in `_call_remote` -- including
    the ones these integrations' own tests install -- must keep accepting the call.
    """
    return {key: value for key, value in (("max_len", max_len), ("head_max_len", head_max_len))
            if value is not None}


def predict_kwargs(model: Optional[str] = None, max_len: Optional[int] = None,
                   head_max_len: Optional[int] = None) -> Dict[str, Any]:
    """The per-request overrides a local runner accepts, with the unset ones omitted."""
    kwargs: Dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    kwargs.update(budget_kwargs(max_len, head_max_len))
    return kwargs


def hook_kwargs(hooks: Optional[Any] = None, on_predict_start: Optional[Any] = None,
                on_predict_end: Optional[Any] = None, hooks_raise: Optional[bool] = None,
                hooks_timeout: Optional[float] = None) -> Dict[str, Any]:
    """The per-call hook overrides, with the unset ones omitted.

    Core reads `None` as "inherit whatever the runner was built with", so an unset hook has to be
    absent rather than passed as `None`. Note the `is not None` tests: `hooks=[]` means "no hooks
    for this call", and `hooks_raise=False` means "keep deciding after a hook fails" -- both are
    decisions a caller made, not absences.
    """
    given = {"hooks": hooks, "on_predict_start": on_predict_start, "on_predict_end": on_predict_end,
             "hooks_raise": hooks_raise, "hooks_timeout": hooks_timeout}
    return {k: v for k, v in given.items() if v is not None}


def reject_remote_hooks(given: Dict[str, Any], base_url: Optional[str]) -> None:
    """Refuse hooks on a remote node rather than dropping them silently.

    A hook is a Python callable that runs inside `predict` -- it can cache a decision, gate one or
    rewrite its state. `laya-serve` has no way to receive or run one, so a node with a `base_url`
    and hooks configured would report success while never calling them.
    """
    if base_url and given:
        raise ValueError(
            "%s run in the local runner and cannot be sent to a laya-serve endpoint; "
            "install them where serve runs, or drop them" % ", ".join(sorted(given))
        )
