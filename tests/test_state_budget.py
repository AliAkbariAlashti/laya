"""Regression: the state budget is sized from the head actually built, not from `head_max_len`.

`head_max_len` is a cap. `build_sequence` fills the option prompt budget only as far as the
question and its option descriptions need, then sizes the state as `max_len - head_len - 1`
from the head it actually built. The README used to describe the remaining state budget as
`max_len - head_max_len`, which is a lower bound rather than the real figure: with the shipped
question it understates the room by 143 tokens on `laya` (463 vs 320) and 210 on
`laya-multilingual` (978 vs 768).

This suite pins the relationship, so a future change to either the renderer or the README
cannot silently desynchronise them. The last two checks are the ones that would have caught
the documentation defect; they use the real checkpoints when present and skip otherwise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import build_sequence  # noqa: E402

PASS, FAIL = [], []

# The same 4-option question the README's token-budget section walks through.
Q = {"t": "choice", "ins": "Which department should handle this request?",
     "crit": {"billing": "invoices, payments, refunds",
              "technical": "bugs, outages, system errors",
              "sales": "pricing, new contracts, plan upgrades",
              "other": "everything else"}}


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


class _FakeTok:
    """Deterministic stand-in: one id per word, so head length is predictable."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def id_of(self, word):
        """The single id this tokenizer assigns to a one-word string."""
        return self(" " + word)["input_ids"][0]


def _head_block(tok, q, head_max_len):
    """Length of the built prompt the state is appended to, excluding the trailing [SEP]."""
    seq, _ = build_sequence(tok, "", q, max_len=10 ** 6, head_max_len=head_max_len)
    return len(seq) - 1


def _kept_state(tok, state, q, max_len, head_max_len):
    head = _head_block(tok, q, head_max_len)
    seq, _ = build_sequence(tok, state, q, max_len=max_len, head_max_len=head_max_len)
    return seq[head:-1]


def test_head_never_exceeds_the_cap_by_more_than_the_delimiters():
    """head_len <= head_max_len + 3 (CLS plus a [SEP] on each side of the marker block)."""
    tok = _FakeTok()
    for head_max_len in (32, 64, 192, 256, 512):
        head = _head_block(tok, Q, head_max_len)
        assert head <= head_max_len + 3, (
            "head=%d exceeds cap %d (+3)" % (head, head_max_len))


def test_state_budget_is_sized_from_the_actual_head():
    """A state longer than the room is cut to exactly `max_len - head_len - 1`."""
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    head = _head_block(tok, Q, head_max_len)
    assert head < head_max_len, "fixture assumes a question that does not fill the cap"

    room = max_len - head - 1
    state = " ".join("w%d" % i for i in range(room + 200))
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    check("kept state length is max_len - head_len - 1", len(kept), room)
    check("room exceeds the cap-based figure", room > max_len - head_max_len, True)


def test_cap_based_figure_understates_the_room():
    """A state of exactly `max_len - head_max_len` tokens is NOT truncated.

    This is the assertion the README's old wording contradicts: it told a caller that
    `max_len - head_max_len` was all the state they had.
    """
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    documented = max_len - head_max_len          # 320 under the old wording
    head = _head_block(tok, Q, head_max_len)
    real_room = max_len - head - 1

    assert real_room > documented, (
        "fixture assumes the cap-based figure understates the room")

    marker = "UNIQUETAILMARKER"
    state = " ".join("w%d" % i for i in range(documented - 3)) + " " + marker
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    assert tok.id_of(marker) in kept, (
        "a %d-token state (the cap-based figure) must survive a %d budget; "
        "it was truncated, so the room is smaller than the README claims"
        % (documented, max_len))
    check("full documented-size state survives", len(kept), documented - 2)


def test_shipped_checkpoints_match_the_documented_room():
    """With each checkpoint's own tokenizer, the state room is the figure the README quotes.

    Skipped when the checkpoints are not present (CI runs without weights).
    """
    lab = os.environ.get("LAYA_LAB_MODELS", "/home/parshu/projects/contri/laya-lab/models")
    cases = [("english", 512, 192, 48, 463), ("multilingual", 1024, 256, 45, 978)]
    try:
        from transformers import AutoTokenizer
    except Exception:
        print("  (skipped: transformers unavailable)")
        return
    checked = 0
    for name, max_len, head_max_len, want_head, want_room in cases:
        path = os.path.join(lab, name, "tokenizer")
        if not os.path.isdir(path):
            print("  (skipped %s: no tokenizer at %s)" % (name, path))
            continue
        tok = AutoTokenizer.from_pretrained(path)
        head = _head_block(tok, Q, head_max_len)
        check("%s head length" % name, head, want_head)
        check("%s state room" % name, max_len - head - 1, want_room)
        checked += 1
    if not checked:
        print("  (skipped: no checkpoints under %s)" % lab)


if __name__ == "__main__":
    for fn in (test_head_never_exceeds_the_cap_by_more_than_the_delimiters,
               test_state_budget_is_sized_from_the_actual_head,
               test_cap_based_figure_understates_the_room,
               test_shipped_checkpoints_match_the_documented_room):
        try:
            fn()
        except AssertionError as e:
            FAIL.append("%s: %s" % (fn.__name__, e))
        else:
            PASS.append(fn.__name__)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    sys.exit(1 if FAIL else 0)
