"""Every autocast-dtype override the runtime reads has to be reachable from the shipped Compose
deployment.

`laya/agent.py` picks the dtype a forward runs in from the environment, and the deployment that
selects the device drops those names:

    $ LAYA_CUDA_AMP=fp16 docker compose -f compose.yaml -f compose.cuda.yaml config | grep LAYA_
          LAYA_DEVICE: cuda
          LAYA_DEVICE: cuda

The operator exported the variable, Compose saw it, and neither service's environment carries it,
so the image serves the checkpoint's own `amp_dtype` no matter what was set. That is not a
rounding difference: README's threshold section measures, on the `parity_fast.py` set (60 states,
288 questions per checkpoint, RTX 2000 Ada), bf16 flipping 3 of 864 argmaxes against the fp32
forward where fp16 flips none, at the same latency. The dtype is a policy dial, and the published
deployment could not turn it.

Why the names are derived rather than transcribed: the same three-way drift (package, markdown,
Compose) is what hid them, and a literal list in this file goes stale the moment the core reads
another one. So the set is read out of `laya/agent.py` -- the names it consumes *near a dtype
decision* -- and the rest of the file checks the deployment and the page against it, in both
directions. `LAYA_MPS_AMP_MIN_ROWS` is read by the same module and is deliberately not required
here, because no image in this repo can select MPS; check 5 keeps that reason falsifiable instead
of freezing it in a comment.

Weight-free, torch-free, no third-party YAML: the same textual style as the Compose section of
`tests/test_packaging.py`. The one docker-dependent check degrades to a note when docker is absent
and never to a pass.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List

PASS: List[str] = []
FAIL: List[str] = []
NOTES: List[str] = []

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSE_FILES = ("compose.yaml", "compose.cuda.yaml", "compose.spark.yaml", "compose.http.yaml")
NAME = re.compile(r"\bLAYA_[A-Z0-9_]+\b")
# The dtype decision itself: a torch dtype object or the checkpoint's `amp_dtype` field. Prose
# that merely names a precision ("MPS fp16 autocast") is not a decision site.
DTYPE_SITE = re.compile(r"torch\.(?:float16|bfloat16|float32)\b|\bamp_dtype\b")
ENVIRON_GET = re.compile(r'os\.environ\.get\("(LAYA_[A-Z0-9_]+)"')


def check(what: str, got: object, want: object) -> None:
    if got == want:
        PASS.append(what)
    else:
        FAIL.append("%s: got %r, want %r" % (what, got, want))


def check_true(what: str, cond: bool, detail: object = "") -> None:
    if cond:
        PASS.append(what)
    else:
        FAIL.append("%s: %s" % (what, detail))


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def dtype_names() -> List[str]:
    """Names `laya/agent.py` reads from the environment within a few lines of a dtype decision."""
    lines = read(os.path.join("laya", "agent.py")).splitlines()
    out = []
    for i, line in enumerate(lines):
        match = ENVIRON_GET.search(line)
        if not match:
            continue
        window = "\n".join(lines[max(0, i - 4): i + 5])
        if DTYPE_SITE.search(window):
            out.append(match.group(1))
    return sorted(set(out))


def env_blocks(text: str) -> Dict[str, Dict[str, str]]:
    """service name -> its `environment` mapping, as (key, raw value) pairs.

    Enough of YAML to read a flat `KEY: "value"` block; these files use no anchors, no
    flow mappings and one service per top-level key, which is what `test_packaging.py` already
    relies on for the same reason (no third-party parser in this suite).
    """
    services: Dict[str, Dict[str, str]] = {}
    service = None
    in_env = False
    in_services = False
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if indent == 0:
            # `services:` opens the block; any other top-level key (`volumes:`, `networks:`)
            # closes it, so a volume name is never mistaken for a service.
            in_services = stripped == "services:"
            service = None
            in_env = False
            continue
        if not in_services:
            continue
        if indent == 2 and stripped.endswith(":"):
            service = stripped[:-1].strip()
            services[service] = {}
            in_env = False
            continue
        if service is None:
            continue
        if indent == 4 and stripped == "environment:":
            in_env = True
            continue
        if indent == 4:
            in_env = False
            continue
        if in_env and indent == 6 and ":" in stripped and not stripped.startswith("- "):
            key, _, value = stripped.partition(":")
            services[service][key.strip()] = value.strip()
    return services


names = dtype_names()

# 1. the derived set is what this PR is about, so it must not be silently empty or renamed away.
check_true("core/derives dtype names from laya/agent.py", len(names) >= 2, names)
check("core/derives the CUDA and CPU dtype overrides", names, ["LAYA_CPU_AMP", "LAYA_CUDA_AMP"])
for name in names:
    check_true("core/%s is read with os.environ.get" % name,
               'os.environ.get("%s"' % name in read(os.path.join("laya", "agent.py")),
               "the name this file is enforcing is no longer how the core reads it")

# 2. every Compose file that selects a device forwards every derived name, in every service.
for rel in COMPOSE_FILES:
    text = read(rel)
    blocks = env_blocks(text)
    check_true("compose/%s parses into services" % rel, bool(blocks), sorted(blocks))
    if "LAYA_DEVICE" not in text:
        NOTES.append("%s: no LAYA_DEVICE, so nothing to forward" % rel)
        continue
    for service, env in sorted(blocks.items()):
        for name in names:
            value = env.get(name)
            check_true("compose/%s/%s forwards %s" % (rel, service, name), value is not None,
                       "%s is read by the runtime but the rendered container would not get it"
                       % name)
            if value is not None:
                # A passthrough *with a default*, not a bare `${NAME}`: without the `:-` a
                # deployment that never mentions the variable is treated as asking for it, and
                # the default here has a meaning the core implements (empty = the checkpoint's
                # own amp_dtype).
                check_true("compose/%s/%s/%s is a host passthrough" % (rel, service, name),
                           re.match(r'^"\$\{%s:-[^"}]*\}"$' % name, value) is not None,
                           "%s: %r -- hardcoding or requiring a dtype here would pin it for "
                           "every user" % (name, value))

# 3. an empty default has to mean "the checkpoint's own dtype", which is what these images shipped
#    before this change. Parsed out of the core so the claim is checked, not assumed.
for rel in COMPOSE_FILES:
    for service, env in sorted(env_blocks(read(rel)).items()):
        for name, value in env.items():
            if name in names and re.match(r'^"\$\{%s:-\}"$' % name, value):
                check_true("compose/%s/%s/%s empty default is safe" % (rel, service, name),
                           'os.environ.get("%s", "")' % name
                           in read(os.path.join("laya", "agent.py")),
                           "the core must treat an unset value as no override")

# 4. the page that documents Compose variables carries a row per derived name.
docker_md = read(os.path.join("docs", "docker.md"))
for name in names:
    check_true("docs/docker.md has a %s row" % name,
               re.search(r"^\| `%s` \|" % name, docker_md, re.M) is not None,
               "a variable Compose forwards with no row in the table is one a reader cannot find")

# 5. the exclusion this file relies on stays falsifiable: MPS's row gate is not forwarded because
#    no Compose file here can select MPS. The day one does, this check fails and the decision has
#    to be made again rather than quietly missing a name.
agent_src = read(os.path.join("laya", "agent.py"))
mps_names = sorted(set(NAME.findall(agent_src)) - set(names))
check_true("core/MPS names are the only runtime names excluded", mps_names == ["LAYA_MPS_AMP_MIN_ROWS"],
           mps_names)
devices = set()
for rel in COMPOSE_FILES:
    devices.update(re.findall(r'LAYA_DEVICE: "\$\{LAYA_DEVICE:-([a-z0-9]+)\}"', read(rel)))
check("compose/devices any file defaults to", sorted(devices), ["cpu", "cuda"])
check_true("docs/docker.md says why MPS is not forwarded",
           "LAYA_MPS_AMP_MIN_ROWS" in docker_md,
           "the omission has to be written down where a reader meets it")

# 6. the real renderer, when this machine has one: the rendered environment must carry the value.
if shutil.which("docker"):
    env = dict(os.environ, LAYA_CUDA_AMP="fp16")
    proc = subprocess.run(["docker", "compose", "-f", "compose.yaml", "-f", "compose.cuda.yaml",
                           "config"], cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        rendered = proc.stdout
        blocks = env_blocks(rendered)
        got = sorted([str(b.get("LAYA_CUDA_AMP", "<absent>")) for b in blocks.values()])
        check_true("render/both GPU services get LAYA_CUDA_AMP=fp16",
                   got == ["fp16", "fp16"],
                   {s: b.get("LAYA_CUDA_AMP") for s, b in blocks.items()})
        check_true("render/LAYA_MPS_AMP_MIN_ROWS stays out of the rendered environment",
                   all("LAYA_MPS_AMP_MIN_ROWS" not in b for b in blocks.values()),
                   sorted(k for b in blocks.values() for k in b if "MPS" in k))
    else:
        NOTES.append("docker compose config exited %d: %s" % (proc.returncode,
                                                              proc.stderr.strip()[:200]))
else:
    NOTES.append("docker not installed: checks 1-5 still hold; the render check did not run")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for note in NOTES:
    print("  NOTE " + note)
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
