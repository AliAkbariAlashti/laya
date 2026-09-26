"""The CVE audit job must see every name the package declares, not just the core five.

`.github/workflows/security.yml` builds `requirements-audit.txt` from `pyproject.toml` and hands it
to `pip-audit`. For a while it did that with a regex anchored on `dependencies = [`, which matches
the `[project]` array and none of `[project.optional-dependencies]` -- so the ten names behind the
`serve`, `fast`, `mcp`, `structured`, `onnx`, `langchain` and `langgraph` extras, and the 57
transitive names they pull in, were outside the audit. `Dockerfile` ends with
`pip install ".[serve]" && pip check`: the HTTP server's dependencies ship in the published image.

Text parsing, not tomllib: the floor is 3.10 and tomllib arrives in 3.11, same as
`tests/test_packaging.py`. The job that runs the extractor pins `python-version: "3.11"`, which is
checked here, and the lifted code is only executed where tomllib exists -- on 3.10 that check
degrades to a note and never to a pass.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL, NOTES = [], [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        # %s, not concatenation: several details below are lists of names, and a gate that raises
        # on the way to reporting a failure leaves the mutant's name unsaid.
        FAIL.append("%s: %s" % (name, detail))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def section(text, header):
    """Lines of one top-level-`  ` job/table, from `header` to the next same-indent key."""
    lines = text.splitlines()
    depth = len(header) - len(header.lstrip())
    out, seen = [], False
    for line in lines:
        if seen and (line.strip() and (len(line) - len(line.lstrip())) <= depth
                     and not line.lstrip().startswith("#")):
            break
        if line.rstrip() == header.rstrip():
            seen = True
        if seen:
            out.append(line)
    return out


# ---------------------------------------------------------------- the declared sets
pyproject = read("pyproject.toml")


def array(body):
    return re.findall(r'"([^"]+)"', body)


core_block = re.search(r"^dependencies = \[(.*?)\]", pyproject, re.S | re.M)
core = array(core_block.group(1)) if core_block else []

# Every `name = [ ... ]` under [project.optional-dependencies], before the next table header.
od = pyproject.split("[project.optional-dependencies]", 1)
extras = {}
if len(od) > 1:
    table = od[1].split("\n[", 1)[0]
    for m in re.finditer(r"^([A-Za-z0-9_-]+) = \[(.*?)\]", table, re.S | re.M):
        extras[m.group(1)] = array(m.group(2))

check_true("pyproject/core parses", len(core) == 5, core)
check("pyproject/extra tables found", sorted(extras),
       ["fast", "langchain", "langgraph", "mcp", "onnx", "serve", "structured"])
declared = list(core) + [s for names in extras.values() for s in names]
expected = sorted(set(declared))
check("pyproject/extras add names the core does not have",
       sorted({re.split(r"[<>=!;\[ ]", s)[0] for s in declared}
              - {re.split(r"[<>=!;\[ ]", s)[0] for s in core}),
       ["fastapi", "langchain-core", "langgraph", "mcp", "onnx", "onnxruntime", "pydantic",
        "python-multipart", "tilelang", "uvicorn"])

# ---------------------------------------------------------------- the job's own extractor
deps_job = "\n".join(section(read(os.path.join(".github", "workflows", "security.yml")),
                             "  deps:"))
check_true("security.yml/has a deps job", "pip-audit" in deps_job, deps_job[:80])

m = re.search(r"python - <<'PY' > requirements-audit\.txt\n(.*?)\n\s*PY\n", deps_job, re.S)
check_true("security.yml/exactly one heredoc writes the audit list",
           m is not None and len(re.findall(r"<<'PY'", deps_job)) == 1,
           "the extractor has to be liftable as text for this file to check it")
extractor = ""
if m:
    body = m.group(1)
    indents = [len(l) - len(l.lstrip()) for l in body.splitlines() if l.strip()]
    extractor = "\n".join(l[min(indents):] if l.strip() else "" for l in body.splitlines())

# Why the extractor is read out of the workflow instead of re-implemented here: the failure mode
# this file exists for is the two surfaces drifting, so a check that carries its own copy of the
# parsing rules would drift in exactly the same direction and stay green.
check_true("extractor/reads the tables with tomllib, not a regex on one array",
           "tomllib" in extractor and "dependencies = \\[" not in extractor, extractor[:200])
check_true("extractor/walks optional-dependencies",
           "optional-dependencies" in extractor, extractor[:200])
check_true("extractor/keeps the declared-not-installed reason",
           "local version" in deps_job and "+cpu" in deps_job,
           "auditing an install would trip --strict on torch's 2.14.0+cpu")
check_true("extractor/still audits the file strictly",
           re.search(r"pip-audit --strict[^\n]*-r requirements-audit\.txt", deps_job) is not None,
           deps_job[:120])
check_true("extractor/prints the scope it audited",
           re.search(r'echo "[^"]*requirements-audit\.txt', deps_job) is not None,
           "when the job goes red the log has to say how many names were in scope")

pin = re.search(r"python-version:\s*[\"'](\d+)\.(\d+)[\"']", deps_job)
check_true("security.yml/the deps job pins python >= 3.11",
           pin is not None and tuple(map(int, pin.groups())) >= (3, 11),
           "tomllib, which the extractor reads the tables with, is 3.11+")

ran = False
if sys.version_info >= (3, 11) and extractor:
    ran = True
    run = subprocess.run([sys.executable, "-c", extractor], cwd=ROOT,
                         capture_output=True, text=True)
    check_true("extractor/runs against the real pyproject.toml",
               run.returncode == 0, run.stderr.strip()[:300])
    emitted = [l.strip() for l in run.stdout.splitlines() if l.strip()]
    check("extractor/emits every declared name, core and extras alike", sorted(set(emitted)),
          expected)
    # Specifiers carried through, not stripped: `fastapi` alone would audit whatever PyPI serves
    # today and hide the floor the package actually supports.
    check_true("extractor/preserves each version specifier",
               all(e in declared for e in emitted),
               [e for e in emitted if e not in declared])
    check_true("extractor/collapses the langchain/langgraph duplicate",
               len(emitted) == len(set(emitted)),
               [s for s in emitted if emitted.count(s) > 1])
    for name in ("fastapi", "uvicorn", "python-multipart"):
        check_true("extractor/audits %s, which the image installs" % name,
                   any(re.split(r"[<>=!;\[ ]", s)[0] == name for s in emitted),
                   "Dockerfile installs it through the `serve` extra")
else:
    NOTES.append("python %d.%d has no tomllib: the extractor was not executed here, only its "
                 "source was checked (the job pins 3.11+)" % sys.version_info[:2])
# Where the interpreter can run the extractor, it must have. Without this the skip above is a
# hole a mutant can open -- `if False` would leave the file green with seven checks missing.
check_true("extractor/ran wherever tomllib exists",
           ran or sys.version_info < (3, 11), "the executable checks were skipped on a 3.11+ run")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for note in NOTES:
    print("  NOTE " + note)
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
