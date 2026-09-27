"""Packaging metadata must match what the dependencies actually need (#34).

Text parsing, not tomllib: the floor is 3.10 and tomllib arrives in 3.11.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def version_tuple(text):
    return tuple(int(part) for part in text.split("."))


pyproject = read("pyproject.toml")
setup_py = read("setup.py")

requires_python = re.search(r'requires-python\s*=\s*"[>=~^]*\s*([\d.]+)"', pyproject)
check_true("pyproject/declares requires-python", requires_python is not None)
floor = version_tuple(requires_python.group(1)) if requires_python else (0, 0)

classifier_versions = [
    version_tuple(v)
    for v in re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', pyproject)
]
check_true("pyproject/advertises specific Python versions", len(classifier_versions) > 0)
below_floor = [".".join(str(p) for p in v) for v in classifier_versions if v < floor]
check("classifiers/none below requires-python", below_floor, [])

# Checkpoints run on answerdotai/ModernBERT-large, which transformers only knows from 4.48.
transformers_floor = re.search(r'"transformers>=([\d.]+)"', pyproject)
check_true("pyproject/pins a transformers floor", transformers_floor is not None)
check_true(
    "transformers/floor covers ModernBERT",
    transformers_floor is not None and version_tuple(transformers_floor.group(1)) >= (4, 48),
    "ModernBERT support starts in transformers 4.48",
)

for field in ("python_requires", "install_requires", "classifiers"):
    check_true(
        "setup.py/does not duplicate %s" % field,
        field not in setup_py,
        "metadata belongs in pyproject.toml only",
    )

# The release job checks the git tag against pyproject, but laya.__version__ is what the server
# and SDK report at runtime, so the two strings must not drift apart.
static_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
check_true("pyproject/declares a static version", static_version is not None)
init_version = re.search(r'^__version__\s*=\s*"([^"]+)"', read(os.path.join("laya", "__init__.py")), re.M)
check_true("laya/__init__ declares a literal __version__", init_version is not None)
check(
    "version/pyproject matches laya.__version__",
    static_version.group(1) if static_version else None,
    init_version.group(1) if init_version else None,
)
for label, match in (("pyproject", static_version), ("laya/__init__", init_version)):
    value = match.group(1) if match else ""
    parts = value.split(".")
    check_true("%s/version is X.Y.Z" % label, len(parts) == 3 and all(p.isdigit() for p in parts), "got %r" % value)

workflow = read(os.path.join(".github", "workflows", "ci.yml"))
ci_versions = [version_tuple(v) for v in re.findall(r'"(\d+\.\d+)"', workflow)]
stale = [".".join(str(p) for p in v) for v in ci_versions if v < floor]
check("ci/tests no Python below requires-python", stale, [])

# Classifiers on PyPI are a support claim. If a version is listed there, CI must run it
# (3.12 was advertised while the matrix jumped 3.11 -> 3.13).
missing_from_ci = [
    ".".join(str(p) for p in v) for v in classifier_versions if v not in ci_versions
]
check("ci/tests every advertised Python version", missing_from_ci, [])


# --------------------------------------------------------------- markdown links
# Nothing checked these, and the README ships to PyPI and to the model card. Only
# targets inside the repository are checked: external URLs would make the suite depend
# on the network, which it must not. The README's Router link pointed at a heading that
# had been renamed, so it silently went nowhere for as long as the rename was in.
def _slug(heading):
    """GitHub's heading anchor: drop anything that is not word/space/hyphen, lowercase,
    then spaces to hyphens."""
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def _headings(path):
    found = set()
    for line in read(path).splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if match:
            found.add(_slug(match.group(1)))
    return found


_md = []
for _dirpath, _dirnames, _filenames in os.walk("."):
    # `.pytest_cache` ships a README of its own and is not part of the repository.
    _dirnames[:] = [d for d in _dirnames
                    if d not in (".git", "__pycache__", "node_modules", ".pytest_cache")]
    _md.extend(os.path.normpath(os.path.join(_dirpath, f))
               for f in _filenames if f.endswith(".md"))
_md = sorted(_md)
_heading_cache = {p: _headings(p) for p in _md}

check_true("md/at least the README, BENCHMARKS and docs are scanned", len(_md) >= 8, _md)
check_true("md/no build directory scanned",
           not any(".pytest_cache" in p for p in _md), _md)

_broken_files, _broken_anchors = [], []
for _path in _md:
    for _label, _target in re.findall(r"\[([^\]]*)\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)",
                                      read(_path)):
        if _target.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        _tpath, _, _frag = _target.partition("#")
        _dest = os.path.normpath(os.path.join(os.path.dirname(_path), _tpath)) if _tpath else _path
        if _tpath and not os.path.exists(_dest):
            _broken_files.append("%s: [%s](%s)" % (_path, _label[:30], _target))
            continue
        if _frag and _dest.endswith(".md") and _frag not in _heading_cache.get(_dest, set()):
            _broken_anchors.append("%s: [%s](%s)" % (_path, _label[:30], _target))

check("md/no link to a file that does not exist", _broken_files, [])
check("md/no anchor that matches no heading", _broken_anchors, [])
# ---------------------------------------------------------------- Compose layout
# `compose.http.yaml` is an override, so it is merged onto `compose.yaml` rather than
# read on its own. These checks are textual because the suite takes no third-party
# dependency and PyYAML is not one; the real validation is `docker compose config`,
# which the Docker workflow runs for every file combination.
http = read("compose.http.yaml")
base = read("compose.yaml")
cuda = read("compose.cuda.yaml")
dockerfile = read("Dockerfile")

check_true("compose.http/declares laya-serve", "laya-serve:" in http, http[:200])
check_true("compose.http/runs the server command",
           'command: ["laya-serve"]' in http, "laya-serve is not the container command")
check_true("compose.http/publishes a port", re.search(r"^\s*ports:", http, re.M) is not None)
# Host and container port must come from the same variable, or they drift apart and the
# published port stops reaching the server.
port_map = re.search(r'-\s*"(?:\$\{LAYA_BIND_ADDRESS:-[^}]+\}:)?(\$\{[A-Z_]+:-(\d+)\}):(\$\{[A-Z_]+:-(\d+)\})"', http)
# The API is unauthenticated until LAYA_API_KEY is set, so exposure beyond the host is opt-in.
check_true("compose.http/publishes on loopback unless LAYA_BIND_ADDRESS is set",
           '"${LAYA_BIND_ADDRESS:-127.0.0.1}:' in http)
check_true("compose.http/has a healthcheck on /health",
           "healthcheck:" in http and "/health" in http)
check_true("compose.http/port mapping is present", port_map is not None, http[:300])
if port_map:
    check("compose.http/host port equals container port", port_map.group(2), port_map.group(4))
    check("compose.http/both sides use the same variable", port_map.group(1), port_map.group(3))
check_true("compose.http/the server reads the same variable",
           'LAYA_PORT: "${LAYA_PORT:-8000}"' in http)
check_true("compose.http/shares the model cache",
           "model-cache:/home/laya/.cache/huggingface" in http)
# The base service is what `docker compose run --rm laya` uses; publishing it a port or
# changing its command would be a breaking change to the quickstart.
check_true("compose.http/leaves the quickstart service alone",
           "laya:" not in http, "compose.http.yaml overrides the base `laya` service")

# `pip install .` alone puts no `laya-serve` in the image, so the extra is load-bearing.
check_true("Dockerfile/installs the serve extra", '".[serve]"' in dockerfile, dockerfile[:400])
check_true("Dockerfile/still runs pip check", "pip check" in dockerfile)

# torch 2.14's eager Triton kernels compile on the first CUDA inference and need a C compiler the
# slim runtime image does not have (#365). The kill switch keeps the stock kernels.
check_true("Dockerfile/runtime stage disables torch's native Triton JIT (#365)",
           re.search(r"^\s*TORCH_DISABLE_NATIVE_JIT=1", dockerfile.partition("AS runtime")[2], re.M) is not None,
           "without TORCH_DISABLE_NATIVE_JIT=1 a GPU image serves 500s while /health stays green")

# Overrides for `laya` never reach `laya-serve`, a separate service. If the CUDA override does
# not repeat the args for laya-serve, that service silently serves on CPU.
check_true("compose.cuda/covers laya-serve too",
           re.search(r"^\s{2}laya-serve:", cuda, re.M) is not None,
           "compose.cuda.yaml does not mention laya-serve, so GPU serving would be CPU")
check("compose.cuda/repeats the torch index for the base service",
      len(re.findall(r'TORCH_INDEX: "\$\{LAYA_TORCH_INDEX:-cu128\}"', cuda)), 2)
check("compose.cuda/repeats the device reservation for both services",
      len(re.findall(r"driver: nvidia", cuda)), 2)
check_true("compose.cuda/no stale reference to a missing file",
           "compose.http.yaml on the HTTP preview branch" not in cuda,
           "compose.cuda.yaml still describes compose.http.yaml as living on another branch")

# Every file the Docker workflow validates must exist.
for name in ("compose.yaml", "compose.example.yml", "compose.cuda.yaml", "compose.http.yaml",
             "compose.spark.yaml"):
    check_true("compose/%s exists" % name, os.path.exists(name))


# --------------------------------------------------------------- nix: the deployment layer
# Nix builds and evaluates nothing here in CI (`grep -rn nix .github/workflows/` is empty), so
# these textual checks are the only gate on the two files a NixOS host deploys from. Same style
# as the compose checks above: no nix binary, no third-party dependency.

nix_pkg = read(os.path.join("nix", "package.nix"))
nix_version = re.search(r'^\s*version\s*=\s*"([^"]+)"', nix_pkg, re.M)
check_true("nix/package.nix declares a version", nix_version is not None, nix_pkg[:200])
# The release job compares the tag to pyproject, and pyproject is compared to laya.__version__
# above. This is the third declaration of the same string -- it sat at 0.3.4 for sixteen
# releases because no reader existed.
check("version/nix matches pyproject",
      nix_version.group(1) if nix_version else None,
      static_version.group(1) if static_version else None)

# `services.laya-serve.models` is joined into LAYA_MODELS, which laya.serve splits and hands to
# Router.preload() -- and preload normalises every name (laya/router.py:370), so the server takes
# core's aliases. A closed `enum` in the module can only copy that list and fall behind it: it
# refused ten of the thirteen spellings the same value accepts, and it refused them at nix
# evaluation time, on the way to starting a service that would have been happy.
nix_module = read(os.path.join("nix", "laya-serve.nix"))
models_block = re.search(r"models = lib\.mkOption \{(.*?)\n    \};", nix_module, re.S)
check_true("nix/module has a models option", models_block is not None, nix_module[:200])
block = models_block.group(1) if models_block else ""
type_line = re.search(r"^\s*type\s*=\s*(.+?)\s*$", block, re.M)
check_true("nix/models type declaration found", type_line is not None, block[:200])
declared_type = type_line.group(1) if type_line else ""
check_true("nix/models declares no closed enum over checkpoint names",
           "types.enum" not in declared_type, "type = %s" % declared_type)

# The accepted name set, read out of core rather than transcribed: DEFAULT_MODELS' keys and the
# alias table. ast, not import -- this suite takes no third-party dependency, and torch is one.
def dict_keys(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    # Returning nothing rather than raising keeps the failure a named check: the population
    # guard below reports it. A gate that dies on the way to reporting leaves the cause unsaid.
    return []


router_src = read(os.path.join("laya", "router.py"))
canonical = dict_keys(router_src, "DEFAULT_MODELS")
aliases = dict_keys(router_src, "_ALIASES")
accepted = sorted(set(canonical) | set(aliases))
# Non-vacuity: the derivation has to have found both tables, or every comparison against
# `accepted` below would pass by matching an empty set.
check_true("nix/core name tables are both populated",
           len(canonical) >= 1 and len(aliases) >= 1,
           "DEFAULT_MODELS=%r _ALIASES=%r -- laya/router.py changed shape" % (canonical, aliases))
check("nix/models default is exactly core's canonical checkpoints",
      sorted(n.strip('"') for n in
             re.findall(r'default = \[([^\]]*)\]', block, re.S)[0].split())
      if re.search(r"default = \[", block) else None,
      sorted(canonical))

# Whatever the module tells its reader to type must be what core accepts, and the module runs no
# validator of its own -- so a name added to _ALIASES has to appear here or the option's own
# description goes stale on the day the alias lands.
description = re.search(r"description = ''(.*?)''", block, re.S)
check_true("nix/models has a description", description is not None, block[:200])
described = description.group(1) if description else ""
undocumented = [n for n in accepted if "`%s`" % n not in described]
check_true("nix/models describes every name core accepts", not undocumented,
           "accepted by laya.router.normalise_name but absent from the option text: %s"
           % undocumented)
# And the module must still be the thing that sets LAYA_MODELS, or the checks above describe a
# wire that no longer exists.
check_true("nix/module still joins models into LAYA_MODELS",
           re.search(r'LAYA_MODELS = lib\.concatStringsSep "," cfg\.models;', nix_module) is not None,
           "the models option no longer feeds LAYA_MODELS; these checks need retargeting")

# ---- every knob the server reads from the environment has to be reachable from the module
# `laya.serve` has no config file and no CLI flag for any of it: it configures itself from LAYA_*
# and nothing else. On a NixOS host the unit's environment is the only thing that can hand those
# variables over, so a name the module never assigns is a control that host cannot ask for. Two
# were missing: `LAYA_LOG_LEVEL`, and `LAYA_MAX_CONCURRENT` -- the admission bound, which is what
# keeps the bodies buffered in memory bounded and refuses the excess with 503.
serve_src = read(os.path.join("laya", "serve.py"))
served = sorted(
    set(re.findall(r'environ\.get\("(LAYA_[A-Z_]+)"', serve_src))
    | set(re.findall(r'_env_bool\("(LAYA_[A-Z_]+)"', serve_src))
)
# Non-vacuity again: if the derivation found nothing, both directions below pass for free.
check_true("nix/serve.py's environment reads were found", len(served) >= 5,
           "got %r -- retarget this if laya/serve.py changes shape" % (served,))

# An assignment only. The `models` description names `LAYA_MODELS` in prose, and prose that
# mentions a variable sets nothing.
assigned = sorted(set(re.findall(r'\b(LAYA_[A-Z_]+)\s*=', nix_module))
                  | set(re.findall(r'export (LAYA_[A-Z_]+)=', nix_module)))
check("nix/module reaches every env var laya.serve reads",
      [n for n in served if n not in assigned], [])
# The other direction is the silent failure: a misspelled name is a perfectly good string,
# systemd exports it, no Python ever looks at it, and the operator's setting does nothing.
check("nix/module assigns no env var laya.serve never reads",
      [n for n in assigned if n not in served], [])

# An option that nothing reads is a promise the unit does not keep, and the mirror case -- a
# setting the unit applies with no option to turn it -- is a host that cannot change it.
declared = re.findall(r"^    ([a-zA-Z]+) = lib\.mkOption \{", nix_module, re.M)
check_true("nix/module's options were found", len(declared) >= 5,
           "got %r -- retarget this if the option block changes shape" % (declared,))
check("nix/every declared option is used by the unit",
      [n for n in declared if "cfg.%s" % n not in nix_module], [])

# Both new options are opt-in: unset means the unit exports nothing and the server's own default
# applies, so a host that ignores them gets today's behaviour byte for byte.
for opt in ("logLevel", "maxConcurrent"):
    _b = re.search(r"^    %s = lib\.mkOption \{(.*?)\n    \};" % opt, nix_module, re.S | re.M)
    check_true("nix/module declares %s" % opt, _b is not None, "option not found")
    _t = _b.group(1) if _b else ""
    check_true("nix/%s is opt-in (nullOr, default null)" % opt,
               _t.count("nullOr") == 1 and re.search(r"^\s*default = null;", _t, re.M) is not None,
               _t.strip()[:120])
    check_true("nix/%s is guarded by a != null optionalAttrs" % opt,
               re.search(r"lib\.optionalAttrs \(cfg\.%s != null\)" % opt, nix_module) is not None,
               "the unit would export the variable even when the host left it unset")

# The same lesson as `models`, stated for the whole module: no option may carry a closed list of
# names that somebody else validates. uvicorn checks the log level, laya checks the checkpoint
# name, and a copy here can only fall behind them -- which is how the module came to refuse ten
# spellings of a value the server accepts.
check_true("nix/module declares no closed enum over names it does not own",
           "types.enum" not in nix_module, "an enum in this module is a copy of someone "
           "else's list; defer to the thing that validates the value")

# --------------------------------------------------------------- declared extras
# The runtime error in laya/structured.py tells users to install `laya[structured]`, and the
# docs and README repeat it. A reference to an extra pyproject.toml does not declare is a dead
# end for anyone who follows it, so every `laya[...]` in the code and docs must resolve (#348).
_extra_section = pyproject.split("[project.optional-dependencies]")[1].split("\n[")[0]
_declared_extras = set(re.findall(r"^([a-z][\w-]*)\s*=\s*\[", _extra_section, re.M))
_referenced_extras = {}
for _dirpath, _dirnames, _filenames in os.walk("."):
    # `.venv` is where CONTRIBUTING tells contributors to install, and it is not the repository.
    _dirnames[:] = [d for d in _dirnames
                    if d not in (".git", "__pycache__", "node_modules", ".pytest_cache", ".venv")]
    for _f in _filenames:
        if not _f.endswith((".py", ".md", ".yml", ".toml")):
            continue
        _p = os.path.normpath(os.path.join(_dirpath, _f))
        for _group in re.findall(r"laya\[([a-z][\w,-]*)\]", read(_p)):
            for _extra in _group.split(","):
                _referenced_extras.setdefault(_extra.strip(), set()).add(_p)

check("extras/every referenced extra is declared",
      sorted(set(_referenced_extras) - _declared_extras), [])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all packaging tests passed")
sys.exit(1 if FAIL else 0)
