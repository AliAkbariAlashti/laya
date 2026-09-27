"""Checkpoint download regression tests; tiny local weights, no network required.

Run: python tests/test_download.py
"""
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from huggingface_hub.utils import filter_repo_objects  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import Agent, load  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402


class _NoRuntime:
    """Stand-in for the onnxruntime module. `ONNXAgent.__init__` binds the name before it
    touches the Hub, and the download asserted on below happens well before a session is
    built, so nothing here is ever called."""


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tmp.name) / "repo"
        cls.repo.mkdir()
        config = BertConfig(vocab_size=6, hidden_size=64, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=128)
        config.save_pretrained(cls.repo / "encoder")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]",
        )
        tokenizer.save_pretrained(cls.repo / "tokenizer")
        model = DecisionModel(BertModel(config), head_layers=0)
        save_file(model.state_dict(), cls.repo / "model.safetensors")
        (cls.repo / "rl_agent_config.json").write_text(json.dumps({
            "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
            "max_len": 64, "head_max_len": 32,
        }))
        # Hub repository paths use '/' even when the fixture lives on Windows.
        cls.runtime_files = {p.relative_to(cls.repo).as_posix()
                             for p in cls.repo.rglob("*") if p.is_file()}
        for subfolder in ("multilingual", "typed-decisions", "variants/english"):
            for filename in cls.runtime_files:
                target = cls.repo / subfolder / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cls.repo / filename, target)
        (cls.repo / "README.md").write_text("An unrelated model card")
        (cls.repo / "eval").mkdir()
        (cls.repo / "eval" / "results.json").write_text("{}")
        cls.questions = {"q": {"type": "choice", "instructions": "Pick one",
                               "criteria": ["yes", "no"]}}
        cls.expected = load(str(cls.repo), device="cpu").predict("hello", cls.questions)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_download(self, repo_id, subfolder=None):
        # Use the Hub's real file-filter semantics, then load and run the selected files.
        # Only transport is replaced; tokenizer, weights, model construction and inference
        # use the same code as a real checkpoint download.
        with tempfile.TemporaryDirectory() as destination:
            downloaded = []

            def snapshot(repo_id, **kwargs):
                files = [p.relative_to(self.repo).as_posix()
                         for p in self.repo.rglob("*") if p.is_file()]
                selected = filter_repo_objects(files, allow_patterns=kwargs.get("allow_patterns"),
                                               ignore_patterns=kwargs.get("ignore_patterns"))
                for filename in selected:
                    target = Path(destination) / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.repo / filename, target)
                    downloaded.append(filename)
                return destination

            with patch("huggingface_hub.snapshot_download", side_effect=snapshot) as download:
                agent = load(repo_id, device="cpu", subfolder=subfolder, token="test-token")
            self.assertEqual(download.call_count, 1)
            self.assertEqual(download.call_args.args[0], repo_id)
            self.assertEqual(download.call_args.kwargs["token"], "test-token")
            self.assertEqual(agent.predict("hello", self.questions), self.expected)
            prefix = subfolder + "/" if subfolder else ""
            self.assertEqual(set(downloaded), {prefix + name for name in self.runtime_files})

    def test_default_english_does_not_download_sibling_checkpoints(self):
        self.check_download("convaiinnovations/laya")

    def test_custom_root_checkpoint(self):
        self.check_download("test/custom-model")

    def test_each_subfolder_loads_independently(self):
        for subfolder in ("multilingual", "typed-decisions", "variants/english"):
            with self.subTest(subfolder=subfolder):
                self.check_download("test/bundled-models", subfolder)

    def test_local_paths_do_not_download(self):
        with patch("huggingface_hub.snapshot_download") as download:
            for subfolder in (None, "multilingual", "variants/english"):
                with self.subTest(subfolder=subfolder):
                    agent = load(str(self.repo), device="cpu", subfolder=subfolder)
                    self.assertEqual(agent.predict("hello", self.questions), self.expected)
            download.assert_not_called()

    def test_loading_skips_initialization_and_preserves_weights(self):
        weights = load_file(self.repo / "model.safetensors")
        for bundled_encoder in (True, False):
            with self.subTest(bundled_encoder=bundled_encoder), tempfile.TemporaryDirectory() as destination:
                path = Path(destination)
                for filename in self.runtime_files:
                    if not bundled_encoder and filename.startswith("encoder/"):
                        continue
                    target = path / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.repo / filename, target)
                cfg_path = path / "rl_agent_config.json"
                cfg = json.loads(cfg_path.read_text())
                cfg["encoder"] = str(self.repo / "encoder")
                cfg_path.write_text(json.dumps(cfg))
                rng = torch.get_rng_state()
                with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("Unused base weights")):
                    agent = load(str(path), device="cpu")
                self.assertTrue(torch.equal(torch.get_rng_state(), rng), "Loading initialized random weights")
                for name, value in agent.model.state_dict().items():
                    torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
                self.assertEqual(agent.predict("hello", self.questions), self.expected)

    def test_onnx_agent_accepts_every_hub_option_the_agent_does(self):
        # Both runtimes download the same checkpoint from the same place, so an option that
        # selects *which* checkpoint, or *how* to authenticate for it, has to exist on both.
        # Read off the signatures so the next Hub-side option added to Agent has to be
        # copied across rather than silently diverging again.
        #   fast/compile -- the TileLang path and torch.compile, neither of which exists
        #                    inside onnxruntime.
        #   device       -- ONNXAgent picks an execution provider from what onnxruntime
        #                    reports and takes no override; a different asymmetry, with its
        #                    own fix.
        not_for_onnxruntime = {"fast", "compile", "device"}
        agent_side = (set(inspect.signature(Agent.__init__).parameters)
                      - {"self", "model_id_or_path"} - not_for_onnxruntime)
        onnx_side = set(inspect.signature(ONNXAgent.__init__).parameters) - {"self"}
        missing = agent_side - onnx_side
        self.assertEqual(missing, set(),
                         "ONNXAgent cannot set: %s" % ", ".join(sorted(missing)))

    def test_onnx_agent_forwards_the_token_to_the_download(self):
        # `token` is only useful if it reaches snapshot_download. Drive the real constructor
        # with the transport replaced and read the call it makes: the graph file is absent
        # on purpose, so the load stops right after the download it is meant to authenticate.
        # onnxruntime is imported at the top of __init__ and never used before that stop, so
        # an empty stand-in keeps this a no-extra-required check.
        with tempfile.TemporaryDirectory() as snapshot:
            (Path(snapshot) / "rl_agent_config.json").write_text(json.dumps({
                "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
                "max_len": 64, "head_max_len": 32,
            }))
            with patch.dict(sys.modules, {"onnxruntime": _NoRuntime}), \
                    patch("huggingface_hub.snapshot_download", return_value=snapshot) as download:
                with self.assertRaises(FileNotFoundError):
                    ONNXAgent("test/private-model", token="test-token",
                              onnx_path=str(self.repo / "absent.onnx"))
                self.assertEqual(download.call_args.kwargs["token"], "test-token")
                # the allow-list is unchanged: a token must not widen what gets fetched
                self.assertNotIn("model.safetensors", download.call_args.kwargs["allow_patterns"])

    def test_onnx_agent_token_defaults_and_reads_the_environment(self):
        with tempfile.TemporaryDirectory() as snapshot:
            (Path(snapshot) / "rl_agent_config.json").write_text(json.dumps({
                "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
                "max_len": 64, "head_max_len": 32,
            }))
            with patch.dict(sys.modules, {"onnxruntime": _NoRuntime}), \
                    patch("huggingface_hub.snapshot_download", return_value=snapshot) as download:
                with self.assertRaises(FileNotFoundError):
                    ONNXAgent("test/private-model", onnx_path=str(self.repo / "absent.onnx"))
                self.assertIsNone(download.call_args.kwargs["token"])

                with patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
                    with self.assertRaises(FileNotFoundError):
                        ONNXAgent("test/private-model", onnx_path=str(self.repo / "absent.onnx"))
                    self.assertEqual(download.call_args.kwargs["token"], "env-token")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
