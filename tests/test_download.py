"""Checkpoint download regression tests; tiny local weights, no network required.

Also pins the `LAYA_REVISION` lever of `laya/revisions.py`: which revision reaches `snapshot_download`,
what the Agent and the Router report back, and the two loads that must not see it (a local
directory, and `common.build_model`'s training-time base encoder).

Run: python tests/test_download.py
"""
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

from laya import load  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from laya.revisions import PINNED_REVISIONS  # noqa: E402


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

    # ------------------------------------------------------------------ LAYA_REVISION
    #
    # `PINNED_REVISIONS` published the reviewed commit of each published checkpoint and nothing in
    # `laya/` read it, while `LAYA_SHA256_DIGESTS` -- the other half of the same opt-in pair, in
    # the same module -- was already reachable from the environment. These replace only the Hub
    # transport, so the revision that reaches `snapshot_download`, the one the Agent reports back,
    # and the one a Router publishes in `loaded_revisions` (what `/health` serves) are all the real
    # thing. They live here rather than in `tests/test_revision_pinning.py` because this file's
    # subject is already "what the download call was given", and because this suite runs in CI
    # (`ci.yml:107`, `release.yml:47`) while that one is in no workflow.

    def hub_load(self, repo, env=None, **load_kwargs):
        """`load(repo)` with the Hub transport replaced. Returns (kwargs sent, agent, error)."""
        with patch.dict(os.environ), patch("huggingface_hub.snapshot_download",
                                           return_value=str(self.repo)) as download:
            if env is None:
                os.environ.pop("LAYA_REVISION", None)
            else:
                os.environ["LAYA_REVISION"] = env
            try:
                agent = load(repo, device="cpu", **load_kwargs)
            except Exception as error:                # the cases below assert on it
                return download, None, error
            return download, agent, None

    def sent_revision(self, download):
        return (download.call_args.kwargs if download.call_args else {}).get("revision")

    def test_no_pin_asked_for_keeps_the_hub_default(self):
        download, agent, error = self.hub_load("convaiinnovations/laya")
        self.assertIsNone(error)
        self.assertNotIn("revision", download.call_args.kwargs)
        self.assertIsNone(agent.revision)
        self.assertEqual(agent.predict("hello", self.questions), self.expected)

    def test_environment_revision_reaches_the_download(self):
        download, agent, error = self.hub_load("convaiinnovations/laya", env="ab" * 20)
        self.assertIsNone(error)
        self.assertEqual(self.sent_revision(download), "ab" * 20)
        self.assertEqual(agent.revision, "ab" * 20)

    def test_reviewed_resolves_each_repositorys_own_pin(self):
        self.assertGreater(len(set(PINNED_REVISIONS.values())), 1,
                           "the table has one pin per repository, so this sweep is meaningful")
        for repo, sha in PINNED_REVISIONS.items():
            with self.subTest(repo=repo):
                download, agent, error = self.hub_load(repo, env="reviewed")
                self.assertIsNone(error)
                self.assertEqual(self.sent_revision(download), sha)
                self.assertEqual(agent.revision, sha)

    def test_reviewed_refuses_a_repository_with_no_pin(self):
        download, agent, error = self.hub_load("acme/custom-checkpoint", env="reviewed")
        self.assertIsInstance(error, ValueError)
        self.assertIn("acme/custom-checkpoint", str(error))
        self.assertIn("LAYA_REVISION", str(error))
        self.assertIsNone(agent)
        download.assert_not_called()

    def test_an_explicit_revision_outranks_the_environment(self):
        download, agent, error = self.hub_load("convaiinnovations/laya", env="reviewed",
                                               revision="cd" * 20)
        self.assertIsNone(error)
        self.assertEqual(self.sent_revision(download), "cd" * 20)
        self.assertNotEqual(agent.revision, PINNED_REVISIONS["convaiinnovations/laya"])

    def test_an_empty_or_blank_environment_value_is_not_a_pin(self):
        for env in ("", "   ", "\t"):
            with self.subTest(repr=repr(env)):
                download, agent, error = self.hub_load("convaiinnovations/laya", env=env)
                self.assertIsNone(error)
                self.assertNotIn("revision", download.call_args.kwargs)
                self.assertIsNone(agent.revision)

    def test_a_local_directory_never_consults_the_environment(self):
        download, agent, error = self.hub_load(str(self.repo), env="reviewed")
        self.assertIsNone(error)
        download.assert_not_called()
        self.assertIsNone(agent.revision)
        self.assertEqual(agent.predict("hello", self.questions), self.expected)

    def test_router_publishes_the_pinned_commit(self):
        """`/health`'s `revisions` block is `Router.loaded_revisions`, so the pin shows up there."""
        from laya import Router

        bundle = "convaiinnovations/laya"
        with patch.dict(os.environ), patch("huggingface_hub.snapshot_download",
                                           return_value=str(self.repo)):
            os.environ.pop("LAYA_REVISION", None)
            unpinned = Router()
            unpinned.load("english")
            self.assertEqual(unpinned.loaded_revisions, {"english": None})

            os.environ["LAYA_REVISION"] = "reviewed"
            pinned = Router()
            pinned.load("english")
            self.assertEqual(pinned.loaded_revisions, {"english": PINNED_REVISIONS[bundle]})

    def test_the_onnx_loader_resolves_through_the_same_function(self):
        """`ONNXAgent` must not grow its own copy of the rule, or the env pin covers one loader."""
        from laya import onnx_agent, revisions

        self.assertIs(onnx_agent.resolve_revision, revisions.resolve_revision)

    def test_the_base_encoder_load_ignores_the_environment(self):
        """`common.build_model`'s training-time load is a different repository, with no reviewed
        SHA to resolve to, so `LAYA_REVISION` deliberately does not reach it."""
        import laya.common as common

        seen = {}

        def from_pretrained(name, **kwargs):
            seen.update(kwargs)
            seen["name"] = name
            raise RuntimeError("stop after the call")

        cfg = {"encoder": "unused/base-encoder", "head_layers": 1, "act_costs": {"a": 0}}
        with patch.dict(os.environ, {"LAYA_REVISION": "reviewed"}), \
                patch("transformers.AutoModel.from_pretrained", side_effect=from_pretrained):
            with self.assertRaises(RuntimeError):
                common.build_model(cfg, pretrained=True)
        self.assertEqual(seen["name"], "unused/base-encoder")
        self.assertNotIn("revision", seen)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
