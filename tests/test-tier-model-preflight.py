"""The tier switch preflight accepts only model snapshots its loaders can read."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import constants

from truememory.tier_switch import preflight

COMPLETE = {"config.json": "{}", "model.safetensors": "weights", "tokenizer.json": "{}"}


def cache(directory: str, repository: str, files: dict[str, str]) -> None:
    root = Path(directory) / ("models--" + repository.replace("/", "--"))
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("0123456789abcdef", encoding="utf-8")
    snapshot = root / "snapshots" / "0123456789abcdef"
    snapshot.mkdir(parents=True)
    for name, content in files.items():
        (snapshot / name).write_text(content, encoding="utf-8")


class PartialSnapshots(unittest.TestCase):
    def test_only_snapshots_with_config_weights_and_tokenizer_count(self):
        # Review v5 L-v5-3: a cache holding only config.json passed the old
        # check, and the reranker then failed to load after a full re-embed.
        index = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
        repositories = {
            "synthetic/config-only": ({"config.json": "{}"}, False),
            "synthetic/no-tokenizer": ({"config.json": "{}", "model.safetensors": "weights"}, False),
            "synthetic/no-config": ({"model.safetensors": "weights", "tokenizer.json": "{}"}, False),
            "synthetic/complete": (COMPLETE, True),
            "synthetic/bin-and-vocab": ({"config.json": "{}", "pytorch_model.bin": "weights", "vocab.txt": "x"}, True),
            "synthetic/sharded-partial": ({"config.json": "{}", "tokenizer.json": "{}", "model.safetensors.index.json": index,
                                           "model-1.safetensors": "weights"}, False),
            "synthetic/sharded": ({"config.json": "{}", "tokenizer.json": "{}", "model.safetensors.index.json": index,
                                   "model-1.safetensors": "weights", "model-2.safetensors": "weights"}, True),
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(constants, "HF_HUB_CACHE", directory):
            for repository, (files, _) in repositories.items():
                cache(directory, repository, files)
            found = {repository: preflight.cached_snapshot(repository) for repository in repositories}
            absent = preflight.cached_snapshot("synthetic/never-downloaded")
        self.assertEqual(found, {repository: expected for repository, (_, expected) in repositories.items()})
        self.assertFalse(absent)

    def test_switch_preflight_names_a_partially_downloaded_reranker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(constants, "HF_HUB_CACHE", directory), \
                patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            cache(directory, "minishlab/potion-base-8M", COMPLETE)
            cache(directory, "cross-encoder/ms-marco-MiniLM-L-6-v2", {"config.json": "{}"})
            self.assertEqual(preflight.missing_models("edge"), ["cross-encoder/ms-marco-MiniLM-L-6-v2"])

    def test_local_model_directory_needs_the_same_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "local-model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            partial = preflight.cached_snapshot(str(model))
            for name, content in COMPLETE.items():
                (model / name).write_text(content, encoding="utf-8")
            complete = preflight.cached_snapshot(str(model))
        self.assertEqual((partial, complete), (False, True))


if __name__ == "__main__":
    unittest.main()
