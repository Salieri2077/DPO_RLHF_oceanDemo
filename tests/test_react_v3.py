"""ReAct v3/v3.1 data recipes: phrasing-bank hygiene, held-out isolation and executed demonstrations."""
import json
import re
import unittest
from collections import Counter
from pathlib import Path

from scripts import ocean_react_templates as bank
from scripts.prepare_ocean_react_v3 import (CONVERT_PAIRS, FIELDS, JOB_KIND, MAX_COMMON_RUN, MAX_JACCARD, RECIPES,
                                            V1_REPLAY, leakage_report, oracle, split_bank)

ROOT = Path(__file__).resolve().parents[1]
DATA = {name: ROOT / "data/processed" / name for name in ("react-v3", "react-v3-slots", "react-v3.1")}
AVAILABLE = [d for d in DATA.values() if (d / "agent_manifest.json").exists()]
REACT_V2 = ROOT / "data/processed/react-v2"


def placeholders(category):
    """(required, optional) placeholders of a phrasing category."""
    if category.startswith("calc."):
        return set(f + "_q" for f in FIELDS[category[5:]]), {"au"}
    if category.startswith("clarify."):
        _, op, missing = category.split(".")
        return {next(f for f in FIELDS[op] if f != missing) + "_q"}, set()
    if category.startswith("card."):
        return {f + "_q" for f in category[5:].split("_")}, set()
    if category.startswith(("chain.", "chain_form.")):
        return {"card"}, {"au"}
    return {"card_ref": ({"cid"}, set()), "convert.any": ({"value_q", "to_u"}, set()),
            "convert.distance": ({"value_q", "to_u"}, set()), "convert.speed": ({"value_q", "to_u"}, set()),
            "lookup_missing": ({"card"}, set()), "conflict_hint": ({"card"}, set()), "card_quote": ({"card"}, set()),
            "retrieve": ({"q"}, set()), "direct": ({"e"}, set()), "direct.delimited": ({"e"}, set()),
            "id_lead": ({"rid"}, set()), "id_tail": ({"rid"}, set()),
            "clarify_id": ({"rid", "given_cn", "missing_cn", "asked_cn"}, {"given_q"})}.get(category, (set(), set()))


class TemplateBankTests(unittest.TestCase):
    def test_placeholders_and_characters(self):
        categories = {**bank.CATEGORIES, **bank.CONTRAST_CATEGORIES}
        for category in set(categories) | set(bank.LEGACY_TRAIN):
            required, optional = placeholders(category)
            for template in categories.get(category, []) + bank.LEGACY_TRAIN.get(category, []):
                found = set(re.findall(r"\{([a-z_]+)\}", template))
                self.assertTrue(required <= found <= required | optional, (category, template))
                self.assertIsNone(re.search(r"\d|<|>", template), template)
        for ref in bank.ENGLISH_CARD_REFS:
            self.assertEqual(set(re.findall(r"\{([a-z_]+)\}", ref)), {"cid"})

    def test_heldout_families_and_v2_distance(self):
        families = split_bank()  # raises if a phrasing lands in two splits
        report = leakage_report(families)  # raises if training phrasing is close to v2 val/test
        self.assertLess(report["vs_react_v2_heldout"]["max_jaccard"], MAX_JACCARD)
        self.assertLess(report["vs_react_v2_heldout"]["max_common_run"], MAX_COMMON_RUN)
        for split in ("val", "test"):
            for category in {**bank.CATEGORIES, **bank.CONTRAST_CATEGORIES}:
                self.assertGreaterEqual(len(families[split][category]), 1, (split, category))
        pools = [t for split in families.values() for _, t in split["prefix"]]
        self.assertFalse(set(pools) & set(bank.V2_HELDOUT_PREFIXES))


@unittest.skipUnless(AVAILABLE and REACT_V2.exists(), "generate react-v3 data first")
class GeneratedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent.ocean import read_jsonl
        cls.read = staticmethod(read_jsonl)
        cls.tasks = {d: {s: read_jsonl(d / f"ocean_agent_{s}.jsonl") for s in ("train", "val", "test")} for d in AVAILABLE}
        cls.manifests = {d: json.loads((d / "agent_manifest.json").read_text()) for d in AVAILABLE}

    def test_trainer_manifest_contract(self):
        from agent.ocean import file_hash
        for directory, manifest in self.manifests.items():
            self.assertEqual(manifest["version"], "ocean-react-v2")
            self.assertEqual(manifest["harness_sha256"], file_hash(ROOT / "agent/react.py"))
            self.assertEqual(manifest["tool_sha256"], file_hash(ROOT / "agent/ocean.py"))
            for split in manifest["splits"].values():
                for name, digest in split["artifacts"].items():
                    self.assertEqual(file_hash(directory / name), digest, name)

    @unittest.skipUnless(DATA["react-v3"] in AVAILABLE and DATA["react-v3-slots"] in AVAILABLE, "needs both v3 variants")
    def test_variants_share_tasks_and_differ_only_in_plans(self):
        fixed, slots = self.tasks[DATA["react-v3"]], self.tasks[DATA["react-v3-slots"]]
        for split in ("val", "test", "train"):
            self.assertEqual(fixed[split], slots[split])
        demos = [self.read(DATA[name] / "ocean_agent_sft_val.jsonl") for name in ("react-v3", "react-v3-slots")]
        strip = lambda demo: [re.sub(r"<plan>[^<>]*</plan>", "", m["content"]) for m in demo["conversations"][1:]]
        self.assertEqual(*[[strip(d) for d in variant] for variant in demos])

    def test_counts_and_balance(self):
        for directory, manifest in self.manifests.items():
            for split, sizes in RECIPES[manifest["recipe"]]["sizes"].items():
                expected = Counter()
                for job, count in sizes.items():
                    expected[JOB_KIND.get(job, job)] += count
                counts = Counter(t["kind"] for t in self.tasks[directory][split] if "replay" not in t)
                self.assertEqual(counts, expected, (directory.name, split))
            replayed = Counter(t["kind"] for t in self.tasks[directory]["train"] if t.get("replay"))
            self.assertEqual(replayed, Counter(V1_REPLAY))
            operations = manifest["splits"]["train"]["calculator_operations"]
            self.assertLess(max(operations[o] for o in ("distance", "speed", "time")) /
                            min(operations[o] for o in ("distance", "speed", "time")), 1.5, operations)
            conversions = manifest["splits"]["train"]["conversions"]
            self.assertEqual({tuple(k.split("->")) for k in conversions}, set(CONVERT_PAIRS))

    def test_val_test_are_held_out(self):
        for directory, manifest in self.manifests.items():
            previous = manifest.get("held_out_benchmarks") or [manifest["comparable_eval_sets"]["react_v2"]]
            earlier = [self.read(ROOT / p / f"ocean_agent_{s}.jsonl") for p in previous if "react-" in p for s in ("val", "test")]
            old_sources = {t.get("source_id") for tasks in earlier for t in tasks} - {None}
            old_calcs = {json.dumps(c, sort_keys=True) for tasks in earlier for t in tasks for c in t.get("required_calcs", [])}
            data = self.tasks[directory]
            train_families = {f for t in data["train"] for f in t.get("families", [])}
            keys = {}
            for split, tasks in data.items():
                for task in tasks:
                    if task.get("replay"):
                        continue
                    if split != "train":  # English card references are shared vocabulary, like unit words
                        self.assertFalse(set(task["families"]) - {"card_ref.en"} & train_families, task["id"])
                    self.assertFalse(task.get("source_id") in old_sources, task["id"])
                    for calc in task.get("required_calcs", []):
                        self.assertFalse(json.dumps(calc, sort_keys=True) in old_calcs, task["id"])
                    if "parameter_key" in task:
                        self.assertEqual(keys.setdefault(task["parameter_key"], split), split)
            sources = {s: {t.get("source_id") for t in tasks} - {None} for s, tasks in data.items()}
            self.assertFalse(sources["train"] & sources["val"] or sources["train"] & sources["test"]
                             or sources["val"] & sources["test"])

    def test_prefix_is_not_a_kind_shortcut(self):
        for directory in AVAILABLE:
            kinds_by_prefix = {}
            for task in self.tasks[directory]["train"]:
                for family in task.get("families", []):
                    if family.startswith("prefix#"):
                        kinds_by_prefix.setdefault(family, Counter())[task["kind"]] += 1
            for family, kinds in kinds_by_prefix.items():
                if sum(kinds.values()) >= 60:
                    self.assertEqual(len(kinds), 6, (directory.name, family, kinds))

    def test_demonstrations_replay_through_harness(self):
        from transformers import AutoTokenizer
        from agent.react import ReactSFTDataset, ReactTools, run_trajectory
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
        for directory, manifest in self.manifests.items():
            env = ReactTools(self.read(directory / "ocean_agent_corpus_val.jsonl"))
            seen = set()
            for task in self.tasks[directory]["val"]:
                key = (task["kind"], task["subtype"].split(":")[0])
                if key in seen:
                    continue
                seen.add(key)
                texts = iter(oracle(task, manifest["plan_style"]))
                generate = lambda ids, limit: (tokenizer.encode(next(texts), add_special_tokens=False) + [tokenizer.eos_token_id], [])
                self.assertEqual(run_trajectory(task, tokenizer, env, generate)["metrics"]["success"], 1, task["id"])
            dataset = ReactSFTDataset(directory / "ocean_agent_sft_train.jsonl", tokenizer, 2048)
            for index in range(0, len(dataset), len(dataset) // 40):
                ids, labels = dataset[index]  # raises on any SFT/inference prefix mismatch
                self.assertTrue((labels != -100).any())


if __name__ == "__main__":
    unittest.main()
