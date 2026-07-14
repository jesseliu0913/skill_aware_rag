#!/usr/bin/env python
"""Offline unit tests for Exp07 generalization splits (stdlib unittest).

Builds tiny synthetic records and asserts:
  - drug_holdout yields DISJOINT drug/molecule identity sets across train/test;
  - skill_holdout removes the target skill from train ENTIRELY;
  - source_holdout removes the target source from train entirely;
  - determinism (same seed -> same partition).

No KB / model / GPU access. Run:
  python -m unittest experiments/exp07_generalization/test_make_splits.py
"""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Allow `python -m unittest experiments/exp07_generalization/test_make_splits.py`
# from the repo root as well as running from within this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_splits as ms  # noqa: E402


def fda_rec(rid, drug, task_type="factual"):
    return {
        "id": rid,
        "source": "fdarxbench",
        "drug_name": drug,
        "task_type": task_type,
        "question": f"What is the indication of {drug}?",
        "input_type": "FDA label context",
        "gold_answer": "x",
    }


def mol_rec(rid, smiles, source="molinst_property", task_type="property prediction", question="What is the molecular weight?"):
    return {
        "id": rid,
        "source": source,
        "decoded_smiles": smiles,
        "task_type": task_type,
        "question": question,
        "input_molecule_or_context": smiles,
        "gold_answer": "y",
    }


class TestDrugHoldout(unittest.TestCase):
    def setUp(self):
        # 5 distinct drugs, some with multiple records each.
        self.records = [
            fda_rec("a1", "Aspirin"),
            fda_rec("a2", "Aspirin", task_type="multihop"),
            fda_rec("b1", "Ibuprofen"),
            fda_rec("c1", "Metformin"),
            fda_rec("d1", "Warfarin"),
            fda_rec("e1", "Lisinopril"),
        ]

    def test_drug_sets_disjoint(self):
        train, test, key_fn = ms.split_drug_holdout(self.records, test_frac=0.4, seed=7)
        train_keys = ms.key_set(train, key_fn)
        test_keys = ms.key_set(test, key_fn)
        self.assertGreater(len(test_keys), 0)
        self.assertEqual(train_keys & test_keys, set())  # DISJOINT drug sets

    def test_no_record_lost(self):
        train, test, _ = ms.split_drug_holdout(self.records, test_frac=0.4, seed=7)
        self.assertEqual(len(train) + len(test), len(self.records))

    def test_all_records_of_heldout_drug_go_to_test(self):
        # Aspirin has two records; both must land on the same side.
        train, test, _ = ms.split_drug_holdout(self.records, test_frac=1.0, seed=1)
        self.assertEqual(len(train), 0)
        self.assertEqual(len(test), len(self.records))

    def test_deterministic(self):
        a = ms.choose_drug_holdout_keys(self.records, 0.4, seed=99)
        b = ms.choose_drug_holdout_keys(self.records, 0.4, seed=99)
        c = ms.choose_drug_holdout_keys(self.records, 0.4, seed=100)
        self.assertEqual(a, b)
        # Different seed generally reshuffles; at minimum the API is stable.
        self.assertIsInstance(c, set)

    def test_mol_identity_fallback(self):
        recs = [mol_rec("m1", "CCO"), mol_rec("m2", "c1ccccc1"), mol_rec("m3", "CCN")]
        train, test, key_fn = ms.split_drug_holdout(recs, test_frac=0.34, seed=3)
        self.assertEqual(ms.key_set(train, key_fn) & ms.key_set(test, key_fn), set())


class TestSkillHoldout(unittest.TestCase):
    def setUp(self):
        self.records = [
            fda_rec("f1", "Aspirin", task_type="factual"),       # fda_label_factual
            fda_rec("f2", "Ibuprofen", task_type="multihop"),    # fda_label_multihop
            mol_rec("p1", "CCO"),                                  # molecule_property_numeric
            mol_rec("p2", "CCN"),                                  # molecule_property_numeric
            mol_rec("d1", "CCC", source="molinst_description_guided_design",
                    task_type="description-guided molecule design",
                    question="Synthesize a molecule that ..."),   # molecule_design
        ]

    def test_target_skill_removed_from_train(self):
        target = "molecule_property_numeric"
        train, test, _ = ms.split_skill_holdout(self.records, target)
        train_skills = {ms.skill_key(r) for r in train}
        self.assertNotIn(target, train_skills)  # removed ENTIRELY from train
        # And every held-out record actually has that skill.
        self.assertTrue(all(ms.skill_key(r) == target for r in test))
        self.assertEqual(len(test), 2)

    def test_other_skills_stay_in_train(self):
        train, _, _ = ms.split_skill_holdout(self.records, "molecule_property_numeric")
        train_skills = {ms.skill_key(r) for r in train}
        self.assertIn("fda_label_factual", train_skills)

    def test_holdout_absent_skill_leaves_train_full(self):
        train, test, _ = ms.split_skill_holdout(self.records, "biomedical_open_qa")
        self.assertEqual(len(test), 0)
        self.assertEqual(len(train), len(self.records))


class TestSourceHoldout(unittest.TestCase):
    def test_source_removed_from_train(self):
        records = [
            fda_rec("f1", "Aspirin"),
            mol_rec("m1", "CCO", source="molinst_property"),
            mol_rec("m2", "CCN", source="molinst_open_qa", task_type="Open Question", question="explain"),
        ]
        train, test, key_fn = ms.split_source_holdout(records, "fdarxbench")
        self.assertNotIn("fdarxbench", {ms.source_key(r) for r in train})
        self.assertEqual(len(test), 1)
        self.assertEqual(ms.key_set(train, key_fn) & {"fdarxbench"}, set())


class TestEndToEnd(unittest.TestCase):
    def test_run_writes_files_and_proof(self):
        records = [fda_rec(f"r{i}", d) for i, d in enumerate(["A", "B", "C", "D", "E"])]
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.jsonl"
            inp.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            out = Path(tmp) / "out"
            args = argparse.Namespace(
                input=str(inp), output_dir=str(out), mode="drug_holdout",
                seed=5, test_frac=0.4, holdout_skill="", holdout_source="",
            )
            summary = ms.run(args)
            self.assertTrue((out / "train.jsonl").exists())
            self.assertTrue((out / "test.jsonl").exists())
            self.assertTrue((out / "split_summary.json").exists())
            self.assertTrue(summary["disjoint"])
            self.assertEqual(summary["key_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
