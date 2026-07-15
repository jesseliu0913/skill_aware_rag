#!/usr/bin/env python
"""Offline stdlib unittests for Exp06 skill-override controls + router report.

No KB, no model, no GPU, no network. Run:

  python -m unittest experiments.exp06_skill_controls.test_skill_controls -v
  # or, from repo root:
  python experiments/exp06_skill_controls/test_skill_controls.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import augment_with_skill_kb as aug  # noqa: E402
from router_report import build_confusion  # noqa: E402


def fda_record(rid: str = "fda_1") -> dict:
    return {
        "id": rid,
        "source": "fdarxbench",
        "input_type": "FDA label context",
        "task_type": "factual",
        "question": "What is the recommended dosage of the drug?",
        "input_molecule_or_context": "DOSAGE: 10 mg once daily.",
    }


def mol_property_record(rid: str = "mol_1") -> dict:
    return {
        "id": rid,
        "source": "molinst_property",
        "input_type": "SELFIES",
        "task_type": "property prediction",
        "question": "Predict the logP value of this molecule.",
        "decoded_smiles": "CCO",
    }


class TestOverrideDeterminism(unittest.TestCase):
    def test_random_stable_per_id(self):
        rec = mol_property_record("stable_id_42")
        first = aug.random_skill(rec, "v1")
        for _ in range(20):
            self.assertEqual(aug.random_skill(rec, "v1"), first)

    def test_random_differs_by_id(self):
        # sha1 over id: at least some ids must land on different skills.
        skills = {aug.random_skill({"id": f"id_{i}"}, "v1") for i in range(50)}
        self.assertGreater(len(skills), 1)

    def test_random_matches_manual_sha1(self):
        import hashlib
        rec = {"id": "check_hash"}
        names = aug.skill_names("v1")
        expected = names[int(hashlib.sha1(b"check_hash").hexdigest(), 16) % len(names)]
        self.assertEqual(aug.random_skill(rec, "v1"), expected)

    def test_random_only_valid_skills(self):
        for i in range(30):
            self.assertIn(aug.random_skill({"id": f"r{i}"}, "v1"), aug.skill_names("v1"))

    def test_wrong_maps_consistently_and_never_identity(self):
        for rec in (fda_record(), mol_property_record()):
            gold = aug.gold_field_skill(rec, "v1")
            wrong = aug.wrong_skill(rec, "v1")
            self.assertNotEqual(wrong, gold)
            # deterministic / stable
            self.assertEqual(wrong, aug.wrong_skill(rec, "v1"))
            self.assertIn(wrong, aug.skill_names("v1"))

    def test_wrong_confusion_map_total_and_no_identity(self):
        for gold, wrong in aug.CONFUSION_MAP_V1.items():
            self.assertNotEqual(gold, wrong)
            self.assertIn(wrong, aug.skill_names("v1"))
        # every v1 skill has an entry
        self.assertEqual(set(aug.CONFUSION_MAP_V1.keys()), set(aug.skill_names("v1")))

    def test_oracle_equals_gold_field(self):
        rec = fda_record()
        self.assertEqual(
            aug.infer_record_skill(rec, "v1", "oracle"),
            aug.gold_field_skill(rec, "v1"),
        )


class TestNoneIsByteIdentical(unittest.TestCase):
    def test_none_leaves_routed_skill_unchanged(self):
        for rec in (fda_record(), mol_property_record()):
            for schema in ("legacy", "v1"):
                baseline = (
                    aug.infer_schema_v1_skill(rec) if schema == "v1" else aug.infer_skill(rec)
                )
                self.assertEqual(aug.infer_record_skill(rec, schema, "none"), baseline)
                # default arg also == none
                self.assertEqual(aug.infer_record_skill(rec, schema), baseline)

    def test_augment_record_none_matches_no_override_arg(self):
        rec = mol_property_record()
        a = aug.augment_record(rec, [], "skill_hybrid", 3500, "v1")
        b = aug.augment_record(rec, [], "skill_hybrid", 3500, "v1", "none")
        self.assertEqual(a["skill_schema"]["skill"], b["skill_schema"]["skill"])
        self.assertEqual(a["prompt"], b["prompt"])


class TestPredictedQuestionOnly(unittest.TestCase):
    def test_uses_only_question_text(self):
        # Same question, but flip every gold field: prediction must not move.
        base = {
            "id": "q1",
            "question": "Predict the logP value of this molecule.",
            "source": "fdarxbench",
            "task_type": "factual",
            "input_type": "FDA label context",
            "decoded_smiles": "CCCC",
            "drug_name": "aspirin",
        }
        flipped = dict(base)
        flipped.update({
            "source": "molinst_open_qa",
            "task_type": "Open Question",
            "input_type": "none",
            "decoded_smiles": "",
            "drug_name": "",
        })
        self.assertEqual(
            aug.predict_skill_question_only(base, "v1"),
            aug.predict_skill_question_only(flipped, "v1"),
        )

    def test_gold_field_router_does_depend_on_source(self):
        # Contrast: the gold-field router DOES change when source/task_type flip,
        # proving predicted (above) is genuinely question-only.
        q = "What is this?"
        fda = {"id": "a", "question": q, "source": "fdarxbench", "task_type": "factual", "input_type": "FDA label context"}
        mol = {"id": "a", "question": q, "source": "molinst_property", "task_type": "property prediction"}
        self.assertNotEqual(
            aug.gold_field_skill(fda, "v1"),
            aug.gold_field_skill(mol, "v1"),
        )

    def test_predicted_returns_valid_skill(self):
        for rec in (fda_record(), mol_property_record(), {"id": "x", "question": "design a molecule that inhibits kinase"}):
            self.assertIn(aug.predict_skill_question_only(rec, "v1"), aug.skill_names("v1"))

    def test_predicted_cues(self):
        self.assertEqual(
            aug.predict_skill_question_only({"id": "d", "question": "Design a molecule with high solubility."}, "v1"),
            "molecule_design",
        )
        self.assertEqual(
            aug.predict_skill_question_only({"id": "p", "question": "What is the molecular weight?"}, "v1"),
            "molecule_property_numeric",
        )


class TestConfusionMatrix(unittest.TestCase):
    def test_counts_sum_to_n(self):
        records = [fda_record("f1"), fda_record("f2"), mol_property_record("m1")]
        confusion, n_total, n_correct = build_confusion(records, "v1")
        self.assertEqual(n_total, 3)
        self.assertEqual(sum(confusion.values()), 3)
        self.assertLessEqual(n_correct, n_total)

    def test_diagonal_matches_agreement(self):
        records = [mol_property_record("m1"), mol_property_record("m2")]
        confusion, n_total, n_correct = build_confusion(records, "v1")
        diagonal = sum(v for (g, p), v in confusion.items() if g == p)
        self.assertEqual(diagonal, n_correct)

    def test_perfect_agreement_when_pred_equals_gold(self):
        # A record whose question cue matches its gold field -> on-diagonal.
        rec = mol_property_record("agree")
        # property question + property gold field should agree
        confusion, _, n_correct = build_confusion([rec], "v1")
        gold = aug.gold_field_skill(rec, "v1")
        pred = aug.predict_skill_question_only(rec, "v1")
        if gold == pred:
            self.assertEqual(n_correct, 1)
            self.assertEqual(confusion.get((gold, gold)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
