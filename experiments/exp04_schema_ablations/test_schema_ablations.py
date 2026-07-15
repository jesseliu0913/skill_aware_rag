#!/usr/bin/env python
"""Offline unit tests for the Exp04 (RQ5) schema-component ablations.

These exercise the additive `--ablate` path added to
`retrieval/augment_with_skill_kb.py`. No real KB / GPU is required: prompt
behavior is asserted at the builder-function level, and source routing is
checked against a tiny in-memory KB.

Run:  python -m unittest experiments/exp04_schema_ablations/test_schema_ablations.py
"""

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "retrieval"))

import augment_with_skill_kb as m  # noqa: E402


FDA_RECORD = {
    "id": "f1",
    "source": "fdarxbench",
    "input_type": "FDA label context",
    "task_type": "factual",
    "question": "What is the recommended dose?",
    "input_molecule_or_context": "DOSAGE: 10 mg once daily.",
    "drug_name": "acetaminophen",
    "gold_answer": "10 mg once daily",
}

EVIDENCE = [
    {
        "id": "e1", "score": 9.1, "source": "fdarxbench_label", "relation": "label_context",
        "title": "t", "text": "Dose is 10 mg once daily.", "entities": [], "metadata": {},
    },
    {
        "id": "e2", "score": 3.2, "source": "drugchat_pubchem", "relation": "instruction_fact",
        "title": "t2", "text": "Ethanol logP -0.14.", "entities": [], "metadata": {},
    },
]


def build_prompt(ablate):
    skill = m.infer_record_skill(FDA_RECORD, "v1")
    return m.build_completed_skill_prompt(FDA_RECORD, skill, "skill_hybrid", EVIDENCE, 3500, ablate)


def build_mini_kb(facts):
    """Reproduce load_kb's idf/token_index over an in-memory fact list."""
    doc_freq = {}
    token_index = {}
    for idx, fact in enumerate(facts):
        for token in set(fact["tokens"]):
            doc_freq[token] = doc_freq.get(token, 0) + 1
            token_index.setdefault(token, []).append(idx)
    n_docs = max(len(facts), 1)
    idf = {t: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0) for t, df in doc_freq.items()}
    fact_id_to_idx = {f["id"]: i for i, f in enumerate(facts)}
    return token_index, idf, fact_id_to_idx


MINI_FACTS = [
    {"id": "a", "source": "fdarxbench_label", "relation": "label_context", "title": "",
     "text": "dose ten daily", "tokens": ["dose", "ten", "daily"], "entities": [], "metadata": {}},
    {"id": "b", "source": "primekg", "relation": "drug_effect", "title": "",
     "text": "dose ten daily", "tokens": ["dose", "ten", "daily"], "entities": [], "metadata": {}},
]


class TestParseAblate(unittest.TestCase):
    def test_empty_and_none_are_full_schema(self):
        self.assertEqual(m.parse_ablate(""), set())
        self.assertEqual(m.parse_ablate(None), set())

    def test_comma_separated(self):
        self.assertEqual(m.parse_ablate("skill_id,evidence_org"), {"skill_id", "evidence_org"})

    def test_whitespace_and_iterable(self):
        self.assertEqual(m.parse_ablate(" skill_id , answer_format "), {"skill_id", "answer_format"})
        self.assertEqual(m.parse_ablate({"solving_steps"}), {"solving_steps"})

    def test_unknown_component_raises(self):
        with self.assertRaises(ValueError):
            m.parse_ablate("skill_id,bogus")


class TestPromptDefaultUnchanged(unittest.TestCase):
    def test_none_equals_empty_set(self):
        self.assertEqual(build_prompt(None), build_prompt(set()))

    def test_full_schema_has_every_component(self):
        p = build_prompt(None)
        self.assertIn("\nSkill: fda_label_factual", p)
        self.assertIn("How to solve:", p)
        self.assertIn("Answer format: short_label_grounded_answer", p)
        self.assertIn("Skill-based retrieved knowledge:", p)
        self.assertNotIn("Retrieved KB evidence:", p)


class TestPromptAblations(unittest.TestCase):
    def setUp(self):
        self.full = build_prompt(None)
        self.full_lines = self.full.split("\n")

    def _removed_lines(self, ablate):
        """Lines present in the full prompt but absent after ablation."""
        ablated = build_prompt(ablate).split("\n")
        # multiset difference preserving what disappeared
        remaining = list(ablated)
        removed = []
        for line in self.full_lines:
            if line in remaining:
                remaining.remove(line)
            else:
                removed.append(line)
        return removed, ablated

    def test_ablate_skill_id_removes_only_skill_line(self):
        removed, ablated = self._removed_lines({"skill_id"})
        self.assertEqual(removed, ["Skill: fda_label_factual"])
        self.assertIn("How to solve:", ablated)
        self.assertIn("Answer format: short_label_grounded_answer", ablated)
        self.assertIn("Skill-based retrieved knowledge:", ablated)

    def test_ablate_solving_steps_removes_only_steps_block(self):
        removed, ablated = self._removed_lines({"solving_steps"})
        self.assertIn("How to solve:", removed)
        self.assertTrue(all(r == "How to solve:" or r.strip()[:1].isdigit() for r in removed),
                        f"unexpected removals: {removed}")
        self.assertNotIn("How to solve:", ablated)
        self.assertIn("Skill: fda_label_factual", ablated)
        self.assertIn("Answer format: short_label_grounded_answer", ablated)

    def test_ablate_answer_format_removes_only_answer_format(self):
        removed, ablated = self._removed_lines({"answer_format"})
        self.assertEqual(removed, ["Answer format: short_label_grounded_answer"])
        self.assertIn("Skill: fda_label_factual", ablated)
        self.assertIn("How to solve:", ablated)

    def test_ablate_evidence_org_swaps_header_only(self):
        ablated = build_prompt({"evidence_org"})
        self.assertNotIn("Skill-based retrieved knowledge:", ablated)
        self.assertIn("Retrieved KB evidence:", ablated)
        # the evidence content itself (the dump) is unchanged
        self.assertIn("Dose is 10 mg once daily.", ablated)

    def test_ablate_everything_looks_like_uniform_baseline(self):
        ablated = build_prompt({"skill_id", "source_routing", "solving_steps", "answer_format", "evidence_org"})
        self.assertNotIn("Skill:", ablated)
        self.assertNotIn("How to solve:", ablated)
        self.assertNotIn("Answer format:", ablated)
        self.assertNotIn("Skill-based retrieved knowledge:", ablated)
        self.assertIn("Retrieved KB evidence:", ablated)


class TestSourceRouting(unittest.TestCase):
    def _retrieve(self, ablate):
        token_index, idf, fact_id_to_idx = build_mini_kb(MINI_FACTS)
        return m.retrieve(
            FDA_RECORD, MINI_FACTS, fact_id_to_idx, {}, idf, token_index,
            "skill_hybrid", 8, 20000, "v1", ablate,
        )

    def test_default_routing_filters_to_allowed_sources(self):
        ev = self._retrieve(None)
        sources = {e["source"] for e in ev}
        self.assertEqual(sources, {"fdarxbench_label"})

    def test_ablating_source_routing_retrieves_unconstrained(self):
        ev = self._retrieve({"source_routing"})
        sources = {e["source"] for e in ev}
        self.assertIn("primekg", sources)
        self.assertGreater(len(ev), 1)


class TestAugmentRecord(unittest.TestCase):
    def test_default_record_has_no_ablate_key(self):
        rec = m.augment_record(FDA_RECORD, EVIDENCE, "skill_hybrid", 3500, "v1")
        self.assertNotIn("ablate", rec["skill_schema"])

    def test_ablated_record_records_component_set(self):
        rec = m.augment_record(FDA_RECORD, EVIDENCE, "skill_hybrid", 3500, "v1", {"skill_id", "answer_format"})
        self.assertEqual(rec["skill_schema"]["ablate"], ["answer_format", "skill_id"])
        self.assertNotIn("Skill:", rec["completed_skill_prompt"])
        self.assertNotIn("Answer format:", rec["completed_skill_prompt"])


if __name__ == "__main__":
    unittest.main()
