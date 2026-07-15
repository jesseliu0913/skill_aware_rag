#!/usr/bin/env python
"""Offline stdlib unit tests for the Exp08 necessity-probe scaffold.

Feeds a tiny in-memory KB (facts across two sources that share an alias) and
asserts a collision candidate is produced with distinct A/B sources and a
well-formed normalized record. Requires no `outputs/` artifact.

Run:  python -m unittest experiments/exp08_necessity_benchmark/test_build_necessity_probe.py
"""

import unittest

import build_necessity_probe as bp


# Two sources share the alias "aspirin" under different relations: the gold
# source (fdarxbench_label) answers the label question; the decoy (primekg)
# mentions aspirin under a different relation and is NOT in the intended skill's
# allowed sources -> a discriminative pair.
TINY_KB = [
    {
        "id": "fda_1",
        "source": "fdarxbench_label",
        "relation": "label_context",
        "title": "Aspirin label",
        "text": "Aspirin label warns of gastrointestinal bleeding risk in elderly patients.",
        "aliases": ["Aspirin"],
        "entities": ["aspirin"],
        "metadata": {"object": "gastrointestinal bleeding risk"},
    },
    {
        "id": "pkg_1",
        "source": "primekg",
        "relation": "drug_effect",
        "title": "Aspirin effect",
        "text": "Aspirin is associated with reduced platelet aggregation in patients.",
        "aliases": ["aspirin"],
        "entities": ["aspirin"],
        "metadata": {"object": "reduced platelet aggregation"},
    },
    # A lone fact with no cross-source collision (should not yield a candidate).
    {
        "id": "chembl_1",
        "source": "drugchat_chembl",
        "relation": "instruction_fact",
        "title": "Ibuprofen",
        "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug.",
        "aliases": ["ibuprofen"],
        "entities": ["ibuprofen"],
        "metadata": {},
    },
]

REQUIRED_KEYS = {
    "id", "source", "question", "input_molecule_or_context", "input_type",
    "gold_answer", "task_type", "metadata", "prompt", "messages",
}


class TestCollisionIndex(unittest.TestCase):
    def test_only_cross_source_aliases_kept(self):
        idx = bp.build_collision_index(TINY_KB)
        self.assertIn("aspirin", idx)
        self.assertNotIn("ibuprofen", idx)  # single-source, not a collision
        self.assertGreaterEqual(len(idx["aspirin"]), 2)
        self.assertIn("fdarxbench_label", idx["aspirin"])
        self.assertIn("primekg", idx["aspirin"])

    def test_jaccard_bounds(self):
        self.assertEqual(bp.jaccard(set(), {"a"}), 0.0)
        self.assertEqual(bp.jaccard({"a", "b"}, {"a", "b"}), 1.0)
        self.assertAlmostEqual(bp.jaccard({"a", "b"}, {"b", "c"}), 1 / 3)


class TestBuildProbe(unittest.TestCase):
    def setUp(self):
        self.candidates, self.provenance, self.stats = bp.build_probe(
            TINY_KB, min_sim=0.0, limit=0
        )

    def test_at_least_one_candidate(self):
        self.assertGreaterEqual(len(self.candidates), 1)
        self.assertEqual(len(self.candidates), len(self.provenance))

    def test_distinct_ab_sources(self):
        prov = self.provenance[0]
        self.assertEqual(prov["gold_source"], "fdarxbench_label")
        self.assertEqual(prov["decoy_source"], "primekg")
        self.assertNotEqual(prov["gold_source"], prov["decoy_source"])

    def test_discriminative_pair(self):
        # primekg is not in fda_label_factual's allowed sources -> schema excludes it.
        self.assertTrue(self.provenance[0]["decoy_excluded_by_intended_skill"])
        self.assertTrue(self.provenance[0]["distinct_relation"])

    def test_record_schema_well_formed(self):
        rec = self.candidates[0]
        self.assertTrue(REQUIRED_KEYS.issubset(rec.keys()))
        self.assertEqual(rec["source"], "necessity_probe")
        self.assertEqual(rec["task_type"], "necessity_probe")
        # R1: answer must not live in the input.
        self.assertEqual(rec["input_molecule_or_context"], "")
        self.assertEqual(rec["input_type"], "none")
        # messages mirror prompt/gold_answer for the pipeline.
        self.assertEqual(rec["messages"][0]["content"], rec["prompt"])
        self.assertEqual(rec["messages"][1]["content"], rec["gold_answer"])
        self.assertIn("aspirin", rec["question"].lower())

    def test_gold_is_provisional_not_fabricated(self):
        rec = self.candidates[0]
        val = rec["metadata"]["validation"]
        self.assertFalse(val["validated"])
        self.assertTrue(val["gold_answer_is_provisional"])
        self.assertEqual(self.stats["validated"], 0)

    def test_provenance_kept_separate(self):
        # A/B provenance must not be embedded as retrievable evidence in the record.
        rec = self.candidates[0]
        self.assertEqual(rec["retrieved_kg_evidence"], [])
        self.assertEqual(rec["retrieved_molecule_evidence"], [])

    def test_limit_caps_output(self):
        cands, _, _ = bp.build_probe(TINY_KB, min_sim=0.0, limit=1)
        self.assertLessEqual(len(cands), 1)

    def test_require_discriminative_filters_same_allowed_source(self):
        # If the only decoy were in the allowed set, discriminative mode drops it.
        kb = [
            dict(TINY_KB[0]),
            {
                "id": "fda_2",
                "source": "fdarxbench_label",  # same source as gold -> not a cross-source decoy
                "relation": "label_context",
                "text": "Aspirin label notes dosing for aspirin in adults.",
                "aliases": ["aspirin"],
                "entities": ["aspirin"],
                "metadata": {},
            },
        ]
        cands, _, _ = bp.build_probe(kb, min_sim=0.0)
        self.assertEqual(len(cands), 0)  # no distinct-source decoy available


if __name__ == "__main__":
    unittest.main()
