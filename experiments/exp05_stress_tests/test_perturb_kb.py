#!/usr/bin/env python
"""Offline stdlib unittests for Exp05 perturb_kb (no GPU / no outputs/ needed)."""

import random
import unittest

import perturb_kb as pk


def tiny_facts():
    """A tiny inline KB with >=2 distinct sources (required for decoy relabel)."""
    return [
        {"id": "a1", "source": "primekg", "source_id": "p-1", "title": "T1",
         "text": "aspirin inhibits COX-1 enzyme", "aliases": ["aspirin"],
         "entities": ["aspirin", "COX-1"], "relation": "inhibits",
         "metadata": {"orig": 1}, "tokens": ["aspirin", "cox", "1", "enzyme", "inhibits"]},
        {"id": "b2", "source": "drugchat_chembl", "source_id": "c-2", "title": "T2",
         "text": "ibuprofen has molecular weight 206", "aliases": ["ibuprofen"],
         "entities": ["ibuprofen"], "relation": "property",
         "metadata": {}, "tokens": ["ibuprofen", "molecular", "weight", "206"]},
        {"id": "c3", "source": "fdarxbench_label", "source_id": "f-3", "title": "T3",
         "text": "metformin is indicated for type 2 diabetes", "aliases": ["metformin"],
         "entities": ["metformin"], "relation": "indication",
         "metadata": {"k": "v"}, "tokens": ["metformin", "indicated", "type", "2", "diabetes"]},
        {"id": "d4", "source": "drugchat_pubchem", "source_id": "u-4", "title": "T4",
         "text": "caffeine is a CNS stimulant", "aliases": ["caffeine"],
         "entities": ["caffeine"], "relation": "class",
         "metadata": {}, "tokens": ["caffeine", "cns", "stimulant"]},
    ]

FACT_KEYS = {"id", "source", "source_id", "title", "text", "aliases",
             "entities", "relation", "metadata", "tokens"}


class TestNoisy(unittest.TestCase):
    def test_adds_right_count_and_preserves_originals(self):
        facts = tiny_facts()
        out = pk.perturb(facts, "noisy", random.Random(7), noise_frac=2.0)
        # round(2.0 * 4) = 8 injected, originals preserved => 12 total.
        self.assertEqual(len(out), 12)
        # Originals kept verbatim, in order, at the front.
        self.assertEqual(out[:4], facts)
        added = out[4:]
        self.assertEqual(len(added), 8)
        for f in added:
            self.assertTrue(f["id"].startswith("noise_"))
            self.assertTrue(f["metadata"].get("synthetic_noise"))
            # Injected facts reuse an existing source.
            self.assertIn(f["source"], pk.kb_sources(facts))
            # Full schema preserved.
            self.assertEqual(set(f.keys()), FACT_KEYS)

    def test_zero_frac_adds_nothing(self):
        facts = tiny_facts()
        out = pk.perturb(facts, "noisy", random.Random(1), noise_frac=0.0)
        self.assertEqual(out, facts)


class TestDecoy(unittest.TestCase):
    def test_same_text_wrong_source_flagged(self):
        facts = tiny_facts()
        out = pk.perturb(facts, "decoy", random.Random(3), decoy_per=1, decoy_frac=1.0)
        self.assertEqual(len(out), 8)  # 4 originals + 4 decoys
        self.assertEqual(out[:4], facts)
        decoys = out[4:]
        self.assertEqual(len(decoys), 4)
        orig_by_id = {f["id"]: f for f in facts}
        for d in decoys:
            self.assertTrue(d["metadata"].get("decoy") is True)
            orig = orig_by_id[d["metadata"]["decoy_of"]]
            # Near-duplicate: original text preserved as a prefix.
            self.assertTrue(d["text"].startswith(orig["text"]))
            # Relabeled to a DIFFERENT source.
            self.assertNotEqual(d["source"], orig["source"])
            self.assertIn(d["source"], pk.kb_sources(facts))
            self.assertEqual(set(d.keys()), FACT_KEYS)
            # Tokens recomputed to match the perturbed text.
            self.assertEqual(d["tokens"], pk.tokenize(d["text"]))

    def test_decoy_per_multiplies(self):
        facts = tiny_facts()
        out = pk.perturb(facts, "decoy", random.Random(5), decoy_per=3, decoy_frac=1.0)
        self.assertEqual(len(out) - len(facts), 12)  # 4 * 3


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        facts = tiny_facts()
        a = pk.perturb(facts, "noisy", random.Random(42), noise_frac=1.5)
        b = pk.perturb(facts, "noisy", random.Random(42), noise_frac=1.5)
        self.assertEqual(a, b)
        c = pk.perturb(facts, "decoy", random.Random(42), decoy_per=2)
        d = pk.perturb(facts, "decoy", random.Random(42), decoy_per=2)
        self.assertEqual(c, d)

    def test_different_seed_differs(self):
        facts = tiny_facts()
        a = pk.perturb(facts, "noisy", random.Random(1), noise_frac=2.0)
        b = pk.perturb(facts, "noisy", random.Random(2), noise_frac=2.0)
        self.assertNotEqual(a[4:], b[4:])


if __name__ == "__main__":
    unittest.main()
