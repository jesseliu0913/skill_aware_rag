#!/usr/bin/env python
"""Unit tests for Exp01 heterogeneity metrics."""

import math
import unittest

import source_heterogeneity as sh


class TestHeterogeneity(unittest.TestCase):
    def test_entropy_single_source_is_zero(self):
        dist = {"primekg": 1.0, "drugchat_chembl": 0.0, "drugchat_pubchem": 0.0, "fdarxbench_label": 0.0}
        self.assertAlmostEqual(sh.entropy_bits(dist), 0.0)
        self.assertAlmostEqual(sh.effective_sources(dist), 1.0)

    def test_entropy_uniform_is_log2_n(self):
        dist = {s: 0.25 for s in sh.KB_SOURCES}
        self.assertAlmostEqual(sh.entropy_bits(dist), math.log2(4))
        self.assertAlmostEqual(sh.effective_sources(dist), 4.0)

    def test_js_divergence_identical_is_zero(self):
        d = {s: 0.25 for s in sh.KB_SOURCES}
        self.assertAlmostEqual(sh.js_divergence(d, d), 0.0)

    def test_js_divergence_disjoint_is_one(self):
        p = {"primekg": 1.0, "drugchat_chembl": 0.0, "drugchat_pubchem": 0.0, "fdarxbench_label": 0.0}
        q = {"primekg": 0.0, "drugchat_chembl": 1.0, "drugchat_pubchem": 0.0, "fdarxbench_label": 0.0}
        self.assertAlmostEqual(sh.js_divergence(p, q), 1.0)

    def test_source_distribution_normalizes(self):
        rows = [
            {"dataset": "d", "question_type": "q", "method": "m", "source": "primekg", "mean_within_prompt_share": "0.5"},
            {"dataset": "d", "question_type": "q", "method": "m", "source": "fdarxbench_label", "mean_within_prompt_share": "1.5"},
        ]
        dist = sh.source_distribution(rows)
        sd = dist[("d", "q", "m")]
        self.assertAlmostEqual(sum(sd.values()), 1.0)
        self.assertAlmostEqual(sd["primekg"], 0.25)
        self.assertAlmostEqual(sd["fdarxbench_label"], 0.75)

    def test_allowed_sources_maps_fda_factual_to_label(self):
        self.assertEqual(sh.allowed_sources("fda_factual"), {"fdarxbench_label"})

    def test_allowed_sources_unknown_defaults_all(self):
        self.assertEqual(sh.allowed_sources("nonsense_type"), set(sh.KB_SOURCES))


if __name__ == "__main__":
    unittest.main()
