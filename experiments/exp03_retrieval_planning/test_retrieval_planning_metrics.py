#!/usr/bin/env python
"""Offline unit tests for Exp03 retrieval-planning metrics (stdlib unittest).

Builds tiny synthetic records inline and asserts the metric math directly. No KB,
no GPU, no outputs/ file required.
"""

import math
import re
import unittest

import retrieval_planning_metrics as rpm


# An FDA factual record → gold-skill fda_label_factual → allowed sources
# {"fdarxbench_label"}. Handy to know the allowed set for assertions.
def fda_record(evidence, gold_answer="warning bleeding risk"):
    return {
        "id": "r1",
        "source": "fdarxbench",
        "task_type": "factual",
        "gold_answer": gold_answer,
        "retrieved_kb_evidence": evidence,
    }


def ev(source, text="", title=""):
    return {"source": source, "text": text, "title": title}


class TestHelpers(unittest.TestCase):
    def test_allowed_sources_fda_factual(self):
        rec = fda_record([])
        self.assertEqual(rpm.allowed_sources(rec), {"fdarxbench_label"})

    def test_entropy_single_source_zero(self):
        self.assertAlmostEqual(rpm.entropy_bits({"fdarxbench_label": 4}), 0.0)

    def test_entropy_two_uniform_is_one_bit(self):
        self.assertAlmostEqual(rpm.entropy_bits({"a": 2, "b": 2}), 1.0)

    def test_entropy_four_uniform_is_two_bits(self):
        self.assertAlmostEqual(
            rpm.entropy_bits({"a": 1, "b": 1, "c": 1, "d": 1}), math.log2(4)
        )

    def test_jaccard_identical_is_one(self):
        self.assertAlmostEqual(rpm.jaccard({"x", "y"}, {"x", "y"}), 1.0)

    def test_jaccard_disjoint_is_zero(self):
        self.assertAlmostEqual(rpm.jaccard({"x"}, {"y"}), 0.0)

    def test_method_from_filename_default_regex(self):
        pat = re.compile(rpm.DEFAULT_METHOD_REGEX)
        self.assertEqual(rpm.method_from_filename("test_bm25_v1.jsonl", pat), "bm25")
        # longest-first: skillrouted variant beats plain bm25 / dense_bge
        self.assertEqual(
            rpm.method_from_filename("x_bm25_skillrouted_test.jsonl", pat),
            "bm25_skillrouted",
        )
        self.assertEqual(
            rpm.method_from_filename("aug_dense_bge_skillrouted.jsonl", pat),
            "dense_bge_skillrouted",
        )

    def test_method_from_filename_custom_regex(self):
        pat = re.compile(r"__(?P<method>[a-z]+)__")
        self.assertEqual(
            rpm.method_from_filename("data__foo__test.jsonl", pat), "foo"
        )

    def test_method_family(self):
        self.assertEqual(rpm.method_family("bm25"), "uniform")
        self.assertEqual(rpm.method_family("dense_bge"), "uniform")
        self.assertEqual(rpm.method_family("skill_schema_v1"), "skill")
        self.assertEqual(rpm.method_family("bm25_skillrouted"), "skill")


class TestRecordMetrics(unittest.TestCase):
    def test_all_on_source_precision_one_entropy_zero(self):
        rec = fda_record([ev("fdarxbench_label"), ev("fdarxbench_label")])
        m = rpm.record_metrics(rec, None)
        self.assertAlmostEqual(m["source_precision_at_k"], 1.0)
        self.assertAlmostEqual(m["wrong_source_rate"], 0.0)
        self.assertAlmostEqual(m["source_entropy_bits"], 0.0)
        # retrieved sources {fdarxbench_label} == allowed → Jaccard 1
        self.assertAlmostEqual(m["skill_source_agreement"], 1.0)

    def test_all_off_source_precision_zero(self):
        rec = fda_record([ev("primekg"), ev("drugchat_chembl")])
        m = rpm.record_metrics(rec, None)
        self.assertAlmostEqual(m["source_precision_at_k"], 0.0)
        self.assertAlmostEqual(m["wrong_source_rate"], 1.0)
        # retrieved {primekg, drugchat_chembl} vs allowed {fdarxbench_label} → 0
        self.assertAlmostEqual(m["skill_source_agreement"], 0.0)
        # two distinct sources, uniform → 1 bit
        self.assertAlmostEqual(m["source_entropy_bits"], 1.0)

    def test_half_on_source_precision_half(self):
        rec = fda_record([ev("fdarxbench_label"), ev("primekg")])
        m = rpm.record_metrics(rec, None)
        self.assertAlmostEqual(m["source_precision_at_k"], 0.5)
        self.assertAlmostEqual(m["wrong_source_rate"], 0.5)
        # retrieved {fdarxbench_label, primekg} vs allowed {fdarxbench_label}:
        # intersection 1, union 2 → 0.5
        self.assertAlmostEqual(m["skill_source_agreement"], 0.5)

    def test_gold_tokens_fully_present_recall_one(self):
        rec = fda_record(
            [ev("fdarxbench_label", text="Serious warning: bleeding risk observed")],
            gold_answer="warning bleeding risk",
        )
        m = rpm.record_metrics(rec, None)
        self.assertAlmostEqual(m["evidence_recall_at_k"], 1.0)

    def test_gold_tokens_absent_recall_zero(self):
        rec = fda_record(
            [ev("fdarxbench_label", text="unrelated content here")],
            gold_answer="warning bleeding risk",
        )
        m = rpm.record_metrics(rec, None)
        self.assertAlmostEqual(m["evidence_recall_at_k"], 0.0)

    def test_partial_recall(self):
        rec = fda_record(
            [ev("fdarxbench_label", text="bleeding noted")],
            gold_answer="warning bleeding risk",
        )
        m = rpm.record_metrics(rec, None)
        # 1 of 3 gold tokens ("bleeding") present
        self.assertAlmostEqual(m["evidence_recall_at_k"], 1.0 / 3.0)

    def test_no_evidence_source_metrics_none(self):
        rec = fda_record([])
        m = rpm.record_metrics(rec, None)
        self.assertIsNone(m["source_precision_at_k"])
        self.assertIsNone(m["skill_source_agreement"])
        # recall still defined (gold tokens vs empty evidence → 0)
        self.assertAlmostEqual(m["evidence_recall_at_k"], 0.0)

    def test_evidence_utilization_full_and_none(self):
        rec = fda_record(
            [ev("fdarxbench_label", text="bleeding risk warning")],
            gold_answer="warning bleeding risk",
        )
        # prediction fully drawn from evidence → utilization 1.0
        m = rpm.record_metrics(rec, prediction="bleeding warning")
        self.assertAlmostEqual(m["evidence_utilization"], 1.0)
        # no predictions supplied → None
        m2 = rpm.record_metrics(rec, prediction=None)
        self.assertIsNone(m2["evidence_utilization"])


class TestAggregation(unittest.TestCase):
    def test_compute_groups_and_averages(self):
        # two records, one fully on-source, one fully off-source → mean precision 0.5
        recs = [
            fda_record([ev("fdarxbench_label")]),
            fda_record([ev("primekg")]),
        ]
        pattern = re.compile(rpm.DEFAULT_METHOD_REGEX)

        # monkey-patch loaders to avoid touching disk
        orig_load = rpm.load_jsonl
        rpm.load_jsonl = lambda path: recs
        try:
            rows = rpm.compute(["aug_bm25_test.jsonl"], [], pattern)
        finally:
            rpm.load_jsonl = orig_load

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["dataset"], "fdarxbench")
        self.assertEqual(row["task_type"], "factual")
        self.assertEqual(row["method"], "bm25")
        self.assertEqual(row["method_family"], "uniform")
        self.assertEqual(row["n_records"], 2)
        self.assertAlmostEqual(float(row["source_precision_at_k"]), 0.5)
        self.assertAlmostEqual(float(row["wrong_source_rate"]), 0.5)


if __name__ == "__main__":
    unittest.main()
