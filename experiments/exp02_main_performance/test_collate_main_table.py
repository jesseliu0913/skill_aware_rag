#!/usr/bin/env python
"""Offline stdlib-unittest for collate_main_table (no disk artifacts required)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collate_main_table as C  # noqa: E402


class TestFilenameParsing(unittest.TestCase):
    def test_method_form(self):
        g = C.parse_eval_filename("qwen2_5_7b_fdarxbench_oracle_source_bslret_test_eval.json")
        self.assertEqual(g, {"model": "qwen2_5_7b", "dataset": "fdarxbench",
                             "method": "oracle_source", "variant": "bslret"})

    def test_skillrouted_not_swallowed_by_dense_bge(self):
        # longest-first alternation must match dense_bge_skillrouted, not dense_bge.
        g = C.parse_eval_filename("qwen2_5_3b_mol_instructions_dense_bge_skillrouted_bslret_test_eval.json")
        self.assertIsNotNone(g)
        self.assertEqual(g["method"], "dense_bge_skillrouted")

    def test_plain_form_maps_to_none(self):
        g = C.parse_eval_filename("llama3_2_3b_fdarxbench_lora_test_eval.json")
        self.assertEqual(g["method"], "none")
        self.assertEqual(g["variant"], "lora")

    def test_non_eval_file_rejected(self):
        self.assertIsNone(C.parse_eval_filename("random_file.json"))


class TestHeadlineMetric(unittest.TestCase):
    def test_prefers_numeric_acc(self):
        self.assertEqual(C.headline_metric({"numeric_acc": 0.5, "token_f1": 0.9}), ("numeric_acc", 0.5))

    def test_falls_back_to_token_f1(self):
        self.assertEqual(C.headline_metric({"numeric_acc": None, "token_f1": 0.42}), ("token_f1", 0.42))

    def test_none_when_absent(self):
        self.assertIsNone(C.headline_metric({"exact": 0.1}))


class TestOrderingCheck(unittest.TestCase):
    def _rows(self, u, s, o):
        base = {"model": "m", "dataset": "fdarxbench", "subtask": "fdarxbench", "variant": "bslret"}
        return [
            {**base, "method": "dense_bge", "token_f1": u},
            {**base, "method": "dense_bge_skillrouted", "token_f1": s},
            {**base, "method": "oracle_source", "token_f1": o},
        ]

    def test_holds(self):
        res = C.check_ordering(self._rows(0.30, 0.40, 0.50), "dense_bge", "dense_bge_skillrouted", "oracle_source")
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0]["holds"])
        self.assertAlmostEqual(res[0]["uniform_gap"], 0.10)
        self.assertAlmostEqual(res[0]["oracle_gap"], 0.10)

    def test_upper_violation(self):
        # skill beats oracle -> upper bound violated.
        res = C.check_ordering(self._rows(0.30, 0.60, 0.50), "dense_bge", "dense_bge_skillrouted", "oracle_source")
        self.assertFalse(res[0]["holds"])
        self.assertFalse(res[0]["upper_holds"])
        self.assertTrue(res[0]["lower_holds"])

    def test_lower_violation(self):
        res = C.check_ordering(self._rows(0.50, 0.40, 0.60), "dense_bge", "dense_bge_skillrouted", "oracle_source")
        self.assertFalse(res[0]["holds"])
        self.assertFalse(res[0]["lower_holds"])

    def test_tolerance_absorbs_noise(self):
        res = C.check_ordering(self._rows(0.401, 0.400, 0.500), "dense_bge", "dense_bge_skillrouted",
                               "oracle_source", tol=0.01)
        self.assertTrue(res[0]["holds"])

    def test_incomplete_cell_skipped(self):
        rows = self._rows(0.3, 0.4, 0.5)[:2]  # drop oracle
        res = C.check_ordering(rows, "dense_bge", "dense_bge_skillrouted", "oracle_source")
        self.assertEqual(res, [])


class TestEndToEndGlob(unittest.TestCase):
    def test_collect_from_disk(self):
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d)
            (pred / "qwen2_5_7b_fdarxbench_dense_bge_bslret_test_eval.json").write_text(
                json.dumps([{"dataset": "fdarxbench", "n": 10, "avg_token_f1": 0.3, "exact": 0.1}])
            )
            (pred / "qwen2_5_7b_fdarxbench_oracle_source_bslret_test_eval.json").write_text(
                json.dumps([{"dataset": "fdarxbench", "n": 10, "avg_token_f1": 0.5, "exact": 0.2}])
            )
            rows: list = []
            C.collect_eval_jsons([str(pred)], rows)
            self.assertEqual(len(rows), 2)
            methods = {r["method"] for r in rows}
            self.assertEqual(methods, {"dense_bge", "oracle_source"})
            self.assertEqual(rows[0]["token_f1"], 0.3)


if __name__ == "__main__":
    unittest.main()
