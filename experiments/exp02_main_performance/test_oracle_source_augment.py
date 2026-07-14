#!/usr/bin/env python
"""Offline unittest for oracle_source_augment.oracle_retrieve.

Uses a tiny synthetic KB + hand-built L2-normalized query/fact vectors -- no
encoder, no GPU, no real embedding files. Requires numpy + the local `retrievers`
module (DenseIndex/fact_to_evidence use numpy/faiss, not torch at call time); the
whole thing is skipped cleanly if those imports are unavailable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
    import oracle_source_augment as O
    O.oracle_retrieve  # ensure module imported
    _HAVE = True
    _WHY = ""
except Exception as e:  # pragma: no cover - environment guard
    _HAVE = False
    _WHY = f"{type(e).__name__}: {e}"


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


@unittest.skipUnless(_HAVE, f"numpy/retrievers unavailable: {_WHY}")
class TestOracleRetrieve(unittest.TestCase):
    def setUp(self):
        # 4 facts across 2 sources. Rows are unit vectors in 3-D.
        self.facts = [
            {"id": "L1", "source": "fdarxbench_label", "text": "label a", "relation": "label_context"},
            {"id": "L2", "source": "fdarxbench_label", "text": "label b", "relation": "label_context"},
            {"id": "C1", "source": "drugchat_chembl", "text": "chem a", "relation": "instruction_fact"},
            {"id": "C2", "source": "drugchat_chembl", "text": "chem b", "relation": "instruction_fact"},
        ]
        self.kb = np.stack([
            _unit([1.0, 0.0, 0.0]),   # L1
            _unit([0.9, 0.1, 0.0]),   # L2
            _unit([0.0, 1.0, 0.0]),   # C1
            _unit([0.0, 0.9, 0.1]),   # C2
        ]).astype(np.float32)

    def test_gold_fda_restricts_to_label_source(self):
        # An FDA factual record: gold skill -> fda_label_factual -> {fdarxbench_label}.
        rec = {"source": "fdarxbench", "task_type": "factual", "question": "q",
               "input_type": "FDA label context"}
        # Query points at the chembl cluster, but oracle must NOT be allowed to return chembl:
        # the restriction is what we assert (returned SET of sources), not intra-subset order
        # (which is a tie when the query is orthogonal to the allowed subspace).
        q = np.stack([_unit([0.0, 1.0, 0.0])]).astype(np.float32)
        ev, routing = O.oracle_retrieve([rec], self.facts, q, self.kb, top_k=8)
        self.assertEqual(routing[0][0], "fda_label_factual")
        self.assertEqual({e["source"] for e in ev[0]}, {"fdarxbench_label"})
        # exhaustive over the allowed subset: both label facts returned.
        self.assertEqual({e["id"] for e in ev[0]}, {"L1", "L2"})

    def test_gold_property_restricts_to_chembl_pubchem(self):
        rec = {"source": "molinst_property", "task_type": "property prediction",
               "question": "what is the logp", "decoded_smiles": "CCO"}
        q = np.stack([_unit([1.0, 0.0, 0.0])]).astype(np.float32)  # points at label cluster
        ev, routing = O.oracle_retrieve([rec], self.facts, q, self.kb, top_k=8)
        self.assertEqual(routing[0][0], "molecule_property_numeric")
        self.assertTrue({e["source"] for e in ev[0]} <= {"drugchat_chembl", "drugchat_pubchem"})
        self.assertEqual({e["id"] for e in ev[0]}, {"C1", "C2"})

    def test_dense_order_within_allowed_subset(self):
        # Query aligned inside the allowed (label) subspace -> deterministic dense order.
        rec = {"source": "fdarxbench", "task_type": "factual", "input_type": "FDA label context"}
        q = np.stack([_unit([1.0, 0.0, 0.0])]).astype(np.float32)  # closest to L1, then L2
        ev, _ = O.oracle_retrieve([rec], self.facts, q, self.kb, top_k=8)
        self.assertEqual([e["id"] for e in ev[0]], ["L1", "L2"])

    def test_top_k_caps_and_grouping(self):
        rec = {"source": "fdarxbench", "task_type": "factual", "input_type": "FDA label context"}
        q = np.stack([_unit([1.0, 0.0, 0.0])]).astype(np.float32)
        ev, _ = O.oracle_retrieve([rec], self.facts, q, self.kb, top_k=1)
        self.assertEqual(len(ev[0]), 1)
        self.assertEqual(ev[0][0]["id"], "L1")

    def test_two_records_distinct_routes(self):
        recs = [
            {"source": "fdarxbench", "task_type": "factual", "input_type": "FDA label context"},
            {"source": "molinst_property", "task_type": "property prediction", "question": "logp"},
        ]
        q = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])]).astype(np.float32)
        ev, routing = O.oracle_retrieve(recs, self.facts, q, self.kb, top_k=8)
        self.assertEqual({e["source"] for e in ev[0]}, {"fdarxbench_label"})
        self.assertTrue({e["source"] for e in ev[1]} <= {"drugchat_chembl", "drugchat_pubchem"})


if __name__ == "__main__":
    unittest.main()
