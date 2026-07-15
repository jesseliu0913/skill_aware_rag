import unittest

from analyze_preliminary import answer_token_recall, discovery_partition, numeric_match, question_type, token_f1


class PreliminaryAnalysisTest(unittest.TestCase):
    def test_question_types_do_not_need_gold_answer(self):
        self.assertEqual(question_type({"dataset": "fdarxbench", "task_type": "multihop"}), "fda_multihop")
        self.assertEqual(question_type({"dataset": "mol_instructions", "task_type": "property prediction"}), "molecule_property")

    def test_metrics(self):
        self.assertEqual(token_f1("a useful fact", "useful fact"), 0.8)
        self.assertEqual(numeric_match("The value is 10.4", "10"), 1.0)
        self.assertIsNone(numeric_match("answer", "not numeric"))

    def test_partition_is_stable(self):
        self.assertEqual(discovery_partition("example-1"), discovery_partition("example-1"))
        self.assertIn(discovery_partition("example-1"), {"discovery", "evaluation"})

    def test_answer_token_recall(self):
        self.assertEqual(answer_token_recall("alpha beta extra", "alpha beta"), 1.0)
        self.assertEqual(answer_token_recall("alpha", "alpha beta"), 0.5)


if __name__ == "__main__":
    unittest.main()
