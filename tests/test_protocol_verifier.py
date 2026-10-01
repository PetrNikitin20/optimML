import unittest
from tools.verify_protocol_v2 import ece, recompute_rows


class ProtocolVerifierTests(unittest.TestCase):
    def test_calibration_endpoints(self):
        self.assertEqual(ece([0, 1], [0, 1]), 0)
        self.assertEqual(ece([0.5, 0.5], [1, 1]), 0.5)

    def test_row_metrics_and_identities(self):
        rows = [{"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0,
                 "policy_chosen_logp": -1.0, "policy_rejected_logp": -2.0,
                 "sft_chosen_logp": -1.0, "sft_rejected_logp": -2.0,
                 "presentation_target": float(i % 2 == 0), "presentation_probability": 0.5}
                for i in range(4)]
        result = recompute_rows(rows, 0.1)
        self.assertEqual(result["likelihood_ranking_accuracy"], 1.0)
        self.assertEqual(result["reference_ratio_ranking_accuracy"], 0.0)
        self.assertEqual(result["brier"], 0.25)
        self.assertEqual(result["ece_15"], 0.0)
        rows[0]["d"] = 1
        with self.assertRaises(ValueError):
            recompute_rows(rows, 0.1)


if __name__ == "__main__":
    unittest.main()
