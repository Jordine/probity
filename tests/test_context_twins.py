import hashlib
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from probity_benchmark.build_context_twins import build_twins
from probity_benchmark.fit_local_iteration import dev_value


class ContextTwins(unittest.TestCase):
    def test_text_and_position_cannot_classify_the_paired_claims(self):
        rows = build_twins()
        for split in ["train", "dev", "test"]:
            y, scores, positions, units, groups = [], [], [], [], []
            for row in rows:
                if row["split"] != split:
                    continue
                for index, span in enumerate(row["spans"]):
                    y.append(span["label"])
                    scores.append(
                        int.from_bytes(
                            hashlib.sha256(span["text"].encode()).digest(), "big"
                        )
                    )
                    positions.append(index)
                    units.append(row["id"])
                    groups.append(row["group"])
            y, scores, positions, units, groups = map(
                np.asarray, (y, scores, positions, units, groups)
            )
            self.assertEqual(roc_auc_score(y, scores.astype(float)), 0.5)
            self.assertEqual(roc_auc_score(y, positions), 0.5)
            self.assertAlmostEqual(
                dev_value(y, scores.astype(float), units, groups), 0.5
            )


if __name__ == "__main__":
    unittest.main()
