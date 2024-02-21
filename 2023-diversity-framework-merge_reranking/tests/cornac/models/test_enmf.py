import unittest
import numpy as np
from cornac.models import ENMF
from cornac.datasets import amazon_clothing
from cornac.data import Reader
from cornac.eval_methods import RatioSplit


class TestModels(unittest.TestCase):
    def test_with_enmf_model(self):
        feedback = amazon_clothing.load_feedback(
            reader=Reader(bin_threshold=1.0))
        model = ENMF(num_epochs=2)
        ratio_split = RatioSplit(
            data=feedback,
            test_size=0.2,
            rating_threshold=1.0,
            seed=123,
            exclude_unknowns=True,
            verbose=True,
        )
        ts = ratio_split.train_set
        vs = ratio_split.val_set

        model.fit(ts, vs)

    def test_with_paper_data(self):
        train_data = Reader().read("./tests/ml_train.txt")
        test_data = Reader().read("./tests/ml_test.txt")
        rs = RatioSplit(
            train_data,
            exclude_unknowns=True,
            seed=123,
            rating_threshold=1,
            verbose=True,
        )
        model = ENMF(num_epochs=2)
        ts = rs.train_set
        vs = rs.val_set
        model.fit(ts, vs)
        model_saved = model.save("result")


if __name__ == '__main__':
    unittest.main()
