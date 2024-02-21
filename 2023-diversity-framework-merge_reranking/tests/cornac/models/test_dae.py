import unittest

from cornac.models import DAE
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit


dummy_train=[('0', '1', 2.0), ('1', '2', 4.0), ('2', '3', 3.5), ('2', '1', 3.0), ('1', '0', 2.5), ('0', '2', 3.0), ('0', '3', 4.0), ('1', '3', 2.0)]
dummy_test=[('0', '2', 4.0), ('1', '1', 4.0), ('2', '1', 1.0)]
class TestModels(unittest.TestCase):
    def test_with_dae_model(self):
        data = movielens.load_feedback(variant="100K")
        ratio_split = RatioSplit(
            data=data,
            test_size=0.2,
            exclude_unknowns=True,
            verbose=True,
            seed=123,
            rating_threshold=0.5,
        )
        dae = DAE(
            qk_dims=[200],
            pk_dims=[200],
            n_epochs=100,
            batch_size=100,
            learning_rate=0.001,
            weight_decay=0.0,
            dropout_p=0.5,
            seed=123,
            use_gpu=True,
            verbose=True,
        )
        train = ratio_split.train_set
        val = ratio_split.val_set
        dae.fit(train, val)

    def test_with_dummy_data(self):
        ratio_split = RatioSplit(
            data=dummy_train,
            test_size=0.2,
            exclude_unknowns=True,
            verbose=True,
            seed=123,
            rating_threshold=0.5,
        )
        dae = DAE(
            qk_dims=[1],
            pk_dims=[1],
            n_epochs=10,
            batch_size=100,
            learning_rate=0.001,
            weight_decay=0.0,
            dropout_p=0.5,
            seed=123,
            use_gpu=True,
            verbose=True,
        )
        dae.fit(ratio_split.train_set, ratio_split.val_set)
        model_saved = dae.save("result")


if __name__ == "__main__":
    unittest.main()
