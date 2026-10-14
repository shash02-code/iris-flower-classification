"""Basic tests for the Iris classification workflow."""

import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from iris_classification import compare_models


class IrisClassificationTests(unittest.TestCase):
    def test_dataset_has_expected_shape_and_balanced_classes(self) -> None:
        iris = load_iris()

        self.assertEqual(iris.data.shape, (150, 4))
        self.assertEqual(len(iris.target_names), 3)
        np.testing.assert_array_equal(np.bincount(iris.target), [50, 50, 50])

    def test_selected_model_predicts_valid_labels_for_test_samples(self) -> None:
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            iris.data,
            iris.target,
            test_size=0.2,
            random_state=42,
            stratify=iris.target,
        )

        model_name, model, scores = compare_models(x_train, y_train)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        self.assertIn(model_name, scores)
        self.assertEqual(model_name, max(scores, key=scores.__getitem__))
        self.assertEqual(
            set(scores),
            {"Logistic Regression", "K-Nearest Neighbors", "Decision Tree"},
        )
        self.assertEqual(y_pred.shape, y_test.shape)
        self.assertTrue(set(y_pred).issubset(set(iris.target)))


if __name__ == "__main__":
    unittest.main()
