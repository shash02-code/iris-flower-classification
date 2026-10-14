"""Explore the Iris dataset and compare simple classification models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def explore_data(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    species: list[str],
) -> Path:
    """Save a scatter plot and feature histograms for the Iris dataset."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = ("tab:blue", "tab:orange", "tab:green")

    for class_idx, class_name in enumerate(species):
        class_mask = targets == class_idx
        axes[0, 0].scatter(
            features[class_mask, 2],
            features[class_mask, 3],
            label=class_name,
            color=colors[class_idx],
            alpha=0.75,
        )
    axes[0, 0].set(
        xlabel=feature_names[2],
        ylabel=feature_names[3],
        title="Petal measurements by species",
    )
    axes[0, 0].legend()

    for feature_idx, axis in enumerate(axes.flat[1:5]):
        values = features[:, feature_idx]
        bins = np.linspace(values.min(), values.max(), 13)
        for class_idx, class_name in enumerate(species):
            axis.hist(
                features[targets == class_idx, feature_idx],
                alpha=0.55,
                label=class_name,
                color=colors[class_idx],
                bins=bins,
            )
        axis.set(
            xlabel=feature_names[feature_idx],
            ylabel="Count",
            title=f"Distribution of {feature_names[feature_idx]}",
        )
        axis.legend()

    axes.flat[5].set_visible(False)
    figure.tight_layout()
    plot_path = Path(__file__).resolve().parent / "iris_exploration.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def compare_models(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[str, ClassifierMixin, dict[str, float]]:
    """Select a classifier using stratified cross-validation."""
    models: dict[str, ClassifierMixin] = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {
        name: float(
            cross_val_score(
                model,
                features,
                targets,
                cv=cv,
                scoring="accuracy",
            ).mean()
        )
        for name, model in models.items()
    }
    best_name = max(scores, key=scores.__getitem__)
    return best_name, models[best_name], scores


def main() -> None:
    iris = load_iris()
    features = iris.data
    targets = iris.target
    species = list(iris.target_names)
    feature_names = list(iris.feature_names)

    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Class counts: {dict(zip(species, np.bincount(targets)))}")
    exploration_path = explore_data(features, targets, feature_names, species)
    print(f"Exploration plot saved to: {exploration_path}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )
    best_name, model, cv_scores = compare_models(x_train, y_train)
    print("\n5-fold cross-validation accuracy on training data:")
    for name, score in cv_scores.items():
        print(f"  {name}: {score:.3f}")
    print(f"Selected model: {best_name}")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(species)),
            target_names=species,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred, labels=np.arange(len(species))))

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=species,
        cmap="Blues",
        values_format="d",
    )
    display.figure_.tight_layout()
    matrix_path = Path(__file__).resolve().parent / "iris_confusion_matrix.png"
    display.figure_.savefig(matrix_path, dpi=150)
    plt.close(display.figure_)
    print(f"Confusion matrix plot saved to: {matrix_path}")


if __name__ == "__main__":
    main()
