"""

HELAL Camélia
------
A4SYS - 2025/2026

"""

from typing import Tuple, List, Dict

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def load_data(mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Parameters
    ----------
    X : feature matrix of shape (n_samples, n_features)
    y : target vector of shape (n_samples,)
    feature_names : array of feature names
    """
    
    data = loadmat(mat_path)
    X = data["X"]
    y = data["y"].ravel()  # ravel() → convert to 1D vector
    feature_names = data["feature_names"].ravel()  

    # Convert feature_names from object dtype MATLAB to Python strings
    feature_names = np.array([str(fn[0]) if isinstance(fn, np.ndarray) else str(fn) for fn in feature_names])

    return X, y, feature_names


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split the data into train, validation and test sets, then standardize the features.

    Parameters
    ----------
    X : Feature matrix of shape (n_samples, n_features)
    y : Target vector of shape (n_samples,)
    test_size : Proportion of the dataset kept aside for test
    val_size : Proportion of the train+val part used for validation
    random_state : Seed for reproducibility

    Returns
    -------
    X_train_scaled : ndarray
    X_val_scaled : ndarray
    X_test_scaled : ndarray
    y_train : ndarray
    y_val : ndarray
    y_test : ndarray
    scaler : StandardScaler
    """
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val (inside train+val)
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size,
        random_state=random_state, stratify=y_train_val
    )

    # Fit the scaler on training features only (important to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion matrix"
) -> None:
    
    """
    Parameters
    ----------
    y_true : Ground truth labels
    y_pred : Predicted labels
    class_names : Ordered list of class names to display on the axes
    """
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    # Rotate the tick labels on the x-axis for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate the cells with the integer counts
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str]
) -> None:
    
    """
    Plot one-vs-rest ROC curves for a multi-class classifier

    Parameters
    ----------
    y_true : True class indices of shape (n_samples,)
    y_proba : Predicted probabilities of shape (n_samples, n_classes)
    class_names : Class names corresponding to the columns of `y_proba`
    """
    
    n_classes = len(class_names)
    y_true_bin = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_feature_importances(
    feature_names: np.ndarray,
    importances: np.ndarray,
    top_k: int = 10,
    title: str = "Feature importance"
) -> None:
    
    """
    Plot the top-k most important features as a horizontal bar chart.

    Parameters
    ----------
    feature_names : Names of the features.
    importances : Importance scores for each feature.
    top_k : Number of top features to display. If > number of features => clipped
    """
    
    n_features = len(importances)
    k = min(top_k, n_features)

    # Get indices of the top-k features sorted by importance
    idx = np.argsort(importances)[-k:]
    top_features = feature_names[idx]
    top_importances = importances[idx]

    # Plot horizontally for readability
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(k), top_importances, align="center")
    ax.set_yticks(range(k))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
