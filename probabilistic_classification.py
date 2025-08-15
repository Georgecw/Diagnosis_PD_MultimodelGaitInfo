import numpy as np
from typing import Tuple, Dict


def load_features_labels(
    x_train_path: str,
    y_train_path: str,
    x_test_path: str,
    y_test_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Load train/test features and labels from CSV files.

    Assumptions:
    - Feature CSVs have a header row; the remainder are numeric values.
    - Label CSVs have a single column with a header 'Group'.
    """
    X_train = np.loadtxt(x_train_path, delimiter=",", skiprows=1)
    X_test = np.loadtxt(x_test_path, delimiter=",", skiprows=1)

    y_train_str = np.genfromtxt(y_train_path, delimiter=",", dtype=str, skip_header=1)
    y_test_str = np.genfromtxt(y_test_path, delimiter=",", dtype=str, skip_header=1)

    # Ensure 1D arrays even if a single row
    if y_train_str.ndim > 1:
        y_train_str = y_train_str[:, 0]
    if y_test_str.ndim > 1:
        y_test_str = y_test_str[:, 0]

    # Build label mapping based on training set
    unique_labels = list(dict.fromkeys(list(y_train_str)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    y_train = np.array([label_to_index[label] for label in y_train_str], dtype=int)
    y_test = np.array([label_to_index[label] for label in y_test_str], dtype=int)

    return X_train, y_train, X_test, y_test, label_to_index, index_to_label


def get_x_tilde(X: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def evaluate_basis_functions(length_scale: float, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF basis expansion evaluated at X with centers in Z.
    K(x, z) = exp(-||x - z||^2 / (2 * l^2))
    """
    X2 = np.sum(X ** 2, axis=1)
    Z2 = np.sum(Z ** 2, axis=1)
    ones_Z = np.ones(Z.shape[0])
    ones_X = np.ones(X.shape[0])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / (length_scale ** 2) * r2)


def choose_length_scale_by_median_distance(X: np.ndarray) -> float:
    """
    Heuristic: set l to median pairwise Euclidean distance among training points.
    """
    # Compute pairwise squared distances efficiently
    X2 = np.sum(X ** 2, axis=1)
    r2 = np.maximum(
        np.outer(X2, np.ones(X.shape[0])) - 2 * np.dot(X, X.T) + np.outer(np.ones(X.shape[0]), X2),
        0.0,
    )
    # Take upper triangle (i < j) distances
    iu = np.triu_indices(X.shape[0], k=1)
    dists = np.sqrt(r2[iu])
    # Guard against degenerate case
    median_dist = np.median(dists) if dists.size > 0 else 1.0
    # Avoid extremely small length-scale
    return float(max(median_dist, 1e-3))


def softmax(logits: np.ndarray) -> np.ndarray:
    # logits: N x K
    logits_max = np.max(logits, axis=1, keepdims=True)
    stabilized = logits - logits_max
    exp_vals = np.exp(stabilized)
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    return probs


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    Y = np.zeros((labels.shape[0], num_classes))
    Y[np.arange(labels.shape[0]), labels] = 1.0
    return Y


def predict_proba(X_tilde: np.ndarray, W: np.ndarray) -> np.ndarray:
    # X_tilde: N x D, W: D x K
    logits = np.dot(X_tilde, W)
    return softmax(logits)


def compute_average_ll(X_tilde: np.ndarray, Y_onehot: np.ndarray, W: np.ndarray) -> float:
    probs = predict_proba(X_tilde, W)
    # Avoid log(0)
    eps = 1e-12
    log_probs = np.log(np.clip(probs, eps, 1.0))
    ll = np.sum(Y_onehot * log_probs) / X_tilde.shape[0]
    return float(ll)


def fit_W(
    X_tilde_train: np.ndarray,
    y_train: np.ndarray,
    X_tilde_test: np.ndarray,
    y_test: np.ndarray,
    n_steps: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples, num_features = X_tilde_train.shape
    num_classes = int(np.max(y_train)) + 1

    Y_train_onehot = one_hot(y_train, num_classes)
    Y_test_onehot = one_hot(y_test, num_classes)

    rng = np.random.default_rng(0)
    W = rng.standard_normal(size=(num_features, num_classes)) * 0.01

    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)

    for i in range(n_steps):
        # Train gradient (softmax regression)
        probs_train = predict_proba(X_tilde_train, W)  # N x K
        gradient = np.dot(X_tilde_train.T, (Y_train_onehot - probs_train))  # D x K
        W = W + alpha * gradient

        ll_train[i] = compute_average_ll(X_tilde_train, Y_train_onehot, W)
        ll_test[i] = compute_average_ll(X_tilde_test, Y_test_onehot, W)

    return W, ll_train, ll_test


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    base = "Diagnosis_PD_MultimodelGaitInfo/PreprocessedData"
    x_train_path = f"{base}/X_train_reduced.csv"
    y_train_path = f"{base}/y_train.csv"
    x_test_path = f"{base}/X_test_reduced.csv"
    y_test_path = f"{base}/y_test.csv"

    (
        X_train,
        y_train,
        X_test,
        y_test,
        label_to_index,
        index_to_label,
    ) = load_features_labels(x_train_path, y_train_path, x_test_path, y_test_path)

    # Choose RBF length-scale by median pairwise distance heuristic
    length_scale = choose_length_scale_by_median_distance(X_train)

    # Basis expansion using training points as centers
    Phi_train = evaluate_basis_functions(length_scale, X_train, X_train)
    Phi_test = evaluate_basis_functions(length_scale, X_test, X_train)

    X_tilde_train = get_x_tilde(Phi_train)
    X_tilde_test = get_x_tilde(Phi_test)

    # Optimizer settings (reference code used 0.003 for binary; we keep similar scale)
    n_steps = 1000
    alpha = 0.003

    W, ll_train, ll_test = fit_W(
        X_tilde_train, y_train, X_tilde_test, y_test, n_steps=n_steps, alpha=alpha
    )

    # Predictions
    probs_test = predict_proba(X_tilde_test, W)
    y_pred_test = np.argmax(probs_test, axis=1)

    test_acc = accuracy(y_test, y_pred_test)

    num_classes = len(label_to_index)
    cm = confusion_matrix(y_test, y_pred_test, num_classes=num_classes)

    # Pretty print results
    print("Length-scale (l):", length_scale)
    print("Train LL (last):", ll_train[-1])
    print("Test  LL (last):", ll_test[-1])
    print("Test Accuracy:", f"{test_acc:.4f}")

    # Confusion matrix with label names
    labels_ordered = [index_to_label[i] for i in range(num_classes)]
    print("Labels order:", labels_ordered)
    print("Confusion Matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        print(labels_ordered[i], row.tolist())


if __name__ == "__main__":
    main()


