import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
)


def logistic_stable(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos_mask = x >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[~pos_mask])
    out[~pos_mask] = exp_x / (1.0 + exp_x)
    return out


def predict_prob(X_tilde: np.ndarray, w: np.ndarray) -> np.ndarray:
    return logistic_stable(np.dot(X_tilde, w))


def compute_average_ll(X_tilde: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = predict_prob(X_tilde, w)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def get_x_tilde(X: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((X.shape[0], 1), dtype=float), X), axis=1)


def fit_w(
    X_tilde_train: np.ndarray,
    y_train: np.ndarray,
    X_tilde_test: np.ndarray,
    y_test: np.ndarray,
    n_steps: int,
    alpha: float,
    l2_reg: float = 0.0,
    print_every: int = 20,
):
    rng = np.random.default_rng(0)
    w = rng.standard_normal(X_tilde_train.shape[1])
    ll_train = np.zeros(n_steps, dtype=float)
    ll_test = np.zeros(n_steps, dtype=float)

    for i in range(n_steps):
        p_train = predict_prob(X_tilde_train, w)
        gradient = (y_train - p_train) @ X_tilde_train  # shape: (D,)

        if l2_reg > 0.0:
            w_no_bias = w.copy()
            w_no_bias[0] = 0.0
            gradient = gradient - l2_reg * w_no_bias

        w = w + alpha * gradient
        ll_train[i] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[i] = compute_average_ll(X_tilde_test, y_test, w)

        if (i + 1) % print_every == 0 or i == n_steps - 1:
            preds = (predict_prob(X_tilde_test, w) > 0.5).astype(int)
            acc = float(np.mean(preds == y_test))
            print(f"step {i+1:4d} | ll_train {ll_train[i]:.4f} | ll_test {ll_test[i]:.4f} | acc_test {acc:.3f}")

    return w, ll_train, ll_test


def evaluate_basis_functions(l: float, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    # X: (N, D), Z: (M, D)
    X2 = np.sum(X ** 2, axis=1)
    Z2 = np.sum(Z ** 2, axis=1)
    ones_Z = np.ones(Z.shape[0])
    ones_X = np.ones(X.shape[0])
    r2 = np.outer(X2, ones_Z) - 2.0 * (X @ Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / (l ** 2) * r2)


def load_xy(
    x_train_path: str,
    x_test_path: str,
    y_train_path: str,
    y_test_path: str,
):
    # X
    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)

    # y
    y_train_df = pd.read_csv(y_train_path)
    y_test_df = pd.read_csv(y_test_path)
    y_train_series = y_train_df.iloc[:, 0].astype(str).str.strip()
    y_test_series = y_test_df.iloc[:, 0].astype(str).str.strip()

    label_map = {"Young": 0, "PD_H&Y1": 1}
    try:
        y_train = y_train_series.map(label_map).values.astype(int)
        y_test = y_test_series.map(label_map).values.astype(int)
    except Exception as e:
        raise RuntimeError(f"无法映射标签到 {label_map}: {e}")

    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        raise ValueError("X_train 或 X_test 含有 NaN，请检查输入数据。")

    return X_train, X_test, y_train, y_test


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std, X_test_std


def plot_ll(ll: np.ndarray, title: str, save_path: Optional[str] = None):
    plt.figure()
    plt.plot(np.arange(1, len(ll) + 1), ll, "r-")
    plt.xlabel("Steps")
    plt.ylabel("Average log-likelihood")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def grid_search_rbf(
    X: np.ndarray,
    y: np.ndarray,
    l_grid: List[float],
    k_grid: List[int],
    n_splits: int,
    alpha: float,
    n_steps: int,
    l2_reg: float,
    random_state: int = 0,
    selection_metric: str = "auc",
) -> Tuple[Tuple[float, int], Dict[Tuple[float, int], Dict[str, float]]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    report: Dict[Tuple[float, int], Dict[str, float]] = {}

    for l_val in l_grid:
        for k_val in k_grid:
            val_ll_list: List[float] = []
            val_bal_acc_list: List[float] = []
            val_auc_list: List[float] = []

            for tr_idx, va_idx in cv.split(X, y):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                # 标准化按折内训练集拟合，避免泄漏
                mean = X_tr.mean(axis=0)
                std = X_tr.std(axis=0)
                std[std == 0.0] = 1.0
                X_tr_std = (X_tr - mean) / std
                X_va_std = (X_va - mean) / std

                # k-means 取 k 个中心
                kmeans = KMeans(n_clusters=k_val, random_state=random_state, n_init=10)
                kmeans.fit(X_tr_std)
                centers = kmeans.cluster_centers_

                # RBF 特征
                Phi_tr = evaluate_basis_functions(l_val, X_tr_std, centers)
                Phi_va = evaluate_basis_functions(l_val, X_va_std, centers)
                Xtr_tilde = get_x_tilde(Phi_tr)
                Xva_tilde = get_x_tilde(Phi_va)

                # 训练（减少打印频率）
                _, _, _ = None, None, None
                w, _, _ = fit_w(
                    Xtr_tilde, y_tr, Xva_tilde, y_va,
                    n_steps=n_steps, alpha=alpha, l2_reg=l2_reg,
                    print_every=n_steps + 1,
                )

                # 验证表现
                ll_va = compute_average_ll(Xva_tilde, y_va, w)
                y_prob_va = predict_prob(Xva_tilde, w)
                preds = (y_prob_va > 0.5).astype(int)
                try:
                    bal_acc_va = balanced_accuracy_score(y_va, preds)
                except Exception:
                    bal_acc_va = float("nan")
                try:
                    auc_va = roc_auc_score(y_va, y_prob_va)
                except Exception:
                    auc_va = float("nan")

                val_ll_list.append(ll_va)
                val_bal_acc_list.append(bal_acc_va)
                val_auc_list.append(auc_va)

            # 计算均值与方差（忽略 NaN）
            mean_ll = float(np.nanmean(val_ll_list))
            std_ll = float(np.nanstd(val_ll_list, ddof=1) if len(val_ll_list) > 1 else 0.0)
            mean_bal = float(np.nanmean(val_bal_acc_list))
            std_bal = float(np.nanstd(val_bal_acc_list, ddof=1) if len(val_bal_acc_list) > 1 else 0.0)
            mean_auc = float(np.nanmean(val_auc_list))
            std_auc = float(np.nanstd(val_auc_list, ddof=1) if len(val_auc_list) > 1 else 0.0)

            report[(l_val, k_val)] = {
                "mean_ll_val": mean_ll,
                "std_ll_val": std_ll,
                "mean_bal_acc_val": mean_bal,
                "std_bal_acc_val": std_bal,
                "mean_auc_val": mean_auc,
                "std_auc_val": std_auc,
            }

    # 选取最优：默认按 AUC，若 AUC 为 NaN 则以 balanced accuracy 作为备选
    def score_item(item):
        cfg, stats = item
        primary = stats.get("mean_auc_val", float("nan")) if selection_metric == "auc" else stats.get("mean_bal_acc_val", float("nan"))
        secondary = stats.get("mean_bal_acc_val", float("nan")) if selection_metric == "auc" else stats.get("mean_auc_val", float("nan"))
        # 将 NaN 视为 -inf 以避免被选中
        p = primary if not np.isnan(primary) else -1e9
        s = secondary if not np.isnan(secondary) else -1e9
        return (p, s)

    best_cfg = max(report.items(), key=score_item)[0]
    return best_cfg, report


def print_eval_details(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    # 真实标签分布与多数类基线
    classes, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    majority_baseline = counts.max() / total if total > 0 else float("nan")
    label_names = {0: "Young", 1: "PD_H&Y1"}
    print("\n[Eval] 测试集真实标签分布:")
    for c, ct in zip(classes, counts):
        print(f"  label {int(c)} ({label_names.get(int(c), str(c))}): {ct} ({ct/total:.3f})")
    print(f"  多数类基线准确率: {majority_baseline:.3f}")

    # 预测标签分布，检测是否塌缩到单一类别
    p_classes, p_counts = np.unique(y_pred, return_counts=True)
    print("[Eval] 预测标签分布:")
    for c, ct in zip(p_classes, p_counts):
        print(f"  pred {int(c)} ({label_names.get(int(c), str(c))}): {ct} ({ct/total:.3f})")
    if len(p_classes) == 1:
        only_c = int(p_classes[0])
        print(f"  警告: 预测塌缩为单一类别 {only_c} ({label_names.get(only_c, str(only_c))})")

    # 混淆矩阵与指标
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)
    print("[Eval] 混淆矩阵 [labels 顺序: 0,1]:")
    print(cm)
    try:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        print(f"[Eval] balanced accuracy: {bal_acc:.3f}")
    except Exception:
        pass

    # 概率分布与 AUC
    if y_prob is not None and y_prob.size == total:
        print(
            f"[Eval] 概率统计: min={np.min(y_prob):.3f}, max={np.max(y_prob):.3f}, mean={np.mean(y_prob):.3f}"
        )
        try:
            # roc_auc 需要 y_true 出现两类
            auc = roc_auc_score(y_true, y_prob)
            print(f"[Eval] ROC-AUC: {auc:.3f}")
        except Exception:
            print("[Eval] ROC-AUC 无法计算（可能因单一类别或样本过少）")

    # 分类报告（精确率/召回率/F1）
    try:
        print("[Eval] classification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[label_names.get(0, "0"), label_names.get(1, "1")],
                digits=3,
                zero_division=0,
            )
        )
    except Exception:
        pass

def main():
    # 路径
    xtr = "binClassificationData/X_train_reduced.csv"
    xte = "binClassificationData/X_test_reduced.csv"
    ytr = "binClassificationData/y_train.csv"
    yte = "binClassificationData/y_test.csv"

    # 加载
    X_train, X_test, y_train, y_test = load_xy(xtr, xte, ytr, yte)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    # 仅 RBF：网格搜索 l 与 k（k-means 中心数）
    print("==== RBF 网格搜索 (Stratified K-Fold) ====")
    l_grid: List[float] = [0.2, 0.5, 1.0, 2.0, 5.0]
    k_grid: List[int] = [10, 15, 20]
    n_splits = 5
    alpha = 3e-3
    n_steps = 600
    l2_reg = 1e-3

    best_cfg, cv_report = grid_search_rbf(
        X_train,
        y_train,
        l_grid=l_grid,
        k_grid=k_grid,
        n_splits=n_splits,
        alpha=alpha,
        n_steps=n_steps,
        l2_reg=l2_reg,
        random_state=0,
        selection_metric="auc",
    )

    print("候选配置验证指标（优先 AUC，其次 balanced acc；同时显示 LL）：")
    def sort_key(item):
        stats = item[1]
        auc = stats.get("mean_auc_val", float("nan"))
        bal = stats.get("mean_bal_acc_val", float("nan"))
        # NaN 排后
        auc_sort = auc if not np.isnan(auc) else -1e9
        bal_sort = bal if not np.isnan(bal) else -1e9
        return (auc_sort, bal_sort)

    for (l_val, k_val), stats in sorted(cv_report.items(), key=sort_key, reverse=True):
        print(
            f"  l={l_val:<4} k={k_val:<2} | AUC={stats['mean_auc_val']:.3f} (+/-{stats['std_auc_val']:.3f}) | "
            f"BalAcc={stats['mean_bal_acc_val']:.3f} (+/-{stats['std_bal_acc_val']:.3f}) | "
            f"LL={stats['mean_ll_val']:.4f} (+/-{stats['std_ll_val']:.4f})"
        )

    best_l, best_k = best_cfg
    print(f"选中最优超参: l={best_l}, k={best_k}")

    # 用最优超参在全训练集上重训，并在测试集评估
    print("\n==== RBF 最优超参下最终训练与测试 ====")
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0.0] = 1.0
    Xtr_std = (X_train - X_mean) / X_std
    Xte_std = (X_test - X_mean) / X_std

    kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    kmeans.fit(Xtr_std)
    centers = kmeans.cluster_centers_

    Phi_tr = evaluate_basis_functions(best_l, Xtr_std, centers)
    Phi_te = evaluate_basis_functions(best_l, Xte_std, centers)
    Xrbf_tilde_train = get_x_tilde(Phi_tr)
    Xrbf_tilde_test = get_x_tilde(Phi_te)

    w_rbf, ll_train_rbf, ll_test_rbf = fit_w(
        Xrbf_tilde_train,
        y_train,
        Xrbf_tilde_test,
        y_test,
        n_steps=n_steps,
        alpha=alpha,
        l2_reg=l2_reg,
        print_every=max(1, n_steps // 10),
    )
    preds_rbf = (predict_prob(Xrbf_tilde_test, w_rbf) > 0.5).astype(int)
    acc_rbf = float(np.mean(preds_rbf == y_test))
    print(
        f"[RBF best] acc_test={acc_rbf:.3f}, ll_train={ll_train_rbf[-1]:.4f}, ll_test={ll_test_rbf[-1]:.4f}"
    )

    # 详细评估输出，检测是否只预测多数类等
    y_prob = predict_prob(Xrbf_tilde_test, w_rbf)
    print_eval_details(y_test.astype(int), preds_rbf.astype(int), y_prob)

    plot_dir = "binClassificationData/plots"
    plot_ll(
        ll_train_rbf,
        f"RBF (l={best_l}, k={best_k}) - Train LL",
        os.path.join(plot_dir, f"rbf_l{best_l}_k{best_k}_train_ll.png"),
    )
    plot_ll(
        ll_test_rbf,
        f"RBF (l={best_l}, k={best_k}) - Test LL",
        os.path.join(plot_dir, f"rbf_l{best_l}_k{best_k}_test_ll.png"),
    )


if __name__ == "__main__":
    main()


