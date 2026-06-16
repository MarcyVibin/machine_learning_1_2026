import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

#def loadData(D_path="D.csv", D_out_path="D_out.csv") -> pd.DataFrame:
print("Loading data …")
D_path="D.csv"
D_out_path="D_out.csv"
df     = pd.read_csv(D_path)
df_out = pd.read_csv(D_out_path)

feature_cols = [f"feature_{i}" for i in range(12)]
X = df[feature_cols]
y       = df["label"]
X_out   = df_out[feature_cols]

# Mark origin for combined plots
df["source"]     = "train"
df_out["source"] = "outlier"
df_combined      = pd.concat(
    [df[feature_cols + ["source"]],
    df_out[feature_cols + ["source"]]],
    ignore_index=True
)



#=======================================
#
# Helper Functions
#
#=======================================
def executePlot(fig, fig_name:str, show:bool = True, save:bool = False):
    if save: fig.savefig(fig_name)
    if show: plt.show()
    plt.close()


def printStatistics(should_print:bool = True):
    if should_print:
        print("\n── Summary statistics ───────────────────────────────────────────────")
        print(f"  Training set : {df.shape[0]} samples, {len(feature_cols)} features, {y.nunique()} classes")
        print(f"  Outlier set  : {df_out.shape[0]} samples")
        print("───────────────────────────────────────────────")
        print(df[feature_cols].describe().round(3).to_string())
        print("\nClass distribution:")
        print(y.value_counts().sort_index().to_string())
        print(f"\nMissing values: {df.isnull().sum().sum()}")


def handlePlots(show: bool = True, save_fig: bool = False):

    if not show and not save_fig:
        return

    #========= PLOT 1 =========
    print("\n[Plot 1] Class label distribution …")
    counts = y.value_counts().sort_index()
    colors = sns.color_palette("tab10", n_colors=5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_xlabel("Class label", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Sample count per class", fontsize=13, fontweight="bold")
    axes[0].set_xticks(counts.index)
    axes[0].set_ylim(0, counts.max() * 1.15)
    for i, (cls, cnt) in enumerate(counts.items()):
        axes[0].text(cls, cnt + 20, str(cnt), ha="center", va="bottom", fontsize=10)

    axes[1].pie(counts.values, labels=[f"Class {c}" for c in counts.index],
                autopct="%1.1f%%", colors=colors, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.2))
    axes[1].set_title("Class proportion", fontsize=13, fontweight="bold")

    plt.suptitle("Figure 1 — Class Label Distribution", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    executePlot(fig, "img/01_class_label_distribution", show, save_fig)

    #========= PLOT 2 =========
    print("[Plot 2] Feature distributions per class (violin) …")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    df_plot = df[feature_cols + ["label"]].copy()
    df_plot["label"] = df_plot["label"].astype(str)

    for i, col in enumerate(feature_cols):
        sns.violinplot(data=df_plot, x="label", y=col, ax=axes[i],
                       palette="tab10", inner="box", linewidth=0.8)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Class", fontsize=9)
        axes[i].set_ylabel("Value", fontsize=9)
        axes[i].tick_params(labelsize=8)

    plt.suptitle("Figure 2 — Feature Distributions per Class (Violin Plots)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # fixed: was clipping the title
    executePlot(fig, "img/02_feature_distributions_per_class", show, save_fig)

    #========= PLOT 3 =========
    print("[Plot 3] Correlation heatmap …")
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Figure 3 — Feature Correlation Matrix",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    executePlot(fig, "img/03_correlation_heatmap", show, save_fig)

    #========= PLOT 4 =========
    print("[Plot 4] PCA projection …")
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(pd.concat([X, X_out], ignore_index=True))
    X_scaled     = X_all_scaled[:len(X)]
    X_out_scaled = X_all_scaled[len(X):]

    pca       = PCA(n_components=2, random_state=42)
    X_pca     = pca.fit_transform(X_scaled)
    X_out_pca = pca.transform(X_out_scaled)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in sorted(y.unique()):
        mask = (y == cls).values
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f"Class {cls}", color=colors[cls],
                   alpha=0.4, s=15, edgecolors="none")
    ax.scatter(X_out_pca[:, 0], X_out_pca[:, 1],
               label="Known outliers", color="black",
               marker="x", s=60, linewidths=1.2, zorder=5)
    ax.set_xlabel(f"PC 1 ({explained[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC 2 ({explained[1]:.1f}% variance)", fontsize=12)
    ax.set_title("Figure 4 — PCA Projection (coloured by class, outliers marked)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, markerscale=1.5)
    plt.tight_layout()
    executePlot(fig, "img/04_pca_projection", show, save_fig)

    #========= PLOT 4b =========
    print("[Plot 4b] PCA scree plot …")
    pca_full = PCA(random_state=42).fit(X_scaled)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 13), pca_full.explained_variance_ratio_ * 100,
           color=sns.color_palette("Blues_d", 12), edgecolor="white")
    ax.plot(range(1, 13), cumvar, "o-", color="crimson", label="Cumulative %")
    ax.axhline(95, color="grey", linestyle="--", linewidth=0.8, label="95% threshold")
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance (%)", fontsize=12)
    ax.set_title("Figure 4b — PCA Scree Plot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    executePlot(fig, "img/04b_pca_scree", show, save_fig)

    #========= PLOT 5 =========
    print("[Plot 5] t-SNE projection (this may take ~1 min) …")
    n_tsne = min(2000, len(X_scaled))
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(X_scaled), size=n_tsne, replace=False)

    X_tsne_input = np.vstack([X_scaled[idx], X_out_scaled])
    y_tsne       = y.values[idx]

    tsne    = TSNE(n_components=2, perplexity=40, random_state=42,
                   n_iter=1000, learning_rate="auto", init="pca")
    X_2d     = tsne.fit_transform(X_tsne_input)
    X_2d_in  = X_2d[:n_tsne]
    X_2d_out = X_2d[n_tsne:]

    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in sorted(np.unique(y_tsne)):
        m = y_tsne == cls
        ax.scatter(X_2d_in[m, 0], X_2d_in[m, 1],
                   label=f"Class {cls}", color=colors[cls],
                   alpha=0.45, s=15, edgecolors="none")
    ax.scatter(X_2d_out[:, 0], X_2d_out[:, 1],
               label="Known outliers", color="black",
               marker="x", s=70, linewidths=1.3, zorder=5)
    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.set_title("Figure 5 — t-SNE Projection (coloured by class, outliers marked)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, markerscale=1.5)
    plt.tight_layout()
    executePlot(fig, "img/05_tsne_projection", show, save_fig)

    #========= PLOT 6 =========
    print("[Plot 6] Inlier vs outlier feature distributions …")
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(df[col], bins=40, density=True, alpha=0.5,
                     color="steelblue", label="Inliers", edgecolor="none")
        axes[i].hist(df_out[col], bins=20, density=True, alpha=0.6,
                     color="crimson", label="Outliers", edgecolor="none")
        for data, color in [(df[col], "steelblue"), (df_out[col], "crimson")]:
            kde  = gaussian_kde(data.dropna(), bw_method="scott")
            xgrd = np.linspace(data.min(), data.max(), 300)
            axes[i].plot(xgrd, kde(xgrd), color=color, linewidth=1.8)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Value", fontsize=9)
        axes[i].set_ylabel("Density", fontsize=9)
        axes[i].tick_params(labelsize=8)
        if i == 0:
            axes[i].legend(fontsize=9)

    plt.suptitle("Figure 6 — Feature Distributions: Inliers vs Known Outliers",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    executePlot(fig, "img/06_inlier_vs_outlier_distributions", show, save_fig)

    #========= PLOT 7 =========
    print("[Plot 7] Per-class mean feature heatmap …")
    df_scaled_tmp          = df[feature_cols].copy()
    df_scaled_tmp[:]       = StandardScaler().fit_transform(df_scaled_tmp)
    df_scaled_tmp["label"] = y.values
    class_means            = df_scaled_tmp.groupby("label")[feature_cols].mean()

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(class_means, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, cbar_kws={"label": "Standardised mean"},
                ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_title("Figure 7 — Per-class Mean Feature Profiles (standardised)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    executePlot(fig, "img/07_class_mean_feature_heatmap", show, save_fig)

def testKNN(k=8):
    print("\n── KNN Test ─────────────────────────────────────────────────────────")
    print(f"  Using K = {k}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_vals = X[feature_cols].values
    y_vals = y.values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_vals, y_vals)):
        X_train, X_val = X_vals[train_idx], X_vals[val_idx]
        y_train, y_val = y_vals[train_idx], y_vals[val_idx]

        clf, scaler = trainKNN(X_train, y_train, k=k)
        y_pred = scaler.transform(X_val)
        y_pred = clf.predict(y_pred)

        score = f1_score(y_val, y_pred, average="macro")
        fold_scores.append(score)
        print(f"  Fold {fold+1}: macro-F1 = {score:.4f}")

    print(f"\n  Mean macro-F1 : {np.mean(fold_scores):.4f}")
    print(f"  Std macro-F1  : {np.std(fold_scores):.4f}")
    print("\n── Full classification report (last fold) ───────────────────────────")
    print(classification_report(y_val, y_pred,
                                target_names=[f"Class {i}" for i in range(5)]))
    print(f"  Accuracy : {accuracy_score(y_val, y_pred):.4f}")

def findBestK(k_values=[i for i in range(1, 15)]):
    print("\n── K Selection ──────────────────────────────────────────────────────")
    X_vals = X[feature_cols].values
    y_vals = y.values
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_k, best_score = k_values[0], 0.0
    for k in k_values:
        fold_scores = []
        for train_idx, val_idx in skf.split(X_vals, y_vals):
            clf, scaler = trainKNN(X_vals[train_idx], y_vals[train_idx], k=k)
            y_pred      = clf.predict(scaler.transform(X_vals[val_idx]))
            fold_scores.append(f1_score(y_vals[val_idx], y_pred, average="macro"))

        mean = np.mean(fold_scores)
        print(f"  K={k:2d}  macro-F1={mean:.4f} ± {np.std(fold_scores):.4f}")
        if mean > best_score:
            best_k, best_score = k, mean

    print(f"\n  → Best K = {best_k}  (macro-F1 = {best_score:.4f})")
    return best_k


#=======================================
#
# Section 4 — Model Experimentation
#
#=======================================

def run_experiment(name, clf, X_vals, y_vals, use_scaler=True):
    """5-fold stratified CV. Returns (mean_macro_f1, oof_true, oof_pred)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s = []
    oof_true, oof_pred = [], []

    for train_idx, val_idx in skf.split(X_vals, y_vals):
        X_tr, X_val = X_vals[train_idx], X_vals[val_idx]
        y_tr, y_val = y_vals[train_idx], y_vals[val_idx]

        if use_scaler:
            sc = StandardScaler()
            X_tr  = sc.fit_transform(X_tr)
            X_val = sc.transform(X_val)

        m = clone(clf)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)

        fold_f1s.append(f1_score(y_val, y_pred, average="macro"))
        oof_true.extend(y_val)
        oof_pred.extend(y_pred)

    mean_f1 = np.mean(fold_f1s)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  CV Macro-F1: {mean_f1:.4f} ± {np.std(fold_f1s):.4f}  |  Accuracy: {accuracy_score(oof_true, oof_pred):.4f}")
    print(classification_report(oof_true, oof_pred,
                                target_names=[f"Class {i}" for i in range(5)]))
    return mean_f1, oof_true, oof_pred


def plotConfusionMatrix(y_true, y_pred, title, show=True, save_fig=False,
                        fname="img/08_confusion_matrix_best"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(5)],
                yticklabels=[f"Class {i}" for i in range(5)], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    executePlot(fig, fname, show, save_fig)


def modelExperimentation(show_plot=True, save_fig=False):
    X_vals = X[feature_cols].values
    y_vals = y.values

    experiments = [
        # ── Architecture 1: Logistic Regression ──────────────────────
        ("LR-1: Logistic Regression (C=1)",
         LogisticRegression(max_iter=1000, random_state=42), True),
        ("LR-2: Logistic Regression (C=0.1, balanced)",
         LogisticRegression(C=0.1, class_weight="balanced",
                            max_iter=1000, random_state=42), True),

        # ── Architecture 2: Random Forest ────────────────────────────
        ("RF-1: Random Forest (n=100, default)",
         RandomForestClassifier(n_estimators=100, random_state=42), False),
        ("RF-2: Random Forest (n=300, max_depth=20)",
         RandomForestClassifier(n_estimators=300, max_depth=20,
                                random_state=42), False),
        ("RF-3: Random Forest (n=200, balanced weights)",
         RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42), False),
        ("RF-4: Random Forest (n=300, max_depth=15, min_samples_leaf=2)",
         RandomForestClassifier(n_estimators=300, max_depth=15,
                                min_samples_leaf=2, random_state=42), False),

        # ── Architecture 3: HistGradientBoosting ─────────────────────
        ("HGB-1: HistGradientBoosting (default)",
         HistGradientBoostingClassifier(random_state=42), False),
        ("HGB-2: HistGradientBoosting (lr=0.05, max_iter=300)",
         HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                        random_state=42), False),
        ("HGB-3: HistGradientBoosting (max_depth=6, balanced)",
         HistGradientBoostingClassifier(max_depth=6, class_weight="balanced",
                                        random_state=42), False),

        # ── Architecture 4: MLP ───────────────────────────────────────
        ("MLP-1: MLP (100,)",
         MLPClassifier(hidden_layer_sizes=(100,), max_iter=300,
                       random_state=42), True),
        ("MLP-2: MLP (256, 128)",
         MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                       random_state=42), True),
        ("MLP-3: MLP (256, 128, 64)",
         MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                       random_state=42), True),
    ]

    results = []
    best_f1, best_true, best_pred, best_name = 0.0, None, None, ""

    for name, clf, use_scaler in experiments:
        f1, oof_true, oof_pred = run_experiment(name, clf, X_vals, y_vals,
                                                use_scaler=use_scaler)
        results.append((name, f1))
        if f1 > best_f1:
            best_f1, best_true, best_pred, best_name = f1, oof_true, oof_pred, name

    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY  (sorted by macro-F1)")
    for name, f1 in sorted(results, key=lambda x: -x[1]):
        print(f"  {f1:.4f}  {name}")
    print(f"\n  Best: {best_name}  ({best_f1:.4f})")

    if best_true is not None:
        plotConfusionMatrix(
            best_true, best_pred,
            title=f"Confusion Matrix — {best_name}",
            show=show_plot, save_fig=save_fig,
            fname="img/08_confusion_matrix_best",
        )


#=======================================
#
# Section 5 — Outlier Detection
#
#=======================================

def outlierDetection(show_plot=True, save_fig=False):
    print("\n── Outlier Detection ────────────────────────────────────────────────")

    X_vals     = X[feature_cols].values
    X_out_vals = X_out[feature_cols].values

    # Scale — fit on training data only
    scaler_od = StandardScaler()
    X_scaled     = scaler_od.fit_transform(X_vals)
    X_out_scaled = scaler_od.transform(X_out_vals)

    # ── 1. Select n_components by BIC ──────────────────────────────────
    print("  Selecting GMM components by BIC …")
    best_bic, best_n = np.inf, 1
    for n in range(1, 11):
        gmm = GaussianMixture(n_components=n, random_state=42,
                              max_iter=300, n_init=3)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        print(f"    n={n:2d}  BIC={bic:,.1f}")
        if bic < best_bic:
            best_bic, best_n = bic, n
    print(f"  → Best n_components = {best_n}  (BIC = {best_bic:,.1f})")

    # ── 2. Fit final GMM ───────────────────────────────────────────────
    gmm_final = GaussianMixture(n_components=best_n, random_state=42,
                                max_iter=300, n_init=5)
    gmm_final.fit(X_scaled)

    # ── 3. Compute log-probabilities ───────────────────────────────────
    log_probs_train = gmm_final.score_samples(X_scaled)
    log_probs_out   = gmm_final.score_samples(X_out_scaled)

    # ── 4. Threshold = max log-prob of known outliers ──────────────────
    threshold = np.max(log_probs_out)
    is_outlier_train = (log_probs_train < threshold).astype(int)
    n_flagged = is_outlier_train.sum()

    print(f"\n  Threshold (max D_out log-prob) : {threshold:.4f}")
    print(f"  Training samples flagged       : {n_flagged} / {len(X_vals)}"
          f"  ({100*n_flagged/len(X_vals):.1f}%)")
    print(f"  D_out correctly flagged        : "
          f"{int((log_probs_out <= threshold).sum())} / {len(log_probs_out)}")

    # ── 5. Plot log-prob distributions ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(log_probs_train, bins=60, density=True, alpha=0.55,
            color="steelblue", label="Training data", edgecolor="none")
    ax.hist(log_probs_out, bins=15, density=True, alpha=0.7,
            color="crimson", label="Known outliers ($D_{out}$)", edgecolor="none")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("GMM log-probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Figure — GMM Log-Probability: Training Data vs Known Outliers",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    executePlot(fig, "img/09_outlier_detection", show_plot, save_fig)

    # ── 6. Filter to D_in and re-train RF-2 ───────────────────────────
    inlier_mask = is_outlier_train == 0
    X_in = X_vals[inlier_mask]
    y_in = y.values[inlier_mask]
    print(f"\n  D_in : {X_in.shape[0]} samples  "
          f"({len(X_vals) - X_in.shape[0]} removed)")

    print("\n── Re-training RF-2 on D_in ─────────────────────────────────────────")
    rf2_in = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    f1_in, _, _ = run_experiment("RF-2 retrained on D_in (inliers only)",
                                 rf2_in, X_in, y_in, use_scaler=False)

    print(f"\n  RF-2 on full D   macro-F1 = 0.4746")
    print(f"  RF-2 on D_in     macro-F1 = {f1_in:.4f}")
    print(f"  Change           {f1_in - 0.4746:+.4f}")

    return gmm_final, scaler_od, threshold


#=======================================
#
# Section 6 — Model Explainability
#
#=======================================

def modelExplainability(show_plot=True, save_fig=False):
    import shap
    import lime
    import lime.lime_tabular

    print("\n── Model Explainability ─────────────────────────────────────────────")

    X_vals = X[feature_cols].values
    y_vals = y.values

    # Fixed 80/20 split — RF-2 trained on train portion, explained on test portion
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_vals, y_vals, test_size=0.2, random_state=42, stratify=y_vals
    )
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    rf.fit(X_tr, y_tr)
    print(f"  RF-2 trained on {len(X_tr)} samples, explaining on {len(X_te)}")

    # ── 1. Permutation Feature Importance ──────────────────────────────
    print("  [1/3] Permutation importance (n_repeats=20) …")
    perm = permutation_importance(rf, X_te, y_te, n_repeats=20,
                                  random_state=42, scoring="f1_macro")
    perm_means = perm.importances_mean
    perm_stds  = perm.importances_std
    sorted_pi  = np.argsort(perm_means)[::-1]

    print("  Permutation importance (high → low):")
    for i in sorted_pi:
        print(f"    {feature_cols[i]:12s}  {perm_means[i]:.4f} ± {perm_stds[i]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(12), perm_means[sorted_pi], yerr=perm_stds[sorted_pi],
           capsize=4, color="steelblue", edgecolor="white")
    ax.set_xticks(range(12))
    ax.set_xticklabels([feature_cols[i] for i in sorted_pi],
                       rotation=45, ha="right", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean decrease in macro-F1", fontsize=11)
    ax.set_title("Permutation Feature Importance (RF-2)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    executePlot(fig, "img/10_permutation_importance", show_plot, save_fig)

    # ── 2. SHAP ────────────────────────────────────────────────────────
    print("  [2/3] SHAP values …")
    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_te), size=min(300, len(X_te)), replace=False)
    X_shap     = X_te[sample_idx]

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)   # list of 5 arrays (n_samples, n_features)

    # Mean |SHAP| aggregated across all classes
    if isinstance(shap_values, list):
        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        sv_class1 = shap_values[1]
    else:
        shap_importance = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 \
                          else np.abs(shap_values).mean(axis=0)
        sv_class1 = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    sorted_shap = np.argsort(shap_importance)[::-1]
    print("  Mean |SHAP| per feature (averaged across classes):")
    for i in sorted_shap:
        print(f"    {feature_cols[i]:12s}  {shap_importance[i]:.4f}")

    # Bar chart (custom, to work with executePlot)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(12), shap_importance[sorted_shap], color="steelblue", edgecolor="white")
    ax.set_xticks(range(12))
    ax.set_xticklabels([feature_cols[i] for i in sorted_shap],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean |SHAP value| (across all classes)", fontsize=11)
    ax.set_title("SHAP Feature Importance (RF-2, mean across classes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    executePlot(fig, "img/11_shap_bar", show_plot, save_fig)

    # Beeswarm for Class 1 (majority class)
    shap.summary_plot(sv_class1, X_shap, feature_names=feature_cols, show=False)
    plt.title("SHAP Beeswarm (RF-2, Class 1)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_fig: plt.savefig("img/11b_shap_beeswarm.png", bbox_inches="tight")
    if show_plot: plt.show()
    plt.close()

    # ── 3. LIME ────────────────────────────────────────────────────────
    print("  [3/3] LIME (50 instances) …")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_tr,
        feature_names=feature_cols,
        class_names=[f"Class {i}" for i in range(5)],
        mode="classification",
        random_state=42,
    )

    n_lime       = 50
    lime_idx     = rng.choice(len(X_te), size=n_lime, replace=False)
    lime_total   = np.zeros(12)

    for idx in lime_idx:
        pred_class = int(rf.predict(X_te[[idx]])[0])
        exp = lime_explainer.explain_instance(
            X_te[idx], rf.predict_proba,
            num_features=12, labels=[pred_class],
        )
        for feat_i, weight in exp.as_map()[pred_class]:
            lime_total[feat_i] += abs(weight)

    lime_importance = lime_total / n_lime
    sorted_lime     = np.argsort(lime_importance)[::-1]
    print(f"  Mean |LIME weight| per feature (averaged over {n_lime} instances):")
    for i in sorted_lime:
        print(f"    {feature_cols[i]:12s}  {lime_importance[i]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(12), lime_importance[sorted_lime], color="steelblue", edgecolor="white")
    ax.set_xticks(range(12))
    ax.set_xticklabels([feature_cols[i] for i in sorted_lime],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(f"Mean |LIME weight| ({n_lime} instances)", fontsize=11)
    ax.set_title("LIME Feature Importance (RF-2, averaged over 50 test instances)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    executePlot(fig, "img/12_lime_importance", show_plot, save_fig)

    return perm_means, shap_importance, lime_importance


def dataAnalysis(show_plot:bool = True, save_fig:bool = False):
    
    printStatistics()
    handlePlots(show=show_plot, save_fig=save_fig)



    return

def trainKNN(X_train, y_train, k=8):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_scaled, y_train)
    
    return clf, scaler

def trainFinalModels():
    """Train RF-2 classifier and GMM outlier detector on all training data."""
    X_vals = X[feature_cols].values
    y_vals = y.values
    X_out_vals = X_out[feature_cols].values

    # RF-2 does not use scaling (consistent with experiment config)
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    clf.fit(X_vals, y_vals)

    # GMM outlier detector — scaler fit on training data only
    scaler_od = StandardScaler()
    X_od_scaled = scaler_od.fit_transform(X_vals)
    X_out_scaled = scaler_od.transform(X_out_vals)

    best_bic, best_n = np.inf, 1
    for n in range(1, 11):
        gmm = GaussianMixture(n_components=n, random_state=42, max_iter=300, n_init=3)
        gmm.fit(X_od_scaled)
        bic = gmm.bic(X_od_scaled)
        if bic < best_bic:
            best_bic, best_n = bic, n

    gmm_final = GaussianMixture(n_components=best_n, random_state=42, max_iter=300, n_init=5)
    gmm_final.fit(X_od_scaled)
    threshold = np.max(gmm_final.score_samples(X_out_scaled))

    print(f"  Final model: RF-2 (n=300, max_depth=20), GMM n_components={best_n}, threshold={threshold:.4f}")
    return clf, gmm_final, scaler_od, threshold


def predict(X_test, clf, gmm, scaler_od, threshold):
    X_test_vals = X_test[feature_cols].values
    labels = clf.predict(X_test_vals)
    X_test_scaled = scaler_od.transform(X_test_vals)
    outliers = (gmm.score_samples(X_test_scaled) < threshold).astype(int)
    return labels, outliers


def generate_submission(test_data, clf, gmm, scaler_od, threshold):
    label_predictions, outlier_predictions = predict(test_data, clf, gmm, scaler_od, threshold)

    # IMPORTANT: stick to this format for the submission,
    # otherwise your submission will results in an error
    submission_df = pd.DataFrame(
        {
            "id": test_data["id"],
            "label": label_predictions,
            "outlier": outlier_predictions,
        }
    )
    return submission_df


def main():
    dataAnalysis(False, False)
    best_k = findBestK()
    testKNN(k=best_k)
    modelExperimentation(show_plot=False, save_fig=True)
    outlierDetection(show_plot=False, save_fig=True)
    modelExplainability(show_plot=False, save_fig=True)

    print("\n── Generating Submissions ───────────────────────────────────────────")
    clf, gmm, scaler_od, threshold = trainFinalModels()

    df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df = generate_submission(df_leaderboard, clf, gmm, scaler_od, threshold)
    # IMPORTANT: The submission file must be named "submission_leaderboard_GroupName.csv",
    # replace GroupName with a group name of your choice. If you do not provide a group name,
    # your submission will fail!
    submission_df.to_csv("submission_leaderboard_noClueForName.csv", index=False)
    print("  Leaderboard submission written → submission_leaderboard_noClueForName.csv")

    # For the final leaderboard, change the file name to "submission_final_GroupName.csv"
    df_final = pd.read_csv("D_test_final.csv")
    submission_df = generate_submission(df_final, clf, gmm, scaler_od, threshold)
    submission_df.to_csv("submission_final_noClueForName.csv", index=False)
    print("  Final submission written       → submission_final_noClueForName.csv")

    


if __name__ == "__main__":
    main()
