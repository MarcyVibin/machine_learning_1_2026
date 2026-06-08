import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
import joblib
import warnings
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")
#TODO: Order imports for better looks

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

#classifier and scaler definitions
_clf = None
_scaler = None



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

def testKNN():
    print("\n── KNN Test ─────────────────────────────────────────────────────────")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_vals = X[feature_cols].values
    y_vals = y.values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_vals, y_vals)):
        X_train, X_val = X_vals[train_idx], X_vals[val_idx]
        y_train, y_val = y_vals[train_idx], y_vals[val_idx]

        clf, scaler = trainKNN(X_train, y_train)
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

    best_k, best_score = None, 0.0
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


def predict(X_test):
    global _clf, _scaler

    # Train once and reuse
    if _clf is None or _scaler is None:
        feature_cols_test = [c for c in X_test.columns if c != "id"]
        _clf, _scaler = trainKNN(X[feature_cols].values, y.values)

    X_scaled = _scaler.transform(X_test[[c for c in X_test.columns if c != "id"]].values)
    labels   = _clf.predict(X_scaled)
    outliers = np.zeros(len(X_test), dtype=int)  #TODO: placeholder until outlier detection is implemented

    return labels, outliers

def generate_submission(test_data):
    label_predictions, outlier_predictions = predict(test_data)
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
    testKNN()
    findBestK()

    '''df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df = generate_submission(df_leaderboard)
    # IMPORTANT: The submission file must be named "submission_leaderboard_GroupName.csv",
    # replace GroupName with a group name of your choice. If you do not provide a group name,
    # your submission will fail!
    submission_df.to_csv("submission_leaderboard_GroupName.csv", index=False)
    '''
    
    # For the final leaderboard, change the file name to "submission_final_GroupName.csv"
    df_final = pd.read_csv("D_test_final.csv")
    submission_df = generate_submission(df_final)
    submission_df.to_csv("submission_final_GroupName.csv", index=False)


if __name__ == "__main__":
    main()
