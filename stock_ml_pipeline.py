"""
S&P 500 Stock Market ML Pipeline
==================================
Single-file machine learning pipeline for stock market direction prediction.

Reads ``sp500_stocks.csv`` (produced by collect_data.py), engineers features,
trains three classifiers (Logistic Regression, Random Forest, XGBoost),
evaluates them, generates all figures, and saves trained models.

Requirements:
    pip install -r requirements.txt

Usage:
    python stock_ml_pipeline.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
RAW_CSV = PROJECT_DIR / "sp500_stocks.csv"
OUTPUT_CSV = PROJECT_DIR / "sp500_enriched.csv"
MODEL_DIR = PROJECT_DIR / "models"
IMAGE_DIR = PROJECT_DIR / "images"


# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def enrich_single_ticker(df_ticker: pd.DataFrame) -> pd.DataFrame:
    df = df_ticker.sort_values("date").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]
    volume = df["volume"].astype(float)

    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    df["sma_20"] = close.rolling(window=20).mean()
    df["sma_50"] = close.rolling(window=50).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    df["rsi_14"] = compute_rsi(close, period=14)

    macd_line, signal_line = compute_macd(close, fast=12, slow=26, signal=9)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line

    bb_upper, bb_lower = compute_bollinger(close, period=20, num_std=2.0)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    log_ret = np.log(close / close.shift(1))
    df["rolling_vol_5"] = log_ret.rolling(window=5).std() * np.sqrt(252)
    df["rolling_vol_10"] = log_ret.rolling(window=10).std() * np.sqrt(252)

    df["volume_pct_change"] = volume.pct_change() * 100

    df["high_low_range_pct"] = ((high - low) / close) * 100
    df["open_close_change_pct"] = ((close - opn) / opn) * 100

    df["momentum_3"] = close / close.shift(3) - 1
    df["momentum_5"] = close / close.shift(5) - 1

    df["lag_close_1"] = close.shift(1)
    daily_ret = close.pct_change()
    df["lag_return_1"] = daily_ret.shift(1)
    df["lag_return_3"] = daily_ret.shift(3)
    df["lag_return_5"] = daily_ret.shift(5)

    df["target"] = (close.shift(-1) > close).astype(int)

    df = df.dropna().reset_index(drop=True)
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def build_enriched_dataset() -> pd.DataFrame:
    print("[1/5] FEATURE ENGINEERING")
    print(f"  Loading {RAW_CSV} ...")
    df_raw = pd.read_csv(RAW_CSV)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    print(f"  Raw shape: {df_raw.shape}")

    frames = []
    for ticker in df_raw["ticker"].unique():
        df_t = df_raw[df_raw["ticker"] == ticker].copy()
        df_t = enrich_single_ticker(df_t)
        frames.append(df_t)
        print(f"  {ticker}: {len(df_t)} rows")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Enriched shape: {df.shape}")
    print(f"  Saved to: {OUTPUT_CSV}\n")
    return df


# ============================================================
# 2. PREPROCESSING
# ============================================================

CATEGORICAL_COLS = ["ticker", "sector"]

BASE_COLS = ["open", "high", "low", "close", "volume"]

ENGINEERED_COLS = [
    "daily_return", "log_return",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower",
    "rolling_vol_5", "rolling_vol_10",
    "volume_pct_change",
    "high_low_range_pct", "open_close_change_pct",
    "momentum_3", "momentum_5",
    "lag_close_1", "lag_return_1", "lag_return_3", "lag_return_5",
]

TARGET = "target"


def preprocess(df: pd.DataFrame):
    print("[2/5] PREPROCESSING")

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in BASE_COLS + ENGINEERED_COLS if c in df.columns]
    feature_cols = num_cols + cat_cols

    X = df[feature_cols]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    X_train_np = preprocessor.fit_transform(X_train)
    X_test_np = preprocessor.transform(X_test)

    ohe_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    feature_names = ohe_names + num_cols

    y_train_np = y_train.values.astype(int)
    y_test_np = y_test.values.astype(int)

    print(f"  Train: {X_train_np.shape[0]} samples | Test: {X_test_np.shape[0]} samples")
    print(f"  Features: {X_train_np.shape[1]}\n")

    return X_train_np, X_test_np, y_train_np, y_test_np, preprocessor, feature_names


# ============================================================
# 3. TRAINING
# ============================================================

def train(X_train, y_train, X_test, y_test):
    print("[3/5] TRAINING")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, C=1.0, solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            eval_metric="logloss", n_jobs=-1,
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        tr_acc = model.score(X_train, y_train)
        te_acc = model.score(X_test, y_test)
        trained[name] = model
        print(f"  {name:25s} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")

    print()
    return trained


# ============================================================
# 4. EVALUATION
# ============================================================

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

COLORS = {
    "logistic_regression": "#3b82f6",
    "random_forest": "#10b981",
    "xgboost": "#f59e0b",
}


def evaluate(models, X_test, y_test, feature_names, preprocessor):
    print("[4/5] EVALUATION")

    predictions = {}
    probabilities = {}
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        predictions[name] = y_pred
        probabilities[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        rows.append({"Model": DISPLAY_NAMES[name], "Accuracy": acc, "Precision": prec,
                      "Recall": rec, "F1 Score": f1, "ROC AUC": auc})

    comparison = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    print(comparison.to_string(index=False))
    print()

    for name in models:
        print(f"--- {DISPLAY_NAMES[name]} Classification Report ---")
        print(classification_report(y_test, predictions[name], target_names=["DOWN", "UP"]))
    print()

    return comparison, predictions, probabilities


# ============================================================
# 5. FIGURES
# ============================================================

def save_fig(name: str):
    plt.savefig(IMAGE_DIR / name, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: images/{name}")


def generate_figures(y_test, comparison, predictions, probabilities, models, feature_names, df):
    print("[5/5] GENERATING FIGURES")

    counts = pd.Series(y_test).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["DOWN (0)", "UP (1)"], counts.values, color=["#ef4444", "#10b981"],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, c in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.02,
                str(c), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig("class_distribution.png")

    num_cols = [c for c in feature_names if c in df.columns]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
                    ax=ax, square=True, linewidths=0.3, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_fig("correlation_heatmap.png")

    tree_models = {k: v for k, v in models.items() if k in ["random_forest", "xgboost"]}
    if tree_models:
        fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 8))
        if len(tree_models) == 1:
            axes = [axes]
        for ax, (name, model) in zip(axes, tree_models.items()):
            imp = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_[0])
            n = min(len(imp), len(feature_names))
            imp_df = pd.DataFrame({"Feature": feature_names[:n], "Importance": imp[:n]})
            imp_df = imp_df.sort_values("Importance", ascending=True).tail(20)
            ax.barh(imp_df["Feature"], imp_df["Importance"], color=COLORS[name], edgecolor="white", linewidth=0.5)
            ax.set_title(DISPLAY_NAMES[name], fontsize=12, fontweight="bold")
            ax.set_xlabel("Importance", fontsize=10)
            ax.grid(True, axis="x", alpha=0.3)
        plt.suptitle("Feature Importance (Top 20)", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_fig("feature_importance.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2,
                label=f"{DISPLAY_NAMES[name]} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    save_fig("roc_curve.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in probabilities.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, color=COLORS[name], linewidth=2, label=DISPLAY_NAMES[name])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    save_fig("precision_recall_curve.png")

    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"],
                    cbar=False, linewidths=0.5, linecolor="gray")
        ax.set_title(DISPLAY_NAMES[name], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    x = np.arange(len(comparison))
    width = 0.15
    cmap = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    for i, metric in enumerate(metrics_to_plot):
        vals = comparison[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric, color=cmap[i],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison["Model"].values, fontsize=10)
    ax.legend(loc="lower right", fontsize=9, ncol=5)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig("model_comparison.png")

    for ticker in df["ticker"].unique()[:3]:
        sub = df[df["ticker"] == ticker].sort_values("date").tail(200).copy()
        dates = sub["date"]

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1, 1]})

        ax = axes[0]
        ax.plot(dates, sub["close"], color="#1a1a2e", linewidth=1.5, label="Close")
        if "sma_20" in sub.columns:
            ax.plot(dates, sub["sma_20"], color="#3b82f6", linewidth=1, alpha=0.8, label="SMA 20")
        if "sma_50" in sub.columns:
            ax.plot(dates, sub["sma_50"], color="#ef4444", linewidth=1, alpha=0.8, label="SMA 50")
        ax.set_title(f"{ticker} - Price with SMA 20 & SMA 50", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Price", fontsize=10)

        ax = axes[1]
        if "rsi_14" in sub.columns:
            ax.plot(dates, sub["rsi_14"], color="#8b5cf6", linewidth=1)
            ax.axhline(70, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(30, color="#10b981", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.fill_between(dates, 70, sub["rsi_14"].clip(upper=70), alpha=0.15, color="#ef4444")
            ax.fill_between(dates, 30, sub["rsi_14"].clip(lower=30), alpha=0.15, color="#10b981")
        ax.set_title("RSI (14)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel("RSI", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        if "macd" in sub.columns and "macd_signal" in sub.columns:
            ax.plot(dates, sub["macd"], color="#3b82f6", linewidth=1, label="MACD")
            ax.plot(dates, sub["macd_signal"], color="#ef4444", linewidth=1, label="Signal")
            hist = sub["macd"] - sub["macd_signal"]
            cols = ["#10b981" if v >= 0 else "#ef4444" for v in hist]
            ax.bar(dates, hist, color=cols, alpha=0.6, width=1.5)
        ax.set_title("MACD", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("MACD", fontsize=10)

        ax = axes[3]
        if "bb_upper" in sub.columns and "bb_lower" in sub.columns:
            ax.plot(dates, sub["close"], color="#1a1a2e", linewidth=1.2, label="Close")
            ax.plot(dates, sub["bb_upper"], color="#f59e0b", linewidth=0.8, linestyle="--", label="BB Upper")
            ax.plot(dates, sub["bb_lower"], color="#f59e0b", linewidth=0.8, linestyle="--", label="BB Lower")
            ax.fill_between(dates, sub["bb_upper"], sub["bb_lower"], alpha=0.1, color="#f59e0b")
        ax.set_title("Bollinger Bands", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Price", fontsize=10)

        plt.tight_layout()
        safe = ticker.replace(".", "_")
        save_fig(f"stock_chart_{safe}.png")

    print()


# ============================================================
# MAIN
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    df = build_enriched_dataset()

    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess(df)

    models = train(X_train, y_train, X_test, y_test)

    comparison, predictions, probabilities = evaluate(models, X_test, y_test, feature_names, preprocessor)

    best_model_name = comparison.loc[0, "Model"]
    print(f"  Best model: {best_model_name} (F1={comparison.loc[0, 'F1 Score']:.4f})\n")

    comparison.to_csv(PROJECT_DIR / "model_comparison.csv", index=False)

    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")
    print(f"  Models saved to: {MODEL_DIR}/\n")

    generate_figures(y_test, comparison, predictions, probabilities, models, feature_names, df)

    print("=" * 50)
    print("  PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Enriched CSV    : {OUTPUT_CSV}")
    print(f"  Models          : {MODEL_DIR}/")
    print(f"  Figures         : {IMAGE_DIR}/")
    print(f"  Comparison CSV  : {PROJECT_DIR / 'model_comparison.csv'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
