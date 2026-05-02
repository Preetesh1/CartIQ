# backend/src/evaluator.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time


def evaluate_model(model, test_df, model_name, sample_size=1000):
    """
    Full evaluation on held-out test ratings.

    Uses user_idx / product_idx for prediction — same integer
    space as all CartIQ models.

    Metrics:
    RMSE        — penalizes large errors (squared)
    MAE         — average error in star units
    Coverage    — % of test pairs the model can predict
    Within 1★   — % of predictions within 1 star of truth
    """
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    preds, actuals = [], []
    unpredicted = 0
    start = time.time()

    for _, row in sample.iterrows():
        pred = model.predict_rating(
            int(row["user_idx"]), int(row["product_idx"])
        )
        if pred is not None:
            preds.append(pred)
            actuals.append(row["rating"])
        else:
            unpredicted += 1

    elapsed = time.time() - start
    preds = np.array(preds)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mae = np.mean(np.abs(actuals - preds))
    coverage = len(preds) / len(sample)
    within_1_star = np.mean(np.abs(actuals - preds) <= 1.0)
    within_half_star = np.mean(np.abs(actuals - preds) <= 0.5)

    return {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "coverage": round(coverage, 4),
        "within_1_star": round(within_1_star, 4),
        "within_half_star": round(within_half_star, 4),
        "n_predicted": len(preds),
        "n_unpredicted": unpredicted,
        "eval_time_sec": round(elapsed, 2)
    }


def generate_comparison_report(models_dict, test_df, output_dir=None):
    """
    Full comparison report across all CartIQ models.
    Saves model_comparison.png — goes in README.
    """
    print("\n" + "═"*65)
    print("  📊 CARTIQ MODEL COMPARISON REPORT")
    print("═"*65)

    results = []
    for name, model in models_dict.items():
        print(f"\n  Evaluating {name}...")
        metrics = evaluate_model(model, test_df, name)
        results.append(metrics)

    df = pd.DataFrame(results).set_index("model")

    print(f"\n{'─'*65}")
    print(f"  {'Model':<22} {'RMSE':>8} {'MAE':>8} "
          f"{'Coverage':>10} {'±1 Star':>10}")
    print(f"{'─'*65}")

    for _, row in df.iterrows():
        print(f"  {row.name:<22} {row['rmse']:>8.4f} {row['mae']:>8.4f} "
              f"{row['coverage']:>10.1%} {row['within_1_star']:>10.1%}")

    print(f"{'─'*65}")
    best_rmse = df["rmse"].idxmin()
    print(f"\n  🏆 Best RMSE: {best_rmse} ({df.loc[best_rmse, 'rmse']})")
    print(f"{'═'*65}\n")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _plot_comparison(df, output_dir)

    return df


def _plot_comparison(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("CartIQ — Model Comparison (Amazon Electronics)",
                 fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    models = df.index.tolist()
    x = np.arange(len(models))

    for ax, metric, title, ylim in zip(
        axes,
        ["rmse", "mae", "coverage"],
        ["RMSE (lower is better)", "MAE (lower is better)",
         "Coverage % (higher is better)"],
        [(0.7, 1.2), (0.5, 1.0), (80, 105)]
    ):
        values = df[metric] * (100 if metric == "coverage" else 1)
        ax.bar(x, values, color=colors[:len(models)])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylim(ylim)
        for i, v in enumerate(values):
            label = f"{v:.1f}%" if metric == "coverage" else f"{v:.4f}"
            ax.text(i, v + (0.3 if metric == "coverage" else 0.005),
                    label, ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  📊 Chart saved → {path}")
    plt.close()