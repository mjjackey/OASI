import wandb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

api = wandb.Api()

# ========== configs ==========
WANDB_ENTITY = "mjjackey-university-of-southampton"
WANDB_PROJECT = "goemotion_softinfonce"
GROUPS = [
    "norm_ov_neg_b64",
    "improve_ov_b64",
    # "new_ov_b64",
    "supCon64"
    # "soft"
]
# metrics
METRICS = ["test_acc", "test_weighted_f1", "test_macro_f1"]
group_raw_data = dict()  # store raw list of 5 seeds per metric per group

def calc_metrics():
    results = []
    for group_name in GROUPS:
        # filter runs by tag: group:{group_name}
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"tags": {"$in": [group_name]}}
        )
        run_list = list(runs)
        print(f"Number of matched runs: {len(run_list)}")
        metric_vals = {m: [] for m in METRICS}

        for run in runs:
            summary = run.summary
            for m in METRICS:
                if m in summary:
                    val = float(summary[m])
                    metric_vals[m].append(val)
                    # print(f"  {m}: {val:.4f}")
        
        group_raw_data[group_name] = metric_vals
        
        row = {"group": group_name}
        for m in METRICS:
            arr = np.array(metric_vals[m])
            row[f"{m}_mean"] = np.mean(arr)
            row[f"{m}_std"] = np.std(arr, ddof=1) # sample std n-1 (use for paper!)
        results.append(row)
        df = pd.DataFrame(results)
        print("\n==== Final Result Table ====")
        print(df.round(4))

def run_significance(g1_data, g2_data, g3_data, metric_name):
    print(f"\n===== Significance Test: {metric_name} =====")
    g1 = np.array(g1_data)
    g2 = np.array(g2_data)
    g3 = np.array(g3_data)

    # One-way ANOVA
    f_stat, p_anova = stats.f_oneway(g1, g2, g3)
    print(f"One-way ANOVA: F = {f_stat:.4f}, p-value = {p_anova:.4f}")

    # Tukey HSD
    all_data = np.concatenate([g1, g2, g3])
    group_labels = ["Ov-NegOnly"] * 5 + ["Ov-WholeDen"] * 5 + ["Soft-InfoNCE"] * 5
    tukey_result = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)
    print("\nTukey HSD test:")
    print(tukey_result)

    # Pairwise t-test with Bonferroni correction (3 comparisons)
    n_compare = 3
    alpha_bonf = 0.05 / n_compare
    print(f"\nBonferroni corrected α = {alpha_bonf:.4f}")
    t12, p12 = stats.ttest_ind(g1, g2)
    t13, p13 = stats.ttest_ind(g1, g3)
    t23, p23 = stats.ttest_ind(g2, g3)

    print(f"Ov-NegOnly vs Ov-WholeDen: t={t12:.4f}, p={p12:.4f}, significant={p12 < alpha_bonf}")
    print(f"Ov-NegOnly vs Soft-InfoNCE: t={t13:.4f}, p={p13:.4f}, significant={p13 < alpha_bonf}")
    print(f"Ov-WholeDen vs Soft-InfoNCE: t={t23:.4f}, p={p23:.4f}, significant={p23 < alpha_bonf}")
    return p_anova

def calc_sign():
    # Extract raw arrays for each metric
    g1 = group_raw_data["ov_neg_denominator_b4_16"]
    g2 = group_raw_data["ov_whole_deno_b4_16"]
    g3 = group_raw_data["soft"]

    # Run test for all three metrics
    p_acc = run_significance(g1["test_acc"], g2["test_acc"], g3["test_acc"], "Test Accuracy")
    p_wf1 = run_significance(g1["test_weighted_f1"], g2["test_weighted_f1"], g3["test_weighted_f1"], "Test Weighted F1")
    p_mf1 = run_significance(g1["test_macro_f1"], g2["test_macro_f1"], g3["test_macro_f1"], "Test Macro F1")

def calc_mean_std(arr):
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) # sample std n-1 (use for paper!)
    return mean, std

def get_wandb_bayes():
    api = wandb.Api()

    project = "OvSoftInfoNCE-src"
    sweep_id = "nx4p1pn9"
    # sweep_id = "wqqz39ya"

    sweep = api.sweep(f"{WANDB_ENTITY}/{project}/{sweep_id}")
    runs = sweep.runs

    rows = []
    for run in runs:
        # if run.state != "finished":
        #     continue
        rows.append({
            "state": getattr(run, "state", None),
            "lambda_contrast": run.config["lambda_contrast"],
            "tau_ov": run.config["tau_ov"],
            "alpha_ov": run.config["alpha_ov"],
            "composite_score": run.summary.get("composite_score"),
            "high_overlap_f1": run.summary.get("val_high_overlap_f1"),
            "macro_f1": run.summary.get("Eval_macro_f1"),
            "many_f1": run.summary.get("val_many_f1"),
            "medium_f1": run.summary.get("val_medium_f1"),
            "few_f1": run.summary.get("val_few_f1"),
        })

    df = pd.DataFrame(rows)
    print(df["state"].value_counts(dropna=False)) # 统计每种状态数量
    df_sorted = df.sort_values("composite_score", ascending=False,na_position="last")

    print("==== Top10 candidates ====")
    print(df_sorted.head(10))

    df_sorted.to_csv("../output/sweep_all_runs.csv", index=False)

    # get built-in best run
    best_run = sweep.best_run()
    print("\n==== Best run (W&B built-in) ====")
    print(f"lambda={best_run.config['lambda_contrast']}, tau_ov={best_run.config['tau_ov']}, alpha={best_run.config['alpha_ov']}")
    print(f"composite_score: {best_run.summary['composite_score']}")
    print(f"HighOverlap F1: {best_run.summary['val_high_overlap_f1']}")

def check_sweep_bayes():
    project = "OvSoftInfoNCE-src"
    sweep_id = "wqqz39ya"
    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/{project}/{sweep_id}")
    # 获取所有唯一参数组合
    param_set = set()
    for r in sweep.runs:
        cfg = r.config
        key = (cfg["lambda_contrast"], cfg["tau_ov"], cfg["alpha_ov"])
        if all(k is not None for k in key):
            param_set.add(key)
    print(f"Unique hyperparameter candidates count: {len(param_set)}")
    for p in sorted(param_set):
        print(p)


if __name__ == "__main__":
    # calc_metrics()
    # calc_sign()
    # df = pd.DataFrame(results, columns=["Method", "Acc", "Weighted-F1", "Macro-F1"])
    # print("\n==== Final Result Table ====")
    # print(df.to_markdown(index=False))
    get_wandb_bayes()
    # check_sweep_bayes()
