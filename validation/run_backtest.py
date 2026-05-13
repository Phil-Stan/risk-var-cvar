#!/usr/bin/env python3
"""
回测验证：检验 VaR 模型是否准确
运行：python validation/run_backtest.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)

from data.eda import load_data
from validation.backtest import backtest_historical, backtest_garch, kupiec_test
from models.historical_var import rolling_historical_var

DATA_DIR = os.path.join(PROJ_DIR, "data")
OUT_DIR = os.path.join(PROJ_DIR, "visualization")
os.makedirs(OUT_DIR, exist_ok=True)

tickers = {
    "000001": "上证指数",
    "399006": "创业板指",
    "600036": "招商银行",
    "600519": "贵州茅台",
}
colors = {"000001": "#1f77b4", "399006": "#ff7f0e", "600036": "#2ca02c", "600519": "#d62728"}


def plot_violations(returns, var_series, violations, name, alpha):
    """画出 VaR 与违反点"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(returns.index, returns.values, color="gray", linewidth=0.5, alpha=0.6, label="日收益率")
    ax.plot(var_series.index, var_series.values, color="red", linewidth=0.8, label=f"VaR ({int((1-alpha)*100)}%)")
    # 标记违反点
    violation_dates = returns.index[violations.values]
    violation_vals = returns.values[violations.values]
    ax.scatter(violation_dates, violation_vals, color="red", s=20, zorder=5, label=f"违反 ({violations.sum()} 次)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"{name}  历史模拟法 VaR 回测（窗口=500天）")
    ax.set_ylabel("收益率 / VaR")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"09_backtest_{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_backtest_comparison(all_summaries):
    """所有标的的回测对比图：实际违反率 vs 期望违反率"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (symbol, name) in enumerate(tickers.items()):
        df = all_summaries[all_summaries["标的"] == name]
        methods = df["方法"].values
        actual = [float(v.strip("%")) if "%" in v else float(v) for v in df["实际违反率"].values]
        expected = [float(v.strip("%")) if "%" in v else float(v) for v in df["期望违反率"].values]

        x = np.arange(len(methods))
        width = 0.3
        axes[i].bar(x - width / 2, actual, width, color=colors[symbol], alpha=0.7, label="实际违反率")
        axes[i].bar(x + width / 2, expected, width, color="gray", alpha=0.5, label="期望违反率（α）")
        axes[i].axhline(y=expected[0], color="red", linestyle="--", linewidth=0.8)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(methods, fontsize=9)
        axes[i].set_title(name)
        axes[i].legend(fontsize=8)

    fig.suptitle("VaR 回测：实际违反率 vs 期望违反率", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "10_backtest_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main():
    print("=" * 50)
    print("  VaR 回测验证")
    print("=" * 50)

    all_results = []

    for symbol, name in tickers.items():
        print(f"\n--- {name} ---")
        df = load_data(symbol)
        ret = df["log_return"].dropna()

        # ---- 历史模拟法回测 ----
        print("  运行历史模拟法回测...", end=" ", flush=True)
        bt_hist = backtest_historical(ret, window=500, alphas=[0.05, 0.01])

        for label, res in bt_hist.items():
            print(f"\n    VaR {label}：违反 {res['违反次数']}/{res['回测样本量']} "
                  f"= {res['实际违反率']:.4f}（期望 {res['期望违反率']}）"
                  f"   LR={res['LR 统计量']:.4f}  p={res['p 值']:.4f}  {res['检验结论']}")
            all_results.append({
                "标的": name,
                "方法": "历史模拟法",
                "VaR": label,
                "样本量": res["回测样本量"],
                "违反次数": res["违反次数"],
                "实际违反率": f"{res['实际违反率']:.4f}",
                "期望违反率": f"{res['期望违反率']:.2f}",
                "LR 统计量": f"{res['LR 统计量']:.4f}",
                "p 值": f"{res['p 值']:.4f}",
                "结论": res["检验结论"],
            })

        # 生成违反点可视化（95% VaR）
        print("  生成违反点可视化...", end=" ", flush=True)
        var_hist_95 = bt_hist["95%"]
        plot_violations(ret, var_hist_95["VaR 序列"], var_hist_95["违反序列"], name, 0.05)
        print("完成")

        # ---- GARCH 回测（只对第一个标的做，其余跳过以免太慢）----
        if symbol == "000001":
            print("  运行 GARCH 法回测（此步较慢，正在拟合...）", end=" ", flush=True)
            try:
                bt_garch = backtest_garch(ret, window=500, alphas=[0.05, 0.01], dist="normal")
                for label, res in bt_garch.items():
                    print(f"\n    VaR {label}（GARCH）：违反 {res['违反次数']}/{res['回测样本量']} "
                          f"= {res['实际违反率']:.4f}（期望 {res['期望违反率']}）"
                          f"   LR={res['LR 统计量']:.4f}  p={res['p 值']:.4f}  {res['检验结论']}")
                    all_results.append({
                        "标的": name,
                        "方法": f"GARCH 法",
                        "VaR": label,
                        "样本量": res["回测样本量"],
                        "违反次数": res["违反次数"],
                        "实际违反率": f"{res['实际违反率']:.4f}",
                        "期望违反率": f"{res['期望违反率']:.2f}",
                        "LR 统计量": f"{res['LR 统计量']:.4f}",
                        "p 值": f"{res['p 值']:.4f}",
                        "结论": res["检验结论"],
                    })
                print("  GARCH 回测完成")
            except Exception as e:
                print(f"  GARCH 回测失败: {e}")

    # 汇总保存
    summary = pd.DataFrame(all_results)
    summary_path = os.path.join(OUT_DIR, "backtest_summary.csv")
    summary.to_csv(summary_path, encoding="utf-8-sig", index=False)
    print(f"\n\n回测汇总已保存: {summary_path}")

    # 对比图
    print("生成回测对比图...")
    plot_backtest_comparison(summary)
    print("完成。")


if __name__ == "__main__":
    main()
