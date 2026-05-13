#!/usr/bin/env python3
"""
主流程：加载数据 → 三种 VaR 方法计算 → 对比分析 → 可视化
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# 添加项目根目录到 path
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_DIR)

from data.eda import load_data
from models.historical_var import compute_all as hist_compute, rolling_historical_var
from models.parametric_var import compute_all as param_compute, normality_tests
from models.garch_var import compute_all as garch_compute

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


def run_all_methods():
    """对每个标的运行三种 VaR 方法，输出汇总表"""
    all_results = []

    for symbol, name in tickers.items():
        print(f"\n{'=' * 50}")
        print(f"{name}（{symbol}）")
        print('=' * 50)

        df = load_data(symbol)
        returns = df["log_return"].dropna()

        # ---- 历史模拟法 ----
        hist = hist_compute(returns)
        hist["标的"] = name
        hist["方法"] = "历史模拟法"
        all_results.append(hist)

        print(f"\n[历史模拟法]")
        print(hist.to_string(float_format="%.6f", index=False))

        # ---- 参数法 ----
        param = param_compute(returns)
        param["标的"] = name
        param["方法"] = "参数法"
        all_results.append(param)

        print(f"\n[参数法]")
        print(param.to_string(float_format="%.6f", index=False))

        # ---- 正态性检验 ----
        nt = normality_tests(returns)
        jb_reject = "拒绝" if nt["JB p 值"] < 0.05 else "不拒绝"
        print(f"  正态性检验：JB p={nt['JB p 值']:.6f} → {jb_reject}正态假设")
        print(f"  偏度={nt['偏度']:.3f}，超额峰度={nt['峰度（超额）']:.2f}")

        # ---- GARCH ----
        try:
            garch_df, _ = garch_compute(returns, dist="normal")
            garch_df["标的"] = name
            garch_df["方法"] = "GARCH 法"
            all_results.append(garch_df)

            print(f"\n[GARCH 法（正态残差）]")
            print(garch_df.to_string(float_format="%.6f", index=False))
        except Exception as e:
            print(f"\n[GARCH 法] 失败: {e}")

    # 合并所有结果
    final = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(OUT_DIR, "var_comparison_summary.csv")
    final.to_csv(summary_path, encoding="utf-8-sig", index=False)
    print(f"\n\n汇总表已保存: {summary_path}")
    return final


def plot_var_comparison():
    """图：四种标的 × 三种方法的 95% VaR 对比柱状图"""
    df = load_data("000001")
    returns = df["log_return"].dropna()

    labels = list(tickers.values())
    hist_vars = []
    param_vars = []
    garch_vars = []

    for symbol, name in tickers.items():
        df = load_data(symbol)
        r = df["log_return"].dropna()

        # 历史模拟
        h = hist_compute(r)
        hist_vars.append(h.loc[0, "VaR_95%"])

        # 参数法（正态分布，95%）
        p = param_compute(r)
        p_row = p[(p["置信水平"] == "95%") & (p["分布假设"] == "正态分布")]
        param_vars.append(p_row.iloc[0]["VaR"])

        # GARCH（当前动态 VaR，取负值统一符号）
        try:
            g, _ = garch_compute(r, dist="normal")
            g_row = g[g["置信水平"] == "95%"]
            val = g_row.iloc[0]["VaR（当前动态）"]
            garch_vars.append(-val if val > 0 else val)
        except:
            garch_vars.append(np.nan)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, hist_vars, width, label="历史模拟法", color="#4c72b0")
    bars2 = ax.bar(x, param_vars, width, label="参数法（正态）", color="#dd8452")
    bars3 = ax.bar(x + width, garch_vars, width, label="GARCH 法", color="#55a868")

    ax.set_xlabel("标的")
    ax.set_ylabel("95% VaR（负值表示损失）")
    ax.set_title("95% VaR 对比：三种方法 × 四个标的")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()

    # 在柱子上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2%}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -10),
                        textcoords="offset points",
                        ha="center", va="bottom" if height >= 0 else "top",
                        fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_var_comparison_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_rolling_var():
    """图：滚动历史模拟 VaR — 时变风险"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (symbol, name) in enumerate(tickers.items()):
        df = load_data(symbol)
        ret = df["log_return"].dropna()
        var_series, cvar_series = rolling_historical_var(ret, window=252, alpha=0.05)

        valid_len = len(var_series.dropna())
        axes[i].plot(df["date"].iloc[-valid_len:], var_series.dropna(), color=colors[symbol], linewidth=0.7, label="95% VaR")
        axes[i].plot(df["date"].iloc[-valid_len:], cvar_series.dropna(), color="red", linewidth=0.7, alpha=0.6, label="95% CVaR")
        axes[i].set_ylabel(name, fontsize=10)
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("日期")
    fig.suptitle("滚动历史模拟 VaR / CVaR（252日窗口）", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "07_rolling_var.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_garch_vol():
    """图：GARCH 条件波动率 vs 滚动波动率"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (symbol, name) in enumerate(tickers.items()):
        df = load_data(symbol)
        ret = df["log_return"].dropna()

        # 滚动波动率
        roll_vol = ret.rolling(20).std() * np.sqrt(252)
        roll_valid = roll_vol.dropna()

        # GARCH 条件波动率
        from models.garch_var import fit_garch
        res = fit_garch(ret)
        garch_vol = res.conditional_volatility / 100 * np.sqrt(252)

        dates = df["date"].values
        axes[i].plot(dates[-len(roll_valid):], roll_valid, color="gray", linewidth=0.6, alpha=0.7, label="滚动波动率（20日）")
        axes[i].plot(dates[-len(garch_vol):], garch_vol, color=colors[symbol], linewidth=0.8, label="GARCH 条件波动率")
        axes[i].set_ylabel(name, fontsize=10)
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].set_ylim(0, 0.8)
    axes[-1].set_xlabel("日期")
    fig.suptitle("GARCH 条件波动率 vs 滚动波动率（年化）", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "08_garch_vs_rolling_vol.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def main():
    print("=" * 55)
    print("  VaR 项目主流程：三种方法对比分析")
    print("=" * 55)

    # 1. 计算所有结果
    print("\n>>> 阶段一：VaR 计算与对比")
    summary = run_all_methods()

    # 2. 可视化
    print("\n\n>>> 阶段二：可视化")
    print("生成柱状对比图 ...")
    plot_var_comparison()
    print("生成滚动 VaR 图 ...")
    plot_rolling_var()
    print("生成 GARCH 波动率对比图 ...")
    plot_garch_vol()

    print("\n" + "=" * 55)
    print("  全部完成！结果文件：")
    print(f"    - visualization/var_comparison_summary.csv")
    print(f"    - visualization/06_var_comparison_bar.png")
    print(f"    - visualization/07_rolling_var.png")
    print(f"    - visualization/08_garch_vs_rolling_vol.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
