#!/usr/bin/env python3
"""
EDA —— 探索性数据分析
对四个标的进行收益率分布、波动聚集、正态性检验等分析
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "..", "visualization")
os.makedirs(OUT_DIR, exist_ok=True)

tickers = {
    "000001": "上证指数",
    "399006": "创业板指",
    "600036": "招商银行",
    "600519": "贵州茅台",
}

# 颜色方案
colors = {"000001": "#1f77b4", "399006": "#ff7f0e", "600036": "#2ca02c", "600519": "#d62728"}


def load_data(symbol):
    path = os.path.join(DATA_DIR, symbol, f"{symbol}_daily.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df


def desc_stats(df, name):
    """计算描述性统计"""
    ret = df["log_return"]
    var_95 = np.percentile(ret, 5)
    var_99 = np.percentile(ret, 1)
    cvar_95 = ret[ret <= var_95].mean()

    stats = {
        "样本量": len(ret),
        "均值": ret.mean(),
        "标准差": ret.std(),
        "偏度": ret.skew(),
        "峰度": ret.kurtosis(),  # 超额峰度，正态=0
        "最小值": ret.min(),
        "最大值": ret.max(),
        "95% VaR": var_95,
        "99% VaR": var_99,
        "95% CVaR": cvar_95,
    }
    return stats


def plot_time_series():
    """图1：收益率时序图 —— 观察波动聚集"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (symbol, name) in enumerate(tickers.items()):
        df = load_data(symbol)
        axes[i].plot(df["date"], df["log_return"], color=colors[symbol], linewidth=0.5)
        axes[i].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[i].set_ylabel(name, fontsize=10)
        axes[i].set_ylim(-0.12, 0.12)
    axes[-1].set_xlabel("日期")
    fig.suptitle("日收益率时序（对数收益率）", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_returns_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_histogram():
    """图2：收益率分布直方图 + 正态拟合对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (symbol, name) in enumerate(tickers.items()):
        df = load_data(symbol)
        ret = df["log_return"].values
        mu, sigma = ret.mean(), ret.std()

        axes[i].hist(ret, bins=80, density=True, alpha=0.6, color=colors[symbol],
                      label=f"实际 ({name})")
        x = np.linspace(ret.min(), ret.max(), 500)
        axes[i].plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                     "r-", linewidth=1.5, label="正态拟合")
        import scipy.stats as stats
        axes[i].set_title(f"{name}  偏度={stats.skew(ret):.3f}  峰度={stats.kurtosis(ret, fisher=True):.2f}")
        axes[i].legend(fontsize=8)
    fig.suptitle("收益率分布 vs 正态分布", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_return_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_qq():
    """图3：Q-Q 图 —— 正态性检验可视化"""
    import scipy.stats as stats
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (symbol, name) in enumerate(tickers.items()):
        df = load_data(symbol)
        ret = df["log_return"].values
        stats.probplot(ret, dist="norm", plot=axes[i])
        axes[i].get_lines()[0].set_markersize(2)
        axes[i].get_lines()[0].set_color(colors[symbol])
        axes[i].get_lines()[1].set_color("red")
        axes[i].set_title(f"{name} Q-Q 图")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_qq_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_acf():
    """图4：自相关图 —— 检查收益率/平方收益率的序列依赖"""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for i, (symbol, name) in enumerate(tickers.items()):
        df = load_data(symbol)
        ret = df["log_return"]
        plot_acf(ret, lags=30, ax=axes[i, 0], title=f"{name} 收益率 ACF", zero=False)
        plot_acf(ret ** 2, lags=30, ax=axes[i, 1], title=f"{name} 平方收益率 ACF", zero=False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_acf_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_volatility_cluster():
    """图5：滚动波动率 —— 更直观地看波动聚集"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (symbol, name) in enumerate(tickers.items()):
        df = load_data(symbol)
        df["roll_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)  # 年化
        axes[i].plot(df["date"], df["roll_vol_20"], color=colors[symbol], linewidth=0.8)
        axes[i].set_ylabel(name, fontsize=10)
        axes[i].set_ylim(0, 0.8)
    axes[-1].set_xlabel("日期")
    fig.suptitle("滚动年化波动率（20日窗口）—— 波动聚集现象", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_rolling_volatility.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def main():
    print("=" * 50)
    print("EDA —— 探索性数据分析")
    print("=" * 50)

    # 1. 描述性统计
    print("\n[1/6] 描述性统计")
    stats_rows = []
    for symbol, name in tickers.items():
        df = load_data(symbol)
        stats = desc_stats(df, name)
        stats["标的"] = name
        stats_rows.append(stats)
    stats_df = pd.DataFrame(stats_rows).set_index("标的")
    print(stats_df.to_string(float_format="%.6f"))
    stats_df.to_csv(os.path.join(OUT_DIR, "00_descriptive_stats.csv"), encoding="utf-8-sig")

    # 2. 收益率时序图
    print("\n[2/6] 收益率时序图 ...")
    plot_time_series()

    # 3. 分布直方图
    print("[3/6] 收益率分布直方图 ...")
    plot_histogram()

    # 4. Q-Q 图
    print("[4/6] Q-Q 图 ...")
    plot_qq()

    # 5. 自相关图
    print("[5/6] 自相关图 ...")
    plot_acf()

    # 6. 滚动波动率
    print("[6/6] 滚动波动率 ...")
    plot_volatility_cluster()

    print("\n" + "=" * 50)
    print("EDA 完成，所有图表已保存至 visualization/ 目录")
    print("=" * 50)


if __name__ == "__main__":
    main()
