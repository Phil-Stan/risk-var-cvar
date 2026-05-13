#!/usr/bin/env python3
"""
历史模拟法 VaR / CVaR
无分布假设，直接取历史收益率的经验分位数
"""
import numpy as np
import pandas as pd


def historical_var(returns, alpha=0.05):
    """
    历史模拟法 VaR
    Parameters:
        returns: array-like, 收益率序列
        alpha: 显著性水平，默认 0.05（即 95% VaR）
    Returns:
        var: VaR 值（正值表示损失）
    """
    var = np.percentile(returns, alpha * 100)
    return var


def historical_cvar(returns, alpha=0.05):
    """
    历史模拟法 CVaR (Expected Shortfall)
    即尾部损失的均值
    """
    var = historical_var(returns, alpha)
    tail = returns[returns <= var]
    cvar = tail.mean() if len(tail) > 0 else var
    return cvar


def rolling_historical_var(returns, window=252, alpha=0.05):
    """
    滚动历史模拟 VaR —— 观察 VaR 随时间的变化
    """
    var_series = returns.rolling(window=window).apply(
        lambda x: np.percentile(x, alpha * 100), raw=True
    )
    cvar_series = returns.rolling(window=window).apply(
        lambda x: np.mean(x[x <= np.percentile(x, alpha * 100)]) if len(x[x <= np.percentile(x, alpha * 100)]) > 0 else np.percentile(x, alpha * 100),
        raw=True
    )
    return var_series, cvar_series


def compute_all(returns, alphas=[0.05, 0.01]):
    """
    计算所有水平的历史模拟 VaR 和 CVaR
    返回 DataFrame
    """
    results = {}
    for alpha in alphas:
        var = historical_var(returns, alpha)
        cvar = historical_cvar(returns, alpha)
        label = f"{int((1 - alpha) * 100)}%"
        results[f"VaR_{label}"] = [var]
        results[f"CVaR_{label}"] = [cvar]
    return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    test_returns = np.random.normal(0, 0.02, 1000)
    print("历史模拟法测试结果：")
    print(compute_all(test_returns))
