#!/usr/bin/env python3
"""
参数法 VaR / CVaR
假设收益率服从正态分布或 t 分布，估计参数后反算 VaR
"""
import numpy as np
import pandas as pd
from scipy import stats


def _fit_distribution(returns, dist="norm"):
    """拟合分布参数"""
    if dist == "norm":
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        return mu, sigma
    elif dist == "t":
        # t 分布有三个参数：df, loc, scale
        params = stats.t.fit(returns)
        return params  # (df, loc, scale)
    else:
        raise ValueError(f"不支持的分布: {dist}")


def parametric_var(returns, alpha=0.05, dist="norm"):
    """
    参数法 VaR
    Parameters:
        returns: 收益率序列
        alpha: 显著性水平
        dist: "norm" 或 "t"
    """
    if dist == "norm":
        mu, sigma = _fit_distribution(returns, "norm")
        var = mu + sigma * stats.norm.ppf(alpha)
    elif dist == "t":
        df, loc, scale = _fit_distribution(returns, "t")
        var = loc + scale * stats.t.ppf(alpha, df)
    else:
        raise ValueError(f"不支持的分布: {dist}")
    return var


def parametric_cvar(returns, alpha=0.05, dist="norm"):
    """
    参数法 CVaR (Expected Shortfall)
    对正态分布: CVaR = mu - sigma * phi(z_alpha) / alpha
    对 t 分布: 使用数值积分或解析公式
    """
    if dist == "norm":
        mu, sigma = _fit_distribution(returns, "norm")
        z_alpha = stats.norm.ppf(alpha)
        cvar = mu - sigma * stats.norm.pdf(z_alpha) / alpha
    elif dist == "t":
        df, loc, scale = _fit_distribution(returns, "t")
        t_alpha = stats.t.ppf(alpha, df)
        # t 分布 CVaR 解析公式
        # 参考: https://en.wikipedia.org/wiki/Expected_shortfall#Student's_t_distribution
        tau = t_alpha
        numerator = stats.t.pdf(tau, df) * (df + tau ** 2) / (df - 1)
        denominator = alpha
        cvar = loc - scale * numerator / denominator
        # 当 df <= 1 时 CVaR 不存在，fallback
        if df <= 1:
            cvar = np.nan
    else:
        raise ValueError(f"不支持的分布: {dist}")
    return cvar


def normality_tests(returns):
    """正态性检验：JB 检验 + Shapiro-Wilk"""
    from scipy.stats import jarque_bera, shapiro, normaltest

    jb_stat, jb_p = jarque_bera(returns)
    sw_stat, sw_p = shapiro(returns)

    results = {
        "JB 统计量": jb_stat,
        "JB p 值": jb_p,
        "Shapiro-Wilk 统计量": sw_stat,
        "Shapiro-Wilk p 值": sw_p,
        "偏度": stats.skew(returns),
        "峰度（超额）": stats.kurtosis(returns, fisher=True),
    }
    return results


def compute_all(returns, alphas=[0.05, 0.01]):
    """
    计算所有水平的参数法 VaR/CVaR（正态 + t分布）
    """
    rows = []
    for alpha in alphas:
        label = f"{int((1 - alpha) * 100)}%"
        for dist in ["norm", "t"]:
            dist_name = "正态分布" if dist == "norm" else "t 分布"
            var = parametric_var(returns, alpha, dist)
            cvar = parametric_cvar(returns, alpha, dist)
            rows.append({
                "置信水平": label,
                "分布假设": dist_name,
                "VaR": var,
                "CVaR": cvar,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    test_returns = np.random.standard_t(df=4, size=1000) * 0.02
    print("正态性检验：")
    nt = normality_tests(test_returns)
    for k, v in nt.items():
        print(f"  {k}: {v:.6f}")
    print("\n参数法 VaR/CVaR：")
    print(compute_all(test_returns))
