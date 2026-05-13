#!/usr/bin/env python3
"""
GARCH(1,1) 动态 VaR / CVaR
使用时变条件波动率计算动态 VaR，捕捉波动聚集效应
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats


def fit_garch(returns, p=1, q=1, dist="normal"):
    """
    拟合 GARCH(p,q) 模型
    Parameters:
        returns: 收益率序列 (Series 或 array)
        p: GARCH 阶数
        q: ARCH 阶数
        dist: 残差分布 ("normal" 或 "t" 或 "skewt")
    Returns:
        fitted model
    """
    model = arch_model(returns * 100, vol="Garch", p=p, q=q, dist=dist, mean="Zero")
    # 用 Zero mean 因为金融收益率均值通常不显著
    # 乘以100是为了数值稳定性（收益率在0.01量级，ARCH模型对此敏感）
    res = model.fit(disp="off")
    return res


def garch_var(returns, alpha=0.05, p=1, q=1, dist="normal"):
    """
    GARCH 动态 VaR
    对每一天 t，VaR_t = sigma_t * z_alpha
    （假设均值为0，因为金融日收益率均值通常可忽略）
    返回整个系列的 VaR 序列
    """
    res = fit_garch(returns, p, q, dist)

    # 条件波动率（原始尺度，%
    cond_vol = res.conditional_volatility / 100
    # 标准化残差
    std_resid = res.resid / cond_vol.values
    std_resid = std_resid.dropna()

    # 用标准化残差的经验分位数（避免分布假设误差）
    z_alpha = np.percentile(std_resid, alpha * 100)

    # 或使用理论分位数
    if dist == "normal":
        z_alpha_theory = stats.norm.ppf(alpha)
    elif dist == "t":
        # 从拟合结果获取自由度
        df = res.params.get("nu", 5)
        z_alpha_theory = stats.t.ppf(alpha, df)
    else:
        z_alpha_theory = stats.norm.ppf(alpha)

    # 默认使用理论分位数，更符合 GARCH 模型设定
    var_series = -z_alpha_theory * cond_vol

    return var_series, res


def garch_cvar(returns, alpha=0.05, p=1, q=1, dist="normal"):
    """
    GARCH 动态 CVaR
    """
    res = fit_garch(returns, p, q, dist)
    cond_vol = res.conditional_volatility / 100

    if dist == "normal":
        z_alpha = stats.norm.ppf(alpha)
        cvar_factor = -stats.norm.pdf(z_alpha) / alpha
    elif dist == "t":
        df = res.params.get("nu", 5)
        t_alpha = stats.t.ppf(alpha, df)
        numerator = stats.t.pdf(t_alpha, df) * (df + t_alpha ** 2) / (df - 1)
        cvar_factor = -numerator / alpha
    else:
        z_alpha = stats.norm.ppf(alpha)
        cvar_factor = -stats.norm.pdf(z_alpha) / alpha

    cvar_series = cvar_factor * cond_vol
    return cvar_series, res


def garch_static_var(returns, alpha=0.05, p=1, q=1, dist="normal"):
    """
    用 GARCH 模型的 unconditional volatility 计算静态 VaR
    便于和其他方法对比
    """
    res = fit_garch(returns, p, q, dist)
    # 无条件波动率: omega / (1 - alpha - beta) 的平方根
    omega = res.params["omega"]
    a = res.params.get("alpha[1]", 0)
    b = res.params.get("beta[1]", 0)
    uncond_var = np.sqrt(omega / (1 - a - b)) / 100 if (1 - a - b) > 0 else np.std(returns, ddof=1)

    if dist == "normal":
        z_alpha = stats.norm.ppf(alpha)
    elif dist == "t":
        df = res.params.get("nu", 5)
        z_alpha = stats.t.ppf(alpha, df)
    else:
        z_alpha = stats.norm.ppf(alpha)

    var = -z_alpha * uncond_var
    return var, uncond_var


def compute_all(returns, alphas=[0.05, 0.01], dist="normal"):
    """
    计算 GARCH VaR/CVaR，返回汇总 DataFrame
    """
    rows = []
    for alpha in alphas:
        label = f"{int((1 - alpha) * 100)}%"

        # 动态（序列的最后一个值作为当前 VaR）
        var_series, res = garch_var(returns, alpha, dist=dist)
        cvar_series, _ = garch_cvar(returns, alpha, dist=dist)

        # 无条件（静态）VaR
        static_var, _ = garch_static_var(returns, alpha, dist=dist)

        dist_name = "正态" if dist == "normal" else "t 分布"
        rows.append({
            "置信水平": label,
            "残差分布": dist_name,
            "VaR（无条件）": static_var,
            "VaR（当前动态）": var_series.iloc[-1],
            "CVaR（当前动态）": cvar_series.iloc[-1],
            "VaR 均值（全期）": var_series.mean(),
            "VaR 最大值（全期）": var_series.max(),
            "VaR 最小值（全期）": var_series.min(),
        })
    return pd.DataFrame(rows), res


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    n = 1000
    # 模拟 GARCH 效应
    sigma = np.zeros(n)
    returns = np.zeros(n)
    sigma[0] = 0.02
    for t in range(1, n):
        sigma[t] = np.sqrt(0.00001 + 0.1 * returns[t-1]**2 + 0.85 * sigma[t-1]**2)
        returns[t] = sigma[t] * np.random.normal(0, 1)

    print("GARCH VaR 测试：")
    df, res = compute_all(pd.Series(returns))
    print(df.to_string(float_format="%.6f"))
    print(f"\n模型参数：\n{res.params}")
