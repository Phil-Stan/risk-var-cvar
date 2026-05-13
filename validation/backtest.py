#!/usr/bin/env python3
"""
回测验证模块
对 VaR 模型进行后验分析，检验预测是否准确
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def count_violations(returns, var_series):
    """
    计算 VaR 违反次数（实际损失超过 VaR 的天数）
    Parameters:
        returns: 实际收益率序列
        var_series: VaR 阈值序列（负值，如 -0.0148 表示 95% VaR 为 1.48%）
    Returns:
        violations: 布尔序列，True 表示违反（收益率低于 VaR 阈值）
        violation_rate: 违反率
    """
    violations = returns.values < var_series.values
    violation_rate = violations.mean()
    return violations, violation_rate


def kupiec_test(n_violations, n_total, alpha):
    """
    Kupiec 无条件覆盖检验
    H0: 实际失败率 = alpha（模型正确）
    H1: 实际失败率 != alpha
    LR ~ χ²(1)
    """
    p_hat = n_violations / n_total

    if n_violations == 0:
        # 零违反时，LR 统计量需要特殊处理
        lr = -2 * (n_total * np.log(1 - alpha))
        p_value = stats.chi2.sf(lr, 1)
        return lr, p_value, p_hat

    if n_violations == n_total:
        lr = -2 * (n_total * np.log(alpha))
        p_value = stats.chi2.sf(lr, 1)
        return lr, p_value, p_hat

    # 标准 Kupiec LR 统计量
    lr = -2 * (
        (n_total - n_violations) * np.log(1 - alpha) + n_violations * np.log(alpha)
        - (n_total - n_violations) * np.log(1 - p_hat) - n_violations * np.log(p_hat)
    )
    p_value = stats.chi2.sf(lr, 1)
    return lr, p_value, p_hat


def christoffersen_test(violations, alpha):
    """
    Christoffersen 条件覆盖检验
    同时检验失败率和失败事件的独立性
    LR_cc = LR_uc + LR_ind ~ χ²(2)
    """
    n = len(violations)
    n01 = ((~violations.shift(1).fillna(False)) & violations).sum()
    n00 = ((~violations.shift(1).fillna(False)) & ~violations).sum()
    n10 = (violations.shift(1).fillna(False) & violations).sum()
    n11 = (violations.shift(1).fillna(False) & ~violations).sum()

    # 避免除零
    pi_01 = n01 / (n01 + n00) if (n01 + n00) > 0 else 0
    pi_11 = n10 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_2 = (n01 + n10) / (n01 + n00 + n10 + n11)

    # 独立性的 LR 统计量
    if pi_01 > 0 and pi_11 > 0 and pi_2 > 0:
        lr_ind = -2 * (
            np.log((1 - pi_2) ** (n00 + n11) * pi_2 ** (n01 + n10))
            - np.log((1 - pi_01) ** n00 * pi_01 ** n01 * (1 - pi_11) ** n11 * pi_11 ** n10)
        )
    else:
        lr_ind = 0.0

    # 无条件覆盖 LR（复用 kupiec）
    n_violations = violations.sum()
    _, lr_uc, _ = kupiec_test(n_violations, n, alpha)

    lr_cc = lr_uc + lr_ind
    p_value = stats.chi2.sf(lr_cc, 2)
    return lr_cc, p_value, lr_uc, lr_ind


def backtest_historical(returns, window=500, alphas=[0.05, 0.01]):
    """
    滚动回测历史模拟法 VaR
    用前 window 天的数据计算 VaR，预测下一天
    """
    results = {}
    for alpha in alphas:
        n = len(returns)
        var_pred = np.full(n, np.nan)
        violations = np.full(n, False)

        for t in range(window, n):
            hist_data = returns.iloc[t - window: t]
            var_pred[t] = np.percentile(hist_data, alpha * 100)

        violated = returns.values < var_pred
        n_viol = np.nansum(violated)
        n_valid = np.sum(~np.isnan(var_pred))

        label = f"{int((1 - alpha) * 100)}%"
        lr, pv, p_hat = kupiec_test(n_viol, n_valid, alpha)
        results[label] = {
            "VaR 类型": "历史模拟法",
            "显著性水平 α": alpha,
            "回测样本量": n_valid,
            "违反次数": int(n_viol),
            "实际违反率": p_hat,
            "期望违反率": alpha,
            "LR 统计量": lr,
            "p 值": pv,
            "检验结论": "不拒绝" if pv > 0.05 else "拒绝",
            "VaR 序列": pd.Series(var_pred, index=returns.index),
            "违反序列": pd.Series(violated, index=returns.index),
        }
    return results


def backtest_garch(returns, window=500, alphas=[0.05, 0.01], dist="normal"):
    """
    滚动回测 GARCH VaR
    每次滚动重新拟合 GARCH 模型
    """
    from models.garch_var import fit_garch

    results = {}
    for alpha in alphas:
        n = len(returns)
        var_pred = np.full(n, np.nan)
        violations = np.full(n, False)

        if dist == "normal":
            z_alpha = stats.norm.ppf(alpha)
        elif dist == "t":
            z_alpha = stats.t.ppf(alpha, 5)
        else:
            z_alpha = stats.norm.ppf(alpha)

        for t in range(window, n):
            hist_data = returns.iloc[t - window: t] * 100
            try:
                res = fit_garch(returns.iloc[t - window: t], dist=dist)
                cond_vol = res.conditional_volatility.iloc[-1] / 100
                var_pred[t] = z_alpha * cond_vol  # 负值，与历史模拟法一致
            except Exception:
                var_pred[t] = np.nan

        violated = returns.values < var_pred
        n_viol = np.nansum(violated)
        n_valid = np.sum(~np.isnan(var_pred))

        label = f"{int((1 - alpha) * 100)}%"
        lr, pv, p_hat = kupiec_test(n_viol, n_valid, alpha)
        results[label] = {
            "VaR 类型": f"GARCH 法（{dist}）",
            "显著性水平 α": alpha,
            "回测样本量": n_valid,
            "违反次数": int(n_viol),
            "实际违反率": p_hat,
            "期望违反率": alpha,
            "LR 统计量": lr,
            "p 值": pv,
            "检验结论": "不拒绝" if pv > 0.05 else "拒绝",
            "VaR 序列": pd.Series(var_pred, index=returns.index),
            "违反序列": pd.Series(violated, index=returns.index),
        }
    return results


def backtest_summary(returns, window=500):
    """
    对单个标的运行全部回测，返回汇总表
    """
    rows = []
    var_hist = backtest_historical(returns, window, alphas=[0.05, 0.01])

    for label, res in var_hist.items():
        rows.append({
            "方法": "历史模拟法",
            "VaR": label,
            "样本量": res["回测样本量"],
            "违反次数": res["违反次数"],
            "实际违反率": f"{res['实际违反率']:.4f}",
            "期望违反率": f"{res['期望违反率']:.2f}",
            "LR": f"{res['LR 统计量']:.4f}",
            "p 值": f"{res['p 值']:.4f}",
            "结论": res["检验结论"],
        })

    try:
        var_garch = backtest_garch(returns, window, alphas=[0.05, 0.01], dist="normal")
        for label, res in var_garch.items():
            rows.append({
                "方法": "GARCH 法",
                "VaR": label,
                "样本量": res["回测样本量"],
                "违反次数": res["违反次数"],
                "实际违反率": f"{res['实际违反率']:.4f}",
                "期望违反率": f"{res['期望违反率']:.2f}",
                "LR": f"{res['LR 统计量']:.4f}",
                "p 值": f"{res['p 值']:.4f}",
                "结论": res["检验结论"],
            })
    except Exception as e:
        rows.append({
            "方法": f"GARCH 法（失败: {e}）",
            "VaR": "-",
            "样本量": "-",
            "违反次数": "-",
            "实际违反率": "-",
            "期望违反率": "-",
            "LR": "-",
            "p 值": "-",
            "结论": "-",
        })

    return pd.DataFrame(rows)


def main_backtest(tickers_data):
    """
    对所有标的运行回测，输出汇总
    tickers_data: dict of {name: returns_series}
    """
    all_results = []
    for name, returns in tickers_data.items():
        print(f"\n{'=' * 50}")
        print(f"  {name} 回测验证")
        print('=' * 50)
        df = backtest_summary(returns, window=500)
        print(df.to_string(index=False))
        df.insert(0, "标的", name)
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    test_returns = pd.Series(np.random.normal(0, 0.02, 1500))
    print("回测验证测试（模拟数据）：")
    df = backtest_summary(test_returns, window=500)
    print(df.to_string(index=False))
