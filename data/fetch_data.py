#!/usr/bin/env python3
"""
数据获取脚本 —— 使用 AKShare 获取 A 股日线数据
标的：上证指数、招商银行、贵州茅台、创业板指
"""
import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
END_DATE = "20260513"
START_DATE = "20210101"  # 约5年数据

tickers = {
    "000001": {"name": "上证指数", "type": "index", "prefix": "sh"},
    "600036": {"name": "招商银行", "type": "stock"},
    "600519": {"name": "贵州茅台", "type": "stock"},
    "399006": {"name": "创业板指", "type": "index", "prefix": "sz"},
}

def fetch_index(symbol, prefix, start, end):
    """获取指数日线数据"""
    df = ak.stock_zh_index_daily(symbol=f"{prefix}{symbol}")
    df = df.rename(columns={
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    df = df.sort_values("date").reset_index(drop=True)
    return df

def fetch_stock(symbol, start, end):
    """获取个股日线数据（前复权）"""
    df = ak.stock_zh_a_daily(
        symbol=f"sh{symbol}",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    df = df.rename(columns={
        "outstanding_share": "outstanding_share",
        "turnover": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def calc_returns(df):
    """计算对数收益率和简单收益率"""
    df["log_return"] = df["close"].apply(lambda x: __import__("math").log(x)).diff()
    df["simple_return"] = df["close"].pct_change()
    return df

def main():
    print(f"拉取数据: {START_DATE} ~ {END_DATE}")
    print("-" * 40)

    for symbol, info in tickers.items():
        print(f"正在获取 {info['name']}（{symbol}）... ", end="", flush=True)
        try:
            if info["type"] == "index":
                df = fetch_index(symbol, info.get("prefix", "sh"), START_DATE, END_DATE)
            else:
                df = fetch_stock(symbol, START_DATE, END_DATE)
            
            df = calc_returns(df)
            
            # 保存在标的自己的子目录
            ticker_dir = os.path.join(OUT_DIR, symbol)
            os.makedirs(ticker_dir, exist_ok=True)
            
            csv_path = os.path.join(ticker_dir, f"{symbol}_daily.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            
            print(f"OK {len(df)} 行 -> {csv_path}")
        except Exception as e:
            print(f"FAIL 失败: {e}")

    # 生成数据汇总说明
    summary_path = os.path.join(OUT_DIR, "data_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("数据获取汇总\n")
        f.write(f"时间范围: {START_DATE} ~ {END_DATE}\n")
        f.write(f"数据源: AKShare (免费 A 股数据)\n")
        f.write(f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("-" * 40 + "\n")
        for symbol, info in tickers.items():
            f.write(f"{info['name']}（{symbol}）\n")
        f.write("-" * 40 + "\n")
        f.write("字段说明:\n")
        f.write("  date: 日期\n")
        f.write("  close: 收盘价\n")
        f.write("  log_return: 对数收益率 ln(P_t / P_{t-1})\n")
        f.write("  simple_return: 简单收益率 (P_t - P_{t-1}) / P_{t-1}\n")
    print(f"\n汇总文件: {summary_path}")
    print("数据获取完成。")

if __name__ == "__main__":
    main()
