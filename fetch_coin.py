"""
fetch_coin.py
═════════════
快速下載指定幣種的 15m 和 1h 資料

用法:
    python fetch_coin.py AVAX
    python fetch_coin.py DOGE BNB LINK    # 一次多個
"""

import sys
from download_data import download_and_save

TIMEFRAMES = ['15m', '1h']

if len(sys.argv) < 2:
    print("用法: python fetch_coin.py <幣種> [幣種2 ...]")
    print("範例: python fetch_coin.py AVAX DOGE")
    sys.exit(1)

coins = [c.upper() for c in sys.argv[1:]]

for coin in coins:
    for tf in TIMEFRAMES:
        symbol = f"{coin}/USDT"
        try:
            download_and_save(symbol, tf)
        except Exception as e:
            print(f"  [錯誤] {symbol} {tf}: {e}")

print("\n=== 完成 ===")