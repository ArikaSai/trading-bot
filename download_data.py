"""
download_data.py
════════════════
從幣安下載多幣種歷史 K 線，計算技術指標後存檔。
從 config.json 自動讀取需要下載的幣種與時間框架。

用法:
    python download_data.py              # 下載 config 中所有幣種
    python download_data.py SOL          # 只下載 SOL
    python download_data.py SOL ADA      # 下載 SOL 和 ADA
"""

import sys
import json
import ccxt
import pandas as pd
import time
from pathlib import Path
from strategy import CoreStrategy

START_DATE = '2020-01-01'
END_DATE   = '2026-06-30'


def load_symbols_from_config(config_path="config.json") -> list[dict]:
    """從 config.json 讀取所有需要下載的幣種資訊"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    symbols = []

    # SOL（主策略）
    sol_symbol = config['trading']['symbol']
    sol_tf     = config['trading']['timeframe']
    symbols.append({'symbol': sol_symbol, 'timeframe': sol_tf})

    # ADA Donchian
    if 'ada_donchian' in config:
        ada_symbol = config['ada_donchian']['symbol']
        ada_tf     = config['ada_donchian']['timeframe']
        symbols.append({'symbol': ada_symbol, 'timeframe': ada_tf})

    return symbols


def fetch_binance_data(symbol: str, timeframe: str,
                       start_date: str = START_DATE,
                       end_date: str = END_DATE) -> pd.DataFrame:
    """從幣安下載歷史 K 線（原始 OHLCV，不含指標）。"""
    print(f"\n[DL] {symbol} {timeframe} ({start_date} ~ {end_date})...")
    exchange       = ccxt.binance({'enableRateLimit': True})
    since          = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp  = int(pd.Timestamp(end_date).timestamp() * 1000)
    all_ohlcv      = []

    while since < end_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since        = ohlcv[-1][0] + 1
            current_date = pd.to_datetime(ohlcv[-1][0], unit='ms').strftime('%Y-%m-%d')
            print(f"  -> {current_date}", end='\r')
            time.sleep(0.1)
        except Exception as e:
            print(f"\n  [ERR] {e} - retry in 5s...")
            time.sleep(5)

    print(f"  [OK] {len(all_ohlcv)} raw bars")

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.ffill(inplace=True)
    return df


def download_and_save(symbol: str, timeframe: str,
                      start_date: str = START_DATE,
                      end_date: str = END_DATE):
    """下載 → 計算指標 → 存檔"""
    # 1. 下載原始 OHLCV
    df_raw = fetch_binance_data(symbol, timeframe, start_date, end_date)

    # 2. 計算指標（統一使用 CoreStrategy.prepare_data）
    print(f"  [CALC] indicators...")
    df = CoreStrategy.prepare_data(df_raw)
    print(f"  [OK] {len(df)} bars with indicators")

    # 3. 儲存
    filename = Path('data') / f"{symbol.replace('/', '')}_{timeframe}.csv"
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, float_format='%.4f')
    print(f"  [SAVE] {filename}")
    return df


if __name__ == "__main__":
    all_symbols = load_symbols_from_config()

    # 若有指定幣種，只下載指定的
    if len(sys.argv) > 1:
        requested = [s.upper() for s in sys.argv[1:]]
        all_symbols = [s for s in all_symbols
                       if s['symbol'].split('/')[0] in requested]
        if not all_symbols:
            print(f"[ERR] No matching symbols for: {sys.argv[1:]}")
            print(f"  Available: SOL, ADA")
            sys.exit(1)

    print(f"=== Download {len(all_symbols)} symbol(s) ===")
    for s in all_symbols:
        print(f"  - {s['symbol']} {s['timeframe']}")

    for s in all_symbols:
        download_and_save(s['symbol'], s['timeframe'])

    print(f"\n=== Done ===")
