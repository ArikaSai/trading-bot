# Crypto Trading Bot

雙策略自動化加密貨幣交易系統，部署於 Binance Futures。

## 策略
- **SOL/USDT 15m**：EMA 趨勢跟蹤 + ADX 過濾 + 盤整縮緊 + 2.0R 保本 + ATR Trailing Stop
- **ADA/USDT 1h**：Donchian Channel 動量突破 + ATR Trailing Stop

## 功能
- 自適應 TWAP 拆單執行（資金超過單筆上限時自動拆分至多根 K 棒）
- 連虧熔斷機制（連續虧損 N 次後跳過下一個訊號）
- Discord Webhook 即時通知（進出場、定時報告）
- 雙幣種共享資金池，先到先得保證金分配

## 檔案說明

### 核心
| 檔案 | 說明 |
|------|------|
| `strategy.py` | 策略核心邏輯：技術指標計算（`prepare_data`）、SOL 訊號生成（`generate_signal`） |
| `livebot.py` | 實盤交易主程式：連接幣安 API，監控 SOL + ADA 雙策略，管理倉位與風控 |
| `config.json` | 所有參數的集中設定檔（API 金鑰、策略參數、風控參數、回測時間等） |
| `config_example.json` | config.json 範本（不含敏感資訊，供開源使用） |

### 回測
| 檔案 | 說明 |
|------|------|
| `backtest_sol.py` | SOL 單策略回測引擎：自適應 TWAP、每週提現模擬、前進分析、壓力測試、逐筆明細 |
| `backtest_donchian.py` | Donchian Channel 通用回測引擎：支援任意幣種，CLI 參數覆蓋，自動從 config.json 讀取預設值 |
| `backtest_sol_ada.py` | SOL + ADA 聯合回測：共享資金池雙策略模擬，含個別貢獻分析與理論 vs 實際收益驗證 |
| `rolling_mdd.py` | 滾動回測分析：按月滾動窗口評估策略穩健性，輸出 MDD 排序與統計圖表 |
| `donchian_sweep.py` | Donchian Channel 參數掃描：網格搜索最佳 entry_n / trail_atr / atr_sl_mult 組合 |

### 工具
| 檔案 | 說明 |
|------|------|
| `download_data.py` | 多幣種資料下載：從幣安下載歷史 K 線，自動計算技術指標並存檔至 `data/` |

## 快速開始

```bash
# 1. 安裝依賴
pip install ccxt pandas numpy matplotlib

# 2. 設定
cp config_example.json config.json
# 編輯 config.json，填入幣安 API Key 與 Discord Webhook URL

# 3. 下載歷史資料
python download_data.py          # 下載所有幣種
python download_data.py SOL      # 只下載 SOL

# 4. 回測
python backtest_sol.py           # SOL 單策略回測
python backtest_donchian.py      # ADA Donchian 回測
python backtest_sol_ada.py       # 聯合回測

# 5. 實盤（建議使用 pm2 管理）
pm2 start livebot.py --name livebot --interpreter python3
```

## 回測時間設定

在 `config.json` 的 `backtest` 區塊統一設定回測起訖時間：

```json
"backtest": {
    "start_date": "2023-01-01",
    "end_date": "2026-04-09"
}
```

## 風險提示

本專案僅供學習與研究用途。回測績效基於歷史數據，受複利效應放大，不代表未來表現。實盤交易有虧損風險，請自行評估。
