# 片長與素材不足

一般指定秒數使用 `approx`：60 秒解析成 58–62 秒。在區間內自然收尾，不必精確湊到中心。所有長度預設前後 2 秒，下限最低為 0；未填秒數仍由素材決定。

「不得超過 60 秒」使用 `at_most`：58–60 秒。必須精確 60 秒時才選 `exact`；自訂上下限使用 `range`。明確指定的舊 `preferred` 工作單仍保留原本可更短的行為，網頁標示為「依素材長度（可更短）」。文字 brief 不會暗中覆蓋工作單的秒數欄位，應在長度選單選擇相應模式。

網頁、JSON/YAML 工作單、CLI 和多比例版本都經同一個 Delivery 解析，保存時將 approx / at_most 寫成明確的 range 與上下限。重新載入後顯示實際區間。切換長度模式會清除原範圍欄位的影響。CLI 範例：

```sh
montagewright render /path/to/rushes --output /path/to/output --seconds 60
montagewright render /path/to/rushes --output /path/to/output --seconds 60 --duration-mode at_most
montagewright render /path/to/rushes --output /path/to/output --seconds 60 --duration-mode exact
montagewright render /path/to/rushes --output /path/to/output --seconds 60 --duration-mode range --minimum-seconds 58 --maximum-seconds 62
```

片長規格會傳到選片、節奏、成片審查與編碼後技術驗收。素材不足時保留完整自然的短版，標示未達下限；不靠拉長停留、重播或減速達標。節奏階段若只剩下限不足，保存當前短版，不再買一次節奏呼叫嘗試拉長它。最終技術驗收仍阻擋區間外版本成為正式交付。

同一來源的不同時間可以使用。重播檢查按來源時間與速度計算，不因 span ID 不同而漏掉；後出現的鏡頭必須宣告重播目的。明確為湊長的理由無法豁免，動作慢放與首尾呼應仍由成片審查確認是否成立。文字檢查只能攔下明確的填充理由，不能代替實際觀看。渲染尾格補齊最多兩格，用於解碼與影格取整，不能補出一秒停格。

區間邊界的編碼驗收允許 0.05 秒或一格的取整誤差；「是否在區間內」不等於其他內容、字幕、畫面與聲音品質均已合格。

## 本機驗證（2026-09-08）

- 全套測試 1,196 passed；相對執行目錄修正後，Web／duration 相關測試 39 passed。
- 實際編碼 57／59／61／63 秒影片，approx 與 at_most 共八項技術 QC 結果符合預期；57 秒影片保留 draft-preview 並因片長不符阻擋正式交付。證據：`artifacts/duration-policy-acceptance/result.json`。
- 瀏覽器載入工作單：主版本 58–62 秒與 square 58–60 秒均保留；實際提交免費預檢，兩個版本皆 preflight_ready，輸出路徑正確。證據：`artifacts/rewire-web-runs/b19db3726f4e/out/campaign-manifest.json`。
- 沒有新增 Gemini 呼叫或支出。本輪不是 74 支實際素材的付費重剪，也不代表已驗收新的 Gemini 創意決策。
