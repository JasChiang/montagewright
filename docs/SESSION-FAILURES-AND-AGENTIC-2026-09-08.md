# 過往 session 失敗與最新 Agentic 差異

日期：2026-09-08。此次為歷史證據審查，未新增付費推論、未修改應用程式。核對兩份原始 session JSONL、Git 歷史分支程式、run.log、report.json、費用帳本、QA 影格與 fixture 標註。未重新完整播放所有歷史成片。

## 主要結論

失敗同時包含程式接線、剪輯與幾何不可行、素材身分證據衝突、模型觀察錯誤、供應商工具呼叫限制、以及本機交付 QC。不能把它們全交給 Codex 改程式，亦不能認為改用 Agentic 就會全部消失。

8 月曾以 Codex 取代語意規劃，並非只讓 Codex 處理技術例外。歷史分支 codex/codex-subscription-backend 的 semantic_backend.py 將影片轉成 evidence pack／圖片，經 --image 交給 codex exec。這与目前討論的 Gemini 看影片、Codex 處理技術執行，是不同實驗。該分支仍存在，当前工作分支為 codex/editorial-plan-merge。

## 逐次證據

| 測試 | 實際結果 | 失敗位置 |
|---|---|---|
| 2026-08-25 Codex v6 | 59.521 秒；逐鏡模型審查 8/20 達成意圖 | 寬構圖遭直式裁切；替換後動作時間不夠、重複片段、整片缺秒 |
| 2026-08-25 Codex v7 | 59.288 秒；逐鏡模型審查 3/20 達成意圖 | 同框主體無法成立；修訂計畫只達 58 秒，與 exact 60 秒衝突；master true peak 約 0 dBFS |
| 2026-09-05 Agentic 小測試 | 6 秒影片，5 次 processing_call/result；US$0.018038 | 成功辨識 [2.8, 3.1) 秒短暫 PRESSED 狀態，與 fixture 標註相符；不是和 Static 配對的比較 |
| 2026-09-05 Fold8 Agentic/static 混合完整測試 | 時間線 29 秒、9:16、8 鏡；逐鏡模型審查 3/8 達成意圖 | 裁切、重複角度、動作／身分衝突；重剪 API 400 too many tool calls；true peak +0.1 dBFS |
| 2026-09-08 最新小型接線測試 | 30 秒合成來源，3 秒成片；Agentic 計畫成功 | review_cut 區域 import 遮蔽；Agentic 審查聲稱靜音影片有英文人聲；重剪沒有解決所有問題 |

模型逐鏡通過比例並非人類評分。這些測試片長、要求、模型輸入、程式版本、素材組合不同，不能用 3/20 和 3/8 判定模型優劣。

## 真正重複的失敗原因

1. **修訂計畫與執行狀態脫節。** 原始 9 月 session 記錄最新修復稿被舊 Editorial Plan／provider attempt 蓋回去，造成不必要的再次規劃。亦有修復請求漏 brief、時長、動作要求與參考圖的情況。屬程式問題，可交 Codex 修復，但必須驗證恢復到正確版本。
2. **計畫在來源或畫幅上不可行。** 寬構圖、多台手機同框與文字整體可讀，在 9:16 下未必能同時成立。8 月替換鏡頭又造成缺秒、重疊與動作不完整。應在初剪前把失敗條件回給 Gemini 改選；不能靠放寬驗證硬塞。
3. **來源身分與選中時間窗混淆。** 整支有 Fold8，不代表選到的那幾秒能確認；追蹤成功也不表示選對實體。另有同 SKU 不同實例被誤排除，以及審片缺同一套參考圖的問題。需要統一 scope 和具體影格证據。
4. **付費修復的輸入太大。** 9 月完整測試 replan_shots 仍附全部可用素材，API 回傳工具呼叫過多。這是已確認的直接終止原因；大範圍輸入是應縮減的工程因素，沒有伺服器 trace 可證明它是唯一原因。
5. **審片可能製造錯誤需求。** 9 月 8 日兩次 Agentic 評論聲稱靜音片含英文，觸發音訊重規劃；原流程音訊分支又忽略其他重大問題。後來加入本機靜音證據、固定取樣和分支停止條件。改動同時發生，不能視為單一模式 A/B。
6. **測試成功與成片成功分離。** 8 月 1170 項、9 月初 1155 項測試通過時，成片仍 release_blocked。真實執行才能抓到未走過的接線分支；輸出仍須量測時長、色彩與 true peak，模型 approve 不足以放行。

## Agentic 到底帶來什麼差異

- 可確認已在供應商端使用官方 processing 工具，而不只是傳了 agentic 字串。9 月 5 日完整測試的兩次 editorial_plan 各有 10 次 processing_call/result；逐鏡審查有 25 次；全片審查有 8 次。
- 6 秒 fixture 正確辨識短暫 UI 事件，證明這個案例中可取得較細的動作理解。輸出自述取樣到 10 FPS，但保存的 steps 主要是類型／簽章，因此不能從中獨立驗證內部每一次取樣策略。
- Agentic 的影片導航是模型內部能力，不等於能操作本機剪輯工具；它仍受本機編譯、幾何與工作狀態接線影響。
- 最新小測試的 Agentic 計畫有 16 次 processing_call/result，帳本 US$0.270177。3 秒成片的兩次 Agentic 全片審查分別 US$0.107539、US$0.016876；後來 Static 加靜音證據的審查 US$0.009257、0 次 processing_call。這些數值僅為各次已記錄使用，並非公平的價格／品質基準。
- Static 最後 approve 仍記錄頂部文字被裁切；既存最終報告仍因色彩標籤等原因 release_blocked。批准沒有證明畫面已完全修好。

## 費用應如何解讀

Fold8 完整實驗帳本合計 US$5.295625，187 筆紀錄、跨 6 個 run_id，是多次嘗試累計，不能說一趟用了 187 次呼叫。selection 占 US$1.812034、reference_candidate_discovery 占 US$1.267589、clip_cards 占 US$0.925450、editorial_plan 占 US$0.510368、逐鏡加全片 review 占 US$0.313115。

結束時尚低於設定 US$6，直接原因是 API 400，而非 BudgetSpent。最後失敗沒有可計 usage，所以本機帳本總額不是供應商完整帳單保證。8 月 Codex 報告 spend=0 是 Gemini 美元帳本沒有計 Codex 訂閱使用量，不代表沒有成本。

## 對下一版的具體影響

- Codex 自動修復聚焦：錯誤參數／欄位、版本恢复、工具失敗、技術輸出與流程測試。
- Gemini 保留：選鏡、故事、節奏、主体要求取捨、字幕語意與視聽判斷。
- 修復要先分流原因；API 限制先縮小資料，本機可確定的事先量測，幾何不可行回傳替代選擇。
- 必須新增局部修復證據包：問題接點、相鄰鏡頭、候選來源、失敗條件、前後輸出及已做修改。避免每次重送全庫。
- 要公平驗證模式差異，固定素材 hash、brief、參考圖、模型、輸出上限與工具版本，以同一套標註比對 Agentic 與 Static，再記錄錯誤率、成本和延遲。此次未啟動新的付費比較。

## 可追溯來源

- 原始 session 01a0315c-ea45-74f1-bd0a-cb4d89d1c7e9：行 6820、7273、7723、8148、8431、8453。
- 原始 session 01a060ec-c670-7ac3-9845-64446aa8a452：行 2013、2162、3035、3190、3237、3253、3287。
- artifacts/web-runs/samsung-codex-review-loop-20260825-v6/out/report.json
- artifacts/web-runs/samsung-codex-review-loop-20260825-v7/out/report.json、work/technical-qc.json
- artifacts/agentic-smoke/20260905T015453Z-C_fast_transient_ui.json
- fixtures/annotations/C_fast_transient_ui.json
- artifacts/galaxy-z-fold8-agentic38-full-20260905-v2/report.json、run.log、spend-events.jsonl、repair-audit.md、work/technical-qc.json、work/qa-contact-sheet.jpg
- artifacts/rewire-acceptance/out/run.log、spend-events.jsonl、work/responses/*.json、report.json
