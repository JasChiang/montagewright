# Grounding 獨立驗證

本工具將來源片段、實際裁切與構圖候選分開展示，避免把模型有回覆或辨識到型號當成成片正確。建立驗證頁不呼叫 Gemini。

```sh
.venv/bin/python -m montagewright.grounding_validation \
  --run artifacts/galaxy-z-fold8-rerun-20260908 \
  --output artifacts/grounding-validation-20260909
.venv/bin/python -m http.server 8880 --bind 127.0.0.1 \
  --directory artifacts/grounding-validation-20260909
```

目前案例來自既有草稿的 8 個鏡頭，每個鏡頭有原片、實際成片，以及 16:9、9:16、1:1 各兩種 fit/fill，共 64 支無聲對照片。候選是完整留邊或置中滿版，沒有假裝使用不存在的正確 bbox。來源時間以 current-timeline 為準，檔名包含內容與生成參數雜湊。長鏡頭最多取前 5 秒。

## 已接入主流程

- 主體 grounding 失敗後，不再清除 entity_id 並以中心裁切繼續。
- 保留失敗證據，交給既有 Gemini 編輯工具迴圈提出替代鏡頭；沿用同一 ledger 與預算上限。修正流程有界，並非保證只有一次 API 呼叫。
- 替代案不得移除原本要求的主體，保留音樂位置與未指定修改的對拍設定。已保存的替代案可直接重用。
- 再次失敗會以 grounding_blocked / exit 78 結束，Web 狀態可辨識，不交給技術修復器猜測身份問題。舊快取中明確記錄的身份限制會恢復。
- inspect_source 使用 Agentic video processing；短成片與構圖預覽用固定 4 fps。每輪重新附上累積影片證據，並保存 provider processing steps 供查核。
- 編輯工具可取得既有素材候選索引與失敗原因。候選索引是模型觀察，並非人工真值。

## 驗證邊界

validate_composition 可獨立檢查來源雜湊、來源時間、不同實體的 instance_id、必要主體可見面積及禁止主體。相同型號的兩支手機是兩個實體，不能以一個型號命中代替兩支都入鏡。

這個多實體檢查器目前是獨立驗證層，尚未成為主渲染器的完整逐幀追蹤契約。通過僅代表指定檢查時間符合提供的標註，不代表未取樣畫面也正確。畫面動作、失焦、語意承諾與全片節奏仍需另行檢查。

cases.json 的 human_annotation 預設為 null，human_ground_truth_complete 為 false。頁面的判語來自既有模型審查，不能用來證明模型自身準確率。下一步應建立有人工校對的少量困難案例，再做針對性 Gemini 驗證；不需要先重付 74 支完整 grounding 的費用。

本次本機與 mock 測試涵蓋接線、快取與阻擋行為；沒有新增付費 Gemini 驗證，也未重新剪出通過語意驗收的新成片。
