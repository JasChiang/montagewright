# Editorial Grammar V2

這份文件記錄 2026-07-27 的外部 GPT‑5.6 Pro 方法稽核，以及 JasCueVideoLab 實際採納的泛用修正。目標不是再加一份「一次決定所有事情」的大 prompt，而是在語意意圖與 deterministic executor 之間補上可驗證的剪輯語法。

## 核心分工

```text
素材與 exact evidence
        ↓
敘事／關係／構圖義務
        ↓
候選選擇與 Trim Contract
        ↓
MusicEditPlan
        ↓
Shot Composition
        ↓
Virtual Camera Execution
        ↓
Final mux、QA、人工審核
```

- Gemini 判斷「要表達什麼、哪段素材符合、哪個段落應先後出現、音樂與畫面情緒如何搭配」。
- FFmpeg、MusicMap、PTS 與本機 validator 決定「exact sample、exact frame、真正可用的區間與輸出長度」。
- Grounding 與 SAM 提供主體 geometry；它們不替代敘事、身份或關係判斷。
- Renderer 只執行已核准 contract，不自行補故事、補秒數或改變必要 evidence。

現有 `EvidenceQueryLockV2`、`relation_temporal_mode`、relation carrier、phase predicate、QualitySafeInterval 與 VerticalVirtualCameraPlan 已涵蓋稽核建議的大部分 EditorialIntent／TrimContract。這次沒有再建立一套平行 schema，而是修補兩個實際缺口：music edit grammar，以及 hard cut 被 motion gate 誤判。

## 音樂剪輯語法

### 連續音樂仍是安全 fallback

如果一段連續來源已能自然涵蓋成片，優先保留它。這通常有最好的 musical flow，也最不容易製造不自然的和聲、節奏與能量跳變。

### 多段音樂不是任意拼接

`MusicEditPlanV2` 最多包含四個 passage、三個 join。每個 passage 必須來自 reviewed MusicMap 的 section 或 locked cue：

```text
Gemini／剪輯師
選 section-001 → section-003
描述 intro → climax、low → high
        ↓
本機 MusicMapLock
解析為 [0, 192000) → [384000, 576000)
        ↓
MusicEditPlanV2
以 10 ms micro-crossfade 接合
        ↓
FFmpeg
輸出 exact 383520 samples
```

允許的 join：

- `cut`：只適用於已核准的 section、phrase、downbeat、accent 或 transient 邊界。
- `micro_crossfade`：5–200 ms，用於消除 click 或柔化合法接點，不用來掩蓋錯誤段落選擇。

允許的 ending：

- `natural_track_end`：保留原曲結尾。
- `phrase_fade_out`：在核准段落邊界做短淡出。
- `reviewed_ending_hit`：最後 passage 必須真的結束在 locked `ending_hit` cue。

明確禁止：

- Gemini 直接輸出 sample 或把 MM:SS 當 audio truth。
- 自動 loop 或重播重疊的來源 passage。
- time-stretch。
- 在沒有核准 cue／section 的位置硬切。
- 為了湊成片秒數捏造 ending。

V2 目前是 `requires_human_review=true` 的研究路徑。正式 `feature-delivery` 仍使用單一連續 interval；先經真實音樂 A/B 驗證後，才考慮將 V2 納入交付 orchestrator。

## 虛擬鏡頭語法

虛擬鏡頭有兩種不同事件：

- 連續鏡頭運動：要檢查速度、加速度、jerk、containment 與解析度。
- 剪輯 cut：是 editorial discontinuity，不是無限快的 camera move。

先前程式雖會把太短的 smooth transition 轉成 cut，motion gate 卻仍計算 cut 兩側的位置差，因此把合法 hard cut 誤判成運鏡過快。現在 derivative measurement 會依 cut 拆成多個 continuous run，各自計算 velocity／acceleration／jerk。

泛用 fallback：

1. 同一連續路徑可安全完成：使用平滑 follow／recenter／push／pull。
2. 時間不足但兩個構圖皆成立：使用 hard cut。
3. 需要共同脈絡：先顯示 establishing／joint context，再切 detail。
4. 關係可依序重建：依 phase predicate 呈現 A → B。
5. contract 明確允許犧牲外圍：controlled clipping。
6. 嘗試下一個可行候選。
7. 只有必要 evidence 無法滿版時才用 fit/layout。
8. 必要關係仍無法證明：標記不可行並等待人工，而不是輸出假成功。

## 三個實務例子

### 訪談

Brief 需要「主持人提問、來賓回答、共同反應」。

- 共同反應是 simultaneous relation：要保留 two-shot 或短暫 joint establishing。
- 問答可 sequential：主持人 medium → hard cut → 來賓 medium。
- 若主體距離太遠且一句話很短，不做高速 pan；直接使用 speaker cut。
- 有對白時，音樂 passage 可保持連續並使用 typed ducking。不能因 downbeat 就切斷一句話。

### UI 操作

Brief 需要「看到操作、確認狀態改變」。

- establishing 先交代完整裝置與操作關係。
- detail phase 追蹤手指或 UI region。
- result phase 在 exact state-change frame 後保留閱讀時間。
- 音樂 cue 可以對齊 reveal 或 result，但不能刪掉 setup／action／result 來硬卡拍。
- 若同一 crop 無法同時顯示操作者與 UI，使用 context → detail 的 phase cut，而不是黑邊補滿全程。

### 活動／產品比較

Brief 需要「A 與 B 的外觀或尺度差異」。

- 若觀眾可依序比較，使用固定尺度的 A → B cuts；不得在兩段間任意 push-in 改變比較基準。
- 若硬幣、尺、人物或場景是尺度 reference，它是 relation carrier，不能只保留產品名稱。
- 音樂可由較低能量 section 接高潮 section，但 passage 與 join 都需核准；接點不順先換 section，不是把 crossfade 拉長。

## 如何加速

加速的原則是「昂貴工作只給已通過便宜 gate 的少數候選」，而不是降低證據標準。

### 分層 shortlist

```text
全部素材
  → 本機 probe／shot／quality coarse scan
  → Clip Card 與便宜語意召回 Top-20
  → brief／music 語意排序 Top-5
  → exact-frame／geometry preflight Top-2
  → SAM／identity／render 直到第一個 eligible candidate
```

數量是可配置 budget，不是內容規則。若必要 evidence 只有一個候選，必須保存原因；若候選全失敗，停止而不是增加 paid calls。

### Cache 與共用計算

- Clip Card cache key 包含來源 SHA、segment range、sampling policy、prompt/schema/model version。
- File API 物件按來源 SHA 與有效狀態重用。
- 同一 source interval 的 decoded frames、shot manifest、exact frame 與 frame hash 共用。
- 同一 shot 的多個 target 共用 SAM inference session 與 analysis frames；bbox seed 仍分 target。
- 同一 track 可供 16:9 virtual camera、9:16 reframe、圖卡避讓與 QA 使用，不重算 geometry。
- 已完成的 paid response、raw output、usage 與 binding 全數保存；429／503 不當作 schema repair 重付一次。

### Gemini 分階段

1. 素材級：Clip Card、事件與用途。
2. 專案級：brief＋music 下的章節、Top-K、相對停留。
3. 候選級：只對 shortlisted clip 做 framing proposal。
4. exact-frame 級：只對真正需要 geometry 的 frame 做 bbox／identity verification。
5. 成片級：一次 final sequence QA，而不是每段都再看完整影片。

這份方法不承諾尚未量測的加速百分比。每次真實測試應記錄 video minutes、paid calls、cache hits、tracking frames、磁碟、wall time 與人工修改時間，再決定 Top-K 與分析密度。

## 下一個真實 A/B

在把 V2 接進正式交付前，建議只做一個受控測試：

1. 使用同一 picture lock。
2. A 版使用單一連續 music interval。
3. B 版使用 reviewed 2–3 passage MusicEditPlanV2。
4. 比較 musical flow、接點突兀、敘事能量、ending、人工修改量與成本。
5. 兩版皆以 final mux 執行技術 QC 與一次 Gemini FinalEditQA，再由真人決定是否讓 V2 進 production orchestrator。
