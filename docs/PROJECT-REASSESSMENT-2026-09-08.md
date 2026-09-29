# Gemini 剪輯專案重新評估

評估日期：2026-09-08。目標：使用者提供毛片、可選需求與比例，由 Gemini 理解素材、提出剪輯方案或完成剪輯；需要字幕時，以 Apple ASR 為時間基礎，由 Gemini 校正完整內容並按交付畫面語意斷句。

## 結論

保留本機媒體與幾何核心，重整主流程、模型決策邊界與字幕中介資料；新增轉場時間線能力。沒有證據支持整個專案推倒重寫，也沒有證據支持現在已經可靠達成目標。

現況是功能相當多的固定剪輯流水線。Gemini 的確接收影片並產生剪輯決策，但模型不是自由操作本機剪輯工具的 agent；目前由 Python 編排階段、驗證結構化計畫、執行與修復。這種架構可以達成需求，前提是具備完整素材覆蓋、可回看素材、可修訂決策與可靠的成片檢查，不必為了 agent 名稱開放任意 shell。

## 本次證據與限制

- 檢查目前 `codex/editorial-plan-merge` 的工作目錄，包括既有未提交修改；未改動應用程式碼。
- 閱讀 CLI、Gemini 輸入、planner、Clip Card、字幕回填、裁切、renderer、時鐘、工作單與 release 路徑。
- 執行 `.venv/bin/python -m pytest -q`：**1155 passed，1 warning，55.78 秒**。警告是 Starlette/httpx 棄用提示。
- 本機可找到 FFmpeg、ffprobe、Swift compiler、Apple 轉錄執行檔與 SAM checkpoint；Python 可發現 cv2、torch、sam2、PIL。這是可用性盤點，不是重新驗證所有硬體／模型推論。
- 執行字幕漏句最小重現，證明純插入文字可得到零長度時間區間。
- 查閱 2026-09-05 Fold8 run 的 report、technical QC、release manifest；它仍為 draft/release_blocked。未在本次重新完整觀看或重渲染該影片，因此審片內容是既存模型評語，不能當成本次人工視覺判定。
- 未發出新的 Gemini 付費請求。本次不證明模型在新素材上的理解品質、最新付費端到端成功率或輸出運鏡的視覺驗收。

## 主要落差

### 1. 送入完整影片，不等於完整內容已被檢查

主規劃預設走合併 Editorial Plan，使用附來源編號、來源時間碼及 source map 的 stringout。30 分鐘以上先以既有 Clip Cards 做文字 logging/selects，再把選中的區段送給主規劃。30 分鐘是專案政策，不是 Google 官方上限。

目前主規劃、聽寫與 review 使用 agentic video；素材卡按本機 motion 判斷 agentic 或 static 4/8 FPS。Google 官方明說 agentic 會依問題選擇時間區段、模態與取樣方式。因此不能把上傳完成、模型口頭宣稱看完，或有 processing_call，當成每個區段均被檢查的證據。固定 FPS 也不代表逐幀理解或零漏看。

建議：素材分析應有獨立 coverage ledger，以 source ID、來源時間區間、畫面／聲音檢查狀態、分析版本、失敗與待補查區段記錄。長片先完整分段建立紀錄，再集中深看候選；遇到 Brief 新需求或摘要沒有記載的細節，要能重新查原素材。專案現有 coverage.py 檢查的是成片時間是否有內容支持，不能代替素材瀏覽覆蓋。

依據：`cli.py:2793`、`planner.py:1861`、`planner.py:2264`、`planner.py:3972`、`gemini.py:50`、`coverage.py:1`。

### 2. 缺少清楚的「只提方案」與「直接剪輯」產品入口

目前主要入口是 render，另有免費 preflight 與內部 planning artifacts。preflight 只檢查素材與工具，不是 Gemini 看完後提出方案；內部 JSON 也不等於適合使用者閱讀的提案。

比例未指定時目前預設 9:16，字幕預設 sidecar；不是模型看完素材後選擇。建議保留「未指定」狀態，由 Gemini 提議或依使用者偏好執行，明列假設。需求欄位應區分硬限制與偏好，例如「每顆要有 A」不等於「任何其他人都不得出現」。

應增加兩種結果：提案模式輸出方向、建議比例／長度、段落、引用素材區間及限制；剪輯模式使用同一提案與素材證據直接產生 draft，不能重做一套互不相干的決策。

依據：`cli.py:6310`、`cli.py:6377`、`job.py:22`。

### 3. 字幕回填只完成了部分需求

現況流程是 Apple ASR → Gemini 獨立看影片盲聽 → 第二次 Gemini 比對兩份文字 → 本機 SequenceMatcher 回填。盲聽有要求完整逐字，不是只處理低信心片段；但第二次校正沒有影片／音訊，遇到兩份文字衝突時無法重新聽證據。

更直接的衝突是 transcript prompt 禁止引入 ASR 完全沒有的整句話。回填器對純插入給零長度區間，describe 又丟掉 end <= start 的行。

最小重現：Apple 有「你好」0–1 秒、「再見」3–4 秒；校正稿中間增加「今天介紹手機」，across_lines 回傳新增句為 3–3 秒、measured_chars=0。這不是尚未調好 prompt，而是時間資料根本不存在。

修正方向：保留不可變 Apple 原稿、Gemini 修訂稿及 insert/delete/replace 對應；局部漏字可在既有 Apple span 內標記為推估分配。整句漏辨須定位原音訊區段，重新 Apple ASR 並回映來源時鐘；若依然無錨點，保留未解決項。只有在產品允許另一種本機對齊來源時才加入 forced alignment，且不能把其時間標成 Apple 實測。Apple word span 內拆字也屬推估，並非每字都是獨立量測。

語音是否值得轉錄目前先受 Clip Card 的 speech=content 決定，失敗可能略過。明確要求字幕時應能覆寫這種分類並檢查所選語音覆蓋，不能讓一張誤分類卡永久消失一段發言。即使不燒字幕，訪談剪輯仍需要逐字稿支援選句與切點。

依據：`transcript.py:1180`、`transcript.py:1299`、`transcript.py:1363`、`backfill.py:69`、`prompts/transcript_zh-TW.txt`、`cli.py:2353`。

### 4. 現在的比例斷句不是 Gemini 按成片語意排字幕

Gemini 在來源稿階段按語意切行，但 describe 沒有交付比例／版面輸入。成片端 as_cues → split_cues → _by_sense 主要按字寬、標點與平衡長度切短；沒有每個成片時段的主體／字卡情境判斷。CLI 的 as_cues 出錯還會直接略過。

應把三件事拆開：原始發言、校正文字、交付字幕 cues。剪輯時間線確定後，Gemini 接收校正文字、Apple 對應 ID、比例、字體可用寬度、最大行數、鏡頭／說話者切換、主體與字卡避讓資訊，選擇語意邊界與候選排法。本機驗證像素寬、行數、閱讀速度、最短停留與時碼單調性；不合格時回傳具體哪句超限再重新斷句。Gemini 不產生新的精準時鐘。

改比例應重算交付 cues，不重聽全部素材；改錯字才失效受影響的文字對齊／cues。

依據：`subtitles.py:438`、`subtitles.py:536`、`subtitles.py:885`、`cli.py:4431`。

### 5. 一些剪輯偏好被本機提升成硬規則

coverage.py 定義反應 1.5 秒、結尾 1.5 秒、建立 3 秒、music montage 4 秒等內容支持上限，部分路徑會縮短或拒絕超限片段。這能阻止無內容湊時長，但也可能限制刻意的停頓、風景或慢節奏片段；部分角色有 action/motion 證據延長機制，仍不能把全體創意都用短片常數定義。

建議硬限制只管真實來源時窗、同步、合法幾何、必要主體與使用者明確條件。節奏長短改為可說明的疑點，交給 Gemini 看實際候選／成片判斷；慢節奏可以成立，不能只因超過某個常數就改短。

依據：`coverage.py:29`、`coverage.py:95`、`prompts/editorial_plan_zh-TW.txt`。

### 6. 成片審查不是預設完成條件

render 的 Gemini review 目前選配。已查閱的 Fold8 run 有既存審片指出靜態停留、同角度跳接、切斷副主體；後續因 too many tool calls 中止，technical QC 記錄 master true peak 0.1 dBFS 超過 -1.0。這是該 run 的歷史產物，不足以判定目前程式仍有同一 bug，但足以說明測試全綠不等於交付已完成。

建議剪輯模式至少包含一輪完整成片視聽 review，對問題鏡頭／接點局部回看；保留修復版次與只重做受影響段落。API 失敗要保留最後可看版本，不能標為通過；有界重試或改採較短的確定區間，不重送成功的整批。審查預算應在規劃前預留。

依據：`cli.py:6417`、`review.py:163`、`artifacts/galaxy-z-fold8-agentic38-full-20260905-v2/report.json`、同目錄 `work/technical-qc.json`、`release-manifest.json`。

### 7. 預算確實可能讓專案停在尚無影片的階段

目前 Ledger 是單一累計上限，明確不做各階段資金預留。每次 Gemini 呼叫前，以 countTokens、schema 餘量及 max_output_tokens 保留額度；不足就拋 BudgetSpent。素材卡階段也可能直接停住，CLI 明寫 library 尚未完成，並向外拋出；這時沒有「最佳成片」可交。規劃、字幕或身份確認在初剪前停止，也有同樣問題。

若已經 render，後續 review／repair 的 BudgetSpent 有保存現有成果與停止原因的路徑；不能把這個後段行為概括成全流程都保證交付影片。原始註解「到上限就提供當下最佳剪輯」只在剪輯已存在時成立。

已完成素材卡／逐字稿與決策有快取，同一 output 續跑也扣除歷史支出，不會把 cap 重置。這降低浪費，但不是所有已付費工作都安全：transcript.describe 的盲聽與文字校正是兩次請求，盲聽結果在記憶體裡，只有整個 describe 成功才由呼叫端 save。若第一輪已付費成功、第二輪額度不足或失敗，續跑有重付盲聽的風險；Apple 原始 payload 也應獨立落盤。

另有兩種預算不準確：

- 很多呼叫預留 65,536 output tokens，可能在實際小回應本來可負擔時先拒絕。這是保守預留造成提早停止，不是實際費用已用完。
- settle 加入 agentic tool_use_tokens，但 reserve 沒有單獨的動態 tool-use 預留；5xx 重試也只有 uncertain attempt 紀錄，未知支出不扣成可知美元。因此目前機制不是可保證 Google 帳單絕不超過設定值的硬上限。

建議改成「完成成本管理」：

1. 免費盤點後，根據素材量、語音量、快取與需求估算範圍；說明不確定性，不能保證精確總價。
2. 區分目標預算與使用者授權的最高花費。接近目標額度不終止；授權最高額度仍是新付費呼叫的邊界，不能自行越過。
3. 以剩餘工作圖預留完成核心成果的費用：完整理解、必要文字校正、計畫、必要身份證據、字幕 cues、至少一輪審查。額外美化／多版本／額外修復使用剩餘空間，並不是降低核心理解品質。
4. 預估無法完成時，在大量付費前提出可比較的範圍／成本方案。不能先花掉大半，再告知要加錢。
5. 每個成功付費回應先原子保存，再解析／進入下一階段；以素材 hash、模型、prompt、輸入版本與階段 key 續接。上傳 URI 快取不等於模型推論結果快取，也不等於推論免費。
6. 到付費邊界時，允許已取得全部必要證據的本機編譯、渲染、匯出繼續；不要為純本機動作套 Gemini 額度檢查。若缺的是必要身份證據或完整素材理解，產出已分析資料／方案及缺口，不能冒充完成剪輯。
7. 狀態區分 budget_paused、provider_failed、draft_ready、review_pending、complete，保留接續點及完成尚需的工作；未知扣款須單獨對帳，不盲目重送。

核心判斷：你的擔心成立。問題不只是上限太低，而是「每次呼叫負擔得起」沒有推導出「整項工作負擔得起」。完全拿掉上限同樣不能解決 API 失敗與無限修復；應讓預算以可用成果為單位運作。

依據：`cost.py:1`、`cost.py:171`、`cost.py:210`、`cost.py:277`、`planner.py:553`、`planner.py:82`、`cli.py:2325`、`cli.py:2382`、`cli.py:4341`、`transcript.py:1180`。本次未新增付費呼叫驗證帳單，動態費用風險為程式路徑分析。

## 本機工具：保留、補強、新建

| 能力 | 現況判斷 | 建議 |
| --- | --- | --- |
| 素材盤點／proxy／hash／來源映射 | 已有可重用基礎 | 保留，新增瀏覽覆蓋與標準化轉檔映射 |
| 精準時間線／硬切 | source 與 screen clock、影格量化、handles 已有 | 保留核心，補跨鏡頭時間關係與特定動作局部精定位 |
| 畫面轉場 | renderer 以 segment concat 串接；未見 dissolve/xfade 的畫面執行路徑 | 新增轉場資料結構及 compiler，先 cut、cross dissolve、dip to black |
| 基本數位運鏡 | hold、pan、tilt、push/pull、follow、reveal、compare、multi-stop 已有 | 不整套重寫，統一可行性與 rendered proof |
| 逐幀縮放 | 變動尺寸用 perspective filter，避免 crop w/h 只初始化一次的問題 | 保留，檢查放大品質、邊界、速度與終點 |
| 原生運動分析 | affine/RANSAC 可估平移、縮放、旋轉 | 是分析器，不等於防手震 renderer |
| 防手震 | stabilize_then_reframe 的語義是取停穩後區段 | 若素材需要，新增實際穩定化 transform 與黑邊／裁切預算 |
| 主體跟隨 | Gemini 定身份＋SAM 傳播＋幾何約束 | 保留，補遮擋重現、近似主體、邊界及多主體的實片驗收 |
| 變速 | 單段固定 ratio、setpts＋atempo | 保留；可變速率 ramp、光流補幀未見完整路徑，有實際需求再擴充 |
| 聲音 | 獨立 audio assignments、ducking、音量整形、淡入淡出已有 | 優先驗證編碼後響度／true peak、切字／爆音／room tone；降噪依素材增加 |
| 字幕 | ASR、校正、回填、燒錄已有 | 重整文字修訂／時間 provenance 與成片語意 cues |
| 多畫面合成 | PiP/split screen/screen insert 在工作單明確被擋 | 獨立 compositor，有需求才建；不能由裁切冒充 |
| 手機／混合格式 | VFR、HDR 等目前拒絕；交付僅已驗證整數 FPS 與 H.264 | 優先建立 VFR→CFR 對應與 HDR→SDR 色彩流程，實片驗證後再開放 |
| NLE 匯出 | Premiere/FCPXML、裁切 keyframes 已有 | 新增轉場／ramp 後需驗證外部 NLE 與 MP4 一致性 |

轉場不能只加一個名稱或 FFmpeg 參數：它同時佔用前後兩顆、需要來源 handles、會影響總長、音訊、字幕與主體可見性。應有 outgoing/incoming clip、開始／結束影格、轉場型態、音畫策略及來源可用範圍。先把這層做好，再考慮 wipe、whip、match zoom 或 mask transition；不能用特效遮掩錯誤的選鏡關係。

運鏡重做的觸發條件應是可重現問題：規劃與實際終點不同、方向錯誤、原生運動被重複疊加、固定構圖微抖、變速後路徑時間錯位，或基本推拉需針對每個案例加例外才能成立。現有核心已有不少對應處理，應用渲染回歸矩陣決定替換哪一層，而非先假設要重造全部。

本機動作事件也要區分來源：量測到 PTS 只代表精準位置，不自動證明那格在語意上是「動作完成」。Gemini 先找區段，本機解碼候選並回傳局部片段／影格，確認選中哪一個事件後才定切點。

依據：`renderer.py:417`、`renderer.py:866`、`renderer.py:298`、`reframe.py:575`、`reframe.py:2286`、`capabilities.py:69`、`motion.py:186`、`job.py:290`、`ingest.py:125`。

## 建議主流程

1. 收素材與可選需求：提案／直接剪輯、硬條件、比例未指定狀態、字幕與聲音需求。
2. 本機盤點及必要標準化：素材與 proxy 皆保留來源時間映射。
3. Gemini 全素材分段理解：具名事件、語音、主體、可用區間與覆蓋缺口。
4. 需要語音理解的素材跑 Apple ASR；Gemini 檢查完整內容並可回聽爭議區間，本機保存修訂與時間對應。
5. Gemini 產生使用者可讀方案及同源 Editorial Plan；提案模式在此完成。
6. 本機評估幾何、工具可行性、時間／音訊限制，回傳具體衝突；由 Gemini 修訂創意選擇。
7. 編譯 picture、audio、transition、crop 與 graphics，產生 draft。
8. 依最終時間線與比例做字幕語意斷句、像素排版及時鐘對應。
9. 本機技術 QC ＋ Gemini 全片審查／局部接點檢查，有限修復並只重渲染受影響部分。
10. 提供影片、方案、可編輯 timeline 與未解決事項。檔案成功輸出、模型通過與使用者接受分開記錄。

需要動態工具迴圈時，只提供有型別的 inspect_source、inspect_interval、evaluate_crop、preview_transition、render_draft、inspect_cut、revise_plan 等介面；這是建議介面，並非現有已完成 API。固定 pipeline 仍可承擔大部分編排，不必一次改成通用 agent framework。

## 優先順序與驗收

**第一階段：先讓核心承諾成立。** 完整素材／語音覆蓋、漏句保留及補查、按比例語意 cues、明確提案模式與預設成片 review；同步修正完成費用預留、付費回應 checkpoint 與可恢復預算暫停。移除會冒充技術限制的剪輯偏好硬門檻；保留真正的來源與同步約束。

**第二階段：補常見素材與基本剪法。** VFR/HDR 標準化、基礎轉場 compiler、編碼後音訊 QC；用實片確認既有運鏡品質後只修故障層。若使用者素材全是目前已支援格式，格式工作可延後。

**第三階段：按真需求擴充。** 穩定化、speed ramps、多畫面、特殊轉場與音訊修復。停止把字卡皮膚或新意圖名稱當作核心剪輯完成度。

驗收至少包含：

- 無 Brief 的多支毛片：方案引用真實區間，已檢查／未檢查素材明列。
- 指定 A 的混合主體素材：分別驗證 target-led、每顆有 A、禁止 B，尤其轉場中間格與最後 crop。
- ASR 同音錯字、漏字、整句漏辨、重複發言與多人訪談：既無捏造，又不靜默丟句；無時間錨點明列。
- 同一發言 16:9 與 9:16：按語意重新分 cues，專有名詞不斷裂、不過短閃爍，時間仍對得上聲音。
- 靜態、原生 pan/push、主體移動、遮擋與變速：比對計畫路徑、渲染開中尾與連續播放；無跳動、漂移或誤跟。
- 硬切／溶接／淡黑：總長正確、handles 足夠、沒有黑格／雙影誤義、音訊與字幕無漂移。
- API 審片失敗：最後可看版本與花費保留，只對失敗範圍續做。
- 初剪前、盲聽後校正前、初剪後各自耗盡預算：已付費回應可重用，純本機工作可完成，缺少成果時不可宣稱交付，續跑不重置累計花費。
- 混合手機格式：來源時間、色彩與聲音在標準化後保持可追溯。

## 值得討論的流程邊界

- 「看完」建議定義為每支來源的完整時間區間均有畫面與聲音分析紀錄，重點動作可深看，而不是把每一影格都送最高解析度。即使有覆蓋證據，仍要用抽查／成片回看測量漏看率。
- Apple 細粒度時間是原始錨點，不是絕對無誤的真值。校正文字增加到原 ASR 不存在的區段時，必須重辨、局部對齊或保留待解，不能直接把新增句塞進零長度位置。
- 不需要字幕不等於不需要 ASR；只要剪輯依賴發言內容，仍應用 Apple 時鐘保護切句與音畫同步。純無敘事語音的畫面剪輯才可省略。
- 「只出現特定主體」應能區分主角主導、每顆包含主角、其他主體完全禁止。最後一種條件要檢查成品 crop 與轉場中間格；素材若不支援，要回報不可達，不可偷偷換成前兩種。
- Gemini 可用片段時間定位去找內容，但最終刀點要落到來源實際影格；本機精準時碼和 Gemini 對事件的判斷是兩項不同證據。
- 提案與直接剪輯都應有效；資訊足夠時不用強迫中途核准。預算目標可以是提醒值，額外消費授權的邊界要另外定義。

## 外部依據

- [Google：Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Google：Introducing Agentic Video](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-agentic-video-in-gemini/)
- [Apple：SpeechAnalyzer 與 audioTimeRange 示範](https://developer.apple.com/videos/play/wwdc2025/277/)

外部文件用於確認 video processing 與 ASR 時間屬性的分工；是否在本專案可靠成立，仍以目前程式與實際輸出驗收為準。


## 本輪實作與實際驗證

此節區分已接通的主路徑與尚未證明的剪輯品質；前文問題清單是修改前盤點。

已加入提案／直接剪輯模式、自動比例、完整來源取樣紀錄、API 原始回應持久化與累計預算續跑、後續交付步驟預留額度。目標預算只提醒；硬上限攔截新的付費步驟，回傳 budget_paused（exit 75）。API 回應即使後續解析失敗仍可重用，不把失敗當成重付授權。

Apple 原始辨識結果先保存，再由 Gemini 看完整來源進行獨立聽寫與交叉校正；沒有 Apple 時間錨點的漏句觸發局部重辨，仍無錨點則保留待解證據。Gemini 根據比例、實際字寬與時間做語意分句，字元時間仍由本機回填。成片字幕另存為 Web 共用的字幕權威，重新載入與下載不另行機械重切。

本機 MP4 新增有真實 handles 的溶接及淡黑，維持既有音訊／字幕時間。未新增原生穩定化、多畫面或 VFR/HDR 全格式支援。外部 NLE XML 的轉場仍須手動重建，匯出標記會明示；不可把 MP4 轉場驗證當作 NLE 原生效果驗證。

實際證據：

- `artifacts/rewire-acceptance/out/`：真實 Gemini 素材卡、自動比例、提案、恢復剪輯與審片。初次真實執行抓到 review_cut 的區域 import 遮蔽，已修正。恢复沿用已付費卡片／計畫／回應。零新增預算提案重用成功。
- `artifacts/rewire-acceptance/budget-paused/`：新素材、零預算在付費看素材前暫停，保存預算預估與暫停原因。
- `artifacts/rewire-acceptance/captions/speech.srt`：Apple 本機實際辨識台灣中文合成語音，Gemini 校正、9:16 語意斷句共三次真實呼叫。能修正多處同音錯字，但「這段字幕」仍誤成「就算字幕」；這是接線通過、文字品質未全對的明確反例，不等於真實多人訪談驗收。
- 硬切／溶接／淡黑的 FFmpeg 測試檢查中間格色值、60 格與 2 秒總長，並非只驗 JSON 欄位。
- Web 真實瀏覽器已驗證預設審片、auto 比例、素材輸入及免費預檢；提案下載與預算／模式續跑另由 HTTP 整合測試驗證。

真實審片也證明不能盲目信任評論：agentic 評論在數位靜音影片中捏造英文人聲，並讓音訊修復分支忽略其他畫面問題。已加入本機靜音證據，整片複檢改固定取樣，音訊修復同時携帶所有重大問題並遵守輪數／無進展限制。後續固定取樣審查批准了該片，但畫面裁切仍應以人工檢視及本機限制為準；模型批准不能覆蓋降級／技術 QC，也不能當成「與人類剪輯一樣好」的證明。

以上真實呼叫累計帳本估算：剪輯測試 $0.609421、字幕測試 $0.039166。這是本機 usage 計價紀錄，不是供應商最終帳單。

## GO 實作：可回看的剪輯工作區與局部修復

保留既有 Gemini 主規劃與 FFmpeg 執行器，新增 `editor_workspace.py`。每次提案／成功 render 都保存素材索引、完整 speech/audio spans、使用者 brief、方向、選片、實際時間軸、來源雜湊與版本預覽。修復時每一輪都帶完整索引與當前計畫；索引是導航，不代表模型已看過該段影片。

`replan_shots` 與音訊問題的重新選片已接到結構化工具協定：Gemini 可要求 `inspect_source`、`inspect_cut`、`preview_framing`。本機驗證來源 ID、原始時間範圍與尺寸，產生含 source-clock 對照的影片，下一輪回傳實際影片。每輪最多 3 個工具、最多 3 輪、每段 20 秒、證據總長最多 180 秒；重複請求不重做。新選的來源區間必須落在已回看證據內。這是沿用既有 checkpointed `ask()` 的 JSON 工具協定，並非 MCP server，也不把 shell 指令交給 Gemini。

已接通 fit 全景留白至 Selection → Clip → Segment → compiler → FFmpeg → Web recut；fill、左右平移可先試預覽。真正的數位運鏡仍由既有語意 looks 與本機幾何編譯器執行。fit 與數位推拉／平移不能混用。已有主體身分或排除約束時，既有「裁切區域」證據不足以核可完整原畫面，fit 只能形成待驗證草稿，不能沿用裁切證明自動通過。

逐鏡複檢改為最多 3 顆一組的穩定批次 checkpoint。完整影片以解碼後的畫面與聲音摘要比較；修改敘述但輸出沒變，不再付費複檢。同時保留上一版影片供介面比較。主流程實測另抓到「重剪後只更新 report，resume 仍讀初版 resolved-selection」問題，現已在每次成功 render 時更新續跑依據。

字幕成片複檢最多進行一次文字修復／重燒／再複檢。Gemini 回聽未燒字影片、修全文，程式將修字對回 Apple 字元時間；没有 Apple 錨點的插入不得硬填時間。非文字的遮擋或樣式問題不以改字假裝解決。重燒先寫候選檔，成功後才替換現有成片；語意 cues 與細粒度時間存入同一份 `work/subtitles.json`，供 Web／SRT 重用。

### Codex 技術例外路徑

`technical_repair.py` 已接到 CLI render 例外出口，工作單與 CLI 可透過 `technical_repair`／`--no-technical-repair` 控制。每個 output 最多一次，只處理 Python NameError／AttributeError／TypeError 等接線例外；不處理預算、API、素材語意或審核拒絕。只准更改隔離副本中 1–3 個既有 Python 原始檔，保護 API、預算、checkpoint、release 與修復器自身，使用原始測試套件驗證，再確認工作目錄沒有並行變動，備份原檔、套用並由新程序續跑。這不是所有故障都保證自修的機制。

實測發現 PATH 上 codex-cli 0.149.1 無法使用帳號設定的 gpt-6-astra。改為偵測本機可用版本，採用較新的 `/Applications/ChatGPT.app/Contents/Resources/codex` 0.153.4；未變更全域安裝或帳號模型。可用 `MONTAGEWRIGHT_CODEX_BIN` 明確指定執行檔。沙箱內無法寫 Codex 狀態資料庫的第一次測試失敗；獲准執行正常本機存取後，較新版的隔離修補、原始測試與新程序續跑成功。`artifacts/codex-repair-acceptance/output-v3/work/technical-repair/` 保留證據。Codex 用量獨立於 Gemini 帳本；此小測試回報 input 101602、cached input 75264、output 523 tokens，不代表美元成本，也不是「免費的本機運算」。非互動執行依 [官方 CLI 文件](https://learn.chatgpt.com/docs/codex/cli) 使用本機已登入工具。

### 本次驗收證據與限制

- 真實 C8332 三手機素材：Gemini 要求 fit 預覽、看片後選用，編譯並輸出 16:9／9:16／1:1。已檢查實際影格、尺寸、時長與色彩標籤。首次 3 個 Gemini 呼叫合計 US$0.029176；相同請求續跑新增 0 個呼叫、US$0。證據在 `artifacts/editor-tools-acceptance/first-run-result.json` 與 `result.json`。這是構圖工具整合實測，不是多素材長片品質評比。
- 真正 CLI 閉環：`artifacts/rewire-acceptance/out` 的 3 秒 UI 合成素材，初版逐鏡 0/1 → 工具回看與 fit → 本機重剪 → 逐鏡 1/1，全片 approve 且零 issues。成功修復那一輪新增 US$0.136223 左右，精確值以 spend-events 帳本為準。
- 閉環第一次工具請求因 high thinking 用完 4096 output tokens 而失敗，但已保存草稿。工具導航改為 low thinking／8192 上限，創作規劃保留原設定；不是單純提升整個流程的預算。
- 以上新短片驗收無法證明 74 支真實素材、複雜身分排除、長訪談字幕、多人交談與多比例完整成片已達人工剪輯品質。這些仍須使用實際 brief 驗收；本次沒有再跑整批 74 支付費分析。

續跑驗收補充：全片複檢曾只收到初始「填滿直式」定調，忘記後來為保留完整畫面而採用 fit 的決策，導致反向要求裁回 fill。現在全片與燒字複檢都接入當前選片決策，明確區分使用者硬條件與可修訂的初始創意定調。修復後重新跑閉環，fit 逐鏡 1/1、全片 approve 且零 issues；這兩次 review 新增 US$0.0107 左右。已通過的全片結果另外依解碼畫面＋聲音＋brief＋當前決策快取，避免 resume 因前輪對話不同重新看片。涉及具名身分參考的複檢仍使用原本完整 request checkpoint，不用省略參考證據的 approval shortcut。

測試中有一次停止進行中的 review，供應商沒有回傳 usage；已在帳本標成不確定費用，不能將可觀測美元合計當成完整帳單。`ask()` 現在也會在 KeyboardInterrupt／SystemExit 中止進行中請求時保存同樣的不確定紀錄，不自動重試。

最終驗收：`pytest -q` 1182 passed（1 個 Starlette testclient 棄用警告）；最後前端改動另通過 `node --check` 與真實瀏覽器驗收。完成剪輯直接開啟時能看到素材脈絡、本版、上一版連結，影片 readyState=4、360×640、沒有播放錯誤。主 CLI 使用 `--budget 0` 再續跑：保留已修好的 fit，逐鏡 1/1；重用相同畫面／聲音／brief 的 approve，正常 exit 0，沒有新增 Gemini 呼叫。預算 0 是禁止新增支出的續跑測試，先前已授權的歷史費用仍保留在 cumulative 帳本中。

驗收介面：http://127.0.0.1:8878/run/editor-acceptance 。此入口由 `artifacts/rewire-web-runs/editor-acceptance/out` 指向同一份 CLI 輸出，沒有另造一套展示資料。輸出仍為 draft，因為未記錄素材權利聲明與製作人核准；未發布到社群，也未 commit／push 本次修改。
