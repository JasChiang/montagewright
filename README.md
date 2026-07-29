# JasCueVideoLab

這是一個**完全獨立、實驗性**的 Gemini 3.6 Flash 影片理解與單幀 Grounding 驗證專案。它不是 JasCue 正式產品，不引用也不修改任何 JasCue 程式碼；實驗未通過前，不應將這裡的程式合併回 JasCue。

## Autonomous delivery V1

`codex/autonomous-delivery-v1` 在原本的 semantic planner → deterministic compiler → local executors 上新增兩個 fail-closed profile：

- `autonomous_strict`：所有 hard evidence、exact cue、geometry、technical QA 與有聲 final QA 都通過，才由 `AUTO_POLICY` 產生 `DecisionAuthorityV2` 與 `delivery_eligible`。
- `autonomous_best_effort`：只允許 policy 預先授權的 optional omission、preferred substitution、Top-K 換帶、two-panel 或 solid matte fit；hard evidence 仍不得省略。所有替代寫入 policy-bound degradation manifest。

## Generalized edit compiler

Autonomous presentation is no longer implemented as a growing product- or
subject-specific fallback tree. The production path now projects the legacy
feature plan into an aspect-neutral `semantic-edit-ir-v1`, measures local
`presentation-scene-facts-v1`, asks every policy-authorized capability to
enumerate immutable options, rejects failed or unknown hard constraints, and
then selects with the shared lexicographic optimizer. Every render manifest
retains generated, rejected, and selected option IDs plus their evidence
hashes.

For each selected trim, the local source-camera estimator decodes a small set
of orientation-corrected exact frames, excludes SAM-track subject regions,
validates background optical flow in both directions, and fits a RANSAC affine
transform. The resulting pan/tilt/zoom/static classification retains frame
hashes, PTS, inlier ratio, residual, confidence, and a cache key. Gemini still
owns semantic attention and movement motivation; it does not guess photographed
camera geometry. Unreliable measurement forbids synthetic pan/zoom and falls
back to static framing or a semantically admissible hard cut.

`editing-capability-registry-v2` is the runtime counterpart of the planner
catalog. A capability declares semantic affordances, required artifacts,
preconditions, guarantees, verifiers, local cost, intrusion rank, executor ID,
and zero paid-call ownership. Adding a new edit operation therefore adds one
executor contract; it does not add another content-case branch.

Gemini function calling is an optional semantic negotiation bridge, not an NLE
remote control. The application exposes only policy-authorized read-only
inspection functions and the non-committing `propose_edit_decision` function.
Execution is manual, budget-reserved, limited to one global negotiation and at
most two tool-result rounds, and accepts only immutable local option IDs.
Gemini cannot send timestamps, boxes, FFmpeg expressions, render commands, or
delivery approvals. Clear local decisions add no Gemini call.

Future dialogue-led editing will add transcript and speaker-turn evidence
providers to the same IR and registry. It is intentionally not part of this
branch. The planned source of word-level local timing is Apple
SpeechTranscriber, followed by a bounded Gemini correction pass that may
correct recognized text but may not rewrite the local timing lineage.

Gemini 只負責語意計畫、既有 frame ID 的事件選擇、exact-frame multi-target grounding 與成片觀察；它不能授予 approval、輸出任意 final timestamp/bbox，或自行形成 repair loop。本機最多允許一次 scoped semantic replan、兩次 full final QA，並在每次付費 final QA 前由 `BudgetLedger` 先 reserve 成本與 interaction。

Autonomous 9:16 final QA 必須取得有聲成片、brief，以及下列自動產生、不可變的 JSON：

```text
editorial-beat-contracts.json
music-map.json
cue-plan.json
exact-event-locks.json
reuse-degradation.json
deterministic-delivery-evidence.json
```

缺少任何一份、policy SHA 不符，或 deterministic evidence 有 hard failure，pipeline 會在新的付費工作前停止。歷史 review artifact 不能因為可播放而被自動升級。

`feature-delivery` 的正常 autonomous 路徑不再要求人工準備這六份檔案。呼叫者提供 policy 與含 `feature_id` 的 beat templates 後，入選的最終 source window 會直接重用 dense-frame decoder、以一次 grouped exact-event call 選既有 frame IDs，將 templates 綁到實際 `EvidenceQueryLockV2`，再寫入 `autonomous-evidence-bundle.json`。`--autonomous-context-dir` 與 `--deterministic-delivery-evidence` 只保留作為既有 hash-bound bundle 的相容 override。

Autonomous profile 的新 benchmark 必須使用全新的 output directory 重新產生
editorial plan；`--reuse-feature-plan-raw-output` 永遠不適用於 autonomous。
`--reuse-feature-plan` 只可繼續同一 output namespace、同一 policy SHA，且原始
direct-video plan 必須綁定目前包含 two-panel／solid-fit／intentional-freeze 的
autonomous capability catalog；舊 simple-production plan 會在任何付費 geometry
工作前被拒絕。需要繼續同一次、輸入 hashes 完全相同的 completed picture run 時，
才使用 `--reuse-picture-result`。Samsung strict fixture 仍禁止 solid matte fit；
無法滿版的候選必須先換 Top-K 或使用 policy 允許的 two-panel，全部失敗則 block，
不得輸出黑邊 fit 冒充正式直式成片。

首個 Samsung 9:16 selected-window run 事後確認重用了歷史 editorial plan 與 picture artifacts，且輸出只有 source audio，沒有走到 music mux；因此它只算 ExactEventLock integration evidence，不是新的 autonomous 試剪或 benchmark。相關 run 已歸檔，不再位於 active artifact namespace。下一次 Samsung benchmark 必須使用全新 output directory 與 fresh music-aware plan，並在交付給使用者觀看前明確產生 music-only audition mux。

後續回歸另修正三個會直接影響觀看結果的執行缺口：

- 短暫且貼住 shot 頭尾的 `camera_shake`／`focus_loss`／`motion_blur` review window
  會連同 200 ms settle padding 編譯成 clean trim；shot 中段的運鏡與 rack focus
  仍只保留 review evidence，不會被自動刪除。
- Gemini 的 `music_target` 會影響合法 cue 類型的選擇；MusicMap source cue 必須
  先投影到實際 MusicAssembly output timeline，picture alignment、exact-event
  delta 與 delivery mux 再共同驗證同一組 spans，避免配樂從中段開始後仍拿原曲
  timestamp 驗片。
- 同一 source 的重用必須有 typed authority 與可觀察理由，且 autonomous V1
  同一 source 最多使用兩次；第三次會在 Grounding／SAM／render 前改試 Top-K。

```bash
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab feature-delivery \
  CATALOG.json BRIEF.json \
  --sam-checkpoint CHECKPOINT.pt \
  --music MUSIC.mp3 \
  --music-map-lock MUSIC_MAP.lock.json \
  --aspect 9x16 \
  --execution-profile autonomous_strict \
  --autonomous-policy fixtures/autonomous/samsung-policy-strict.json \
  --editorial-beat-contracts fixtures/autonomous/samsung-editorial-beats.json \
  --reuse-feature-plan \
  --output-dir artifacts/samsung-autonomous-v1
```

## 一般人也看得懂的工作流程

假設手上有一整批還沒整理的拍攝毛片，這套實驗流程會先幫忙「看帶、整理、提出剪輯建議」，而不是一開始就直接把影片自動剪完：

1. **整理素材**：程式先讀取每支影片的長度、尺寸與切鏡等基本資訊，並建立較輕量的分析版本，不必反覆處理原始 4K 檔案。
2. **AI 看帶**：Gemini 逐支理解影片，整理成可重用的 Clip Card，記錄拍到了什麼、有哪些人物或物件、動作是否完整，以及可能適合放在哪一段。
3. **有指定音樂就讓 AI 實際聽音樂**：本機先找精確節拍、重音、能量與段落；Gemini 的選片規劃 call 必須同時取得音樂與 brief，判斷 opening、build、peak、留白與 closing 適合承接什麼畫面。只有沒有提供音樂時，才允許純視覺選片。
4. **提出選片與相對停留建議**：有剪輯 brief 時，AI 會同時依主題、可觀察資訊量、動作是否完整及音樂 flow 挑選素材；沒有 brief 時，則先根據素材內容提出一版故事方向與候選片段。每段另保存 AttentionProfile，分開記錄閱讀負擔、動作進度、重複壓力、刻意停留價值等理由，再由 RhythmPlan 產生最短／偏好／最長停留範圍。系統不會因為畫面被分類成人物、產品、UI 或靜態鏡頭，就套用固定秒數。
5. **真人確認目標**：如果畫面裡有多個相似人物或物件，系統先提出候選，讓使用者確認真正要保留或追蹤的是哪一個，不讓 AI 在後續步驟自行換成相似目標。
6. **確認真正可用的秒數**：只對入選 shot 逐幀量測黑白格、freeze、解碼／PTS 異常及需複核的失焦、模糊與晃動，先算出連續的安全區間，再分配章節片長；不拿包含髒畫面的整個 shot 長度冒充可用容量。
7. **需要時才追蹤與重構**：一般接片不需要物件座標。只有要把橫式影片改成 9:16、跟隨目標、做有目的的 push-in／pull-out／punch-in，或避讓圖卡時，才從原片抽出清楚影格取得 bbox，再由 SAM 追蹤同一個鏡頭內的目標。多主體不再被要求「每一刻全部同框」：Gemini 可標記 `simultaneous`、`sequential`、`relation_core`、`primary_with_context` 或 `independent_detail`。本機只對當下 phase 的 hard anchors 求滿版構圖；連續移動太快時可在 phase boundary 硬切，且 motion gate 不跨硬切計算。滿版依序嘗試靜態裁切、追蹤、phase virtual camera、可控語意裁切與下一個 Top-K 候選；純色 fit 只可作非交付 review preview，模糊背景不屬於正式能力。
8. **保留連續音樂，不拿毛片原音疊上去**：review delivery 只使用已核准的一段連續音樂，優先保留自然收尾；不把同一首歌切成數段交疊、不 time-stretch，也不混入每顆毛片原音造成突兀重疊。若畫面長度與連續音樂無法在容差內對齊，會停止而不是硬裁或 freeze 畫面。
9. **完成版一定走同一條交付鏈**：`feature-delivery` 依序執行 production picture gate、連續音樂 assembly、最終 mux 與 Gemini 成片 QA。任何一層失敗只會留下可稽核的 blocked／review artifact，不會因 MP4 能播放就宣稱完成；QA 通過後仍須真人核准。
10. **輸出人工審核版**：程式產生 16:9／9:16 review cut、構圖紀錄、卡點建議與失敗原因。Gemini 可用一次有聲 16:9 call 檢查 brief、資訊停留、重複、轉場及音樂 flow；9:16 另以靜音 proxy 只檢查裁切、文字與追蹤。QA 只提出觀察，不會自行改片；真人看過選片、頭尾、裁切及節奏結果並核准後，才適合進一步完成正式剪輯。

```text
一批毛片
  → AI 看帶並建立 Clip Cards
  → 有指定音樂時先建立 MusicMap，並讓 Gemini 同時聽音樂、看 brief 與素材
  → Gemini 依版本化能力目錄提出選片、flow、phase coverage 與相對停留
  → 真人確認選片與重要目標
  → shortlisted shots 以 source FPS 建立 QualitySafeInterval
  → 只有需要直式重構或圖卡避讓時才做 bbox／SAM tracking
  → 使用單一連續音樂段落組裝，不混毛片原音
  → 成片後再做 brief／音樂 flow QA 與 9:16 crop-only QA
  → 輸出可播放的人工審核版
  → 真人修改或核准
```

Clip Cards 建立後可以重複使用。同一批素材之後要剪成不同主題、長度或比例時，可以先查既有資料，只重新分析真正入選且需要精確畫面座標的片段。AI 的選片、時間、bbox、mask 與 confidence 都只是待審建議，不會因為 schema 合法就自動成為正式剪輯資料。

### 新增剪輯能力如何被規劃器使用

正式 production path 不在 Gemini 與 FFmpeg／SAM 之間再放一個自由操作工具的 Agent。它使用三層固定責任：

```text
Gemini semantic planner
  讀取 brief、bounded candidates、音樂與 editing-capability-catalog.json
  只回選片、flow、coverage mode、phase anchors 與可接受的語意取捨
        ↓
deterministic intent compiler
  驗證 capability catalog SHA、phase contract、Top-K 與 fallback
        ↓
local executors
  exact PTS、Grounding bbox、SAM、crop path、motion gate、MusicMap、render
```

`editing-capability-catalog-v1` 是小型、run-level 能力目錄，不是每支素材一份龐大 schema。新增工具時必須同時加入一個可泛化的 capability ID、對應的本機 compiler／executor，以及 success／fallback／adversarial tests。目錄內容的 SHA-256 會寫進 `direct-video-edit-plan-v2`；若工具、限制或 fallback 改變，舊 Gemini plan 不能被靜默套到新執行器。

Direct-video planning 明確區分 `fresh` 與 `alternate` editorial run；後續
`feature-delivery --reuse-feature-plan` 只是把已完成、hash-bound 的選片決策交給
geometry／render pipeline，不代表該決策來自舊版試剪。`alternate` 必須綁定上一版
`feature_edit_plan.json`，只在 frontier 內確實存在替代 evidence 的非 hard beat
要求變更，並產生可重算的 `editorial-freshness-manifest-v1`。Hard evidence 不會為了
表面新鮮度被較差 take 取代。Bounded candidate video 預設深度為每章 3 個；仍受
總秒數 preflight 上限約束，不會用增加 paid interaction 的 speculative fan-out
換取多樣性。

Agent 只適合研究、偵錯、比較策略或協助人類改 brief。正式剪輯由 deterministic compiler 與有界 state machine 執行；未來若接上 semantic replan，最多只允許一次，不形成可自行反覆看片、改策略的 Agent loop。

### 可播放不等於可交付

`feature-cut` 現在把媒體生成與剪輯資格拆開。`horizontal.status=rendered`／
`vertical.status=rendered` 只表示 MP4 已成功編碼；真正狀態另存於
`delivery-eligibility.json`、`render-manifest.json` 與 `run-status.json`：

- `partial`：必要 chapter 沒有直接證據；可以輸出缺證據圖卡供審查，但不能冒充完成 brief。
- `review_preview`：媒體可看，但候選、QualityMap、geometry 或 human-intent execution 尚未完整驗證。
- `ready_for_human_review`：本次 execution profile 要求的自動 gate 已通過，可以進入正式人工審核。
- `delivery_eligible`：review profiles 仍必須有 final-sequence QA 與人工核准；autonomous profiles 則必須有完整 deterministic gates、semantic final QA 與 policy-bound `DecisionAuthorityV2`。單獨執行 `feature-cut` 不會自行宣稱可交付。

預設 `--execution-profile review_preview` 保留實驗便利性。要在付費 geometry／
render 前強制 requested-aspect Top-K 與所有可嘗試候選的 ShotQualityMap coverage，
使用 `--execution-profile production_review`。Human reframe policy 只核准
構圖 intent，不會再直接把某次 Grounding、SAM 或 crop execution 標成通過。

### 為什麼重跑可能仍選到相似素材

同一批毛片、相同 brief 與相同音樂，重跑後仍選中幾個相同 hero shot 不一定是錯誤；若那些鏡頭的內容證據、動作完整度與技術品質明顯最好，穩定選中反而是合理結果。真正需要警戒的是「每章只保存 rank 1」，導致後續發現 9:16 不可行、畫面重複或品質不佳時，renderer 已沒有可回溯的替代素材。

因此正式流程把「召回」和「最後使用」分開：

1. 每章先從 Clip Cards 召回不同 asset／event 的 Top-K，而不是只問一次最高分。
2. 使用者有提供音樂時，Top-K planner 會把同一支實際音樂與 Clip Card evidence 一起送給 Gemini；音樂檔 SHA-256 綁進 projection，之後換歌不能誤用舊排名。排名同時保存語意符合、動作完整、品質、構圖可行性、前後鏡頭差異與重複理由，不過早壓成單一 confidence。
   若完整 Top-K Structured Output 太大，應先保存已驗證的 evidence-bound plan，再用一次精簡的 actual-audio reranker 只選既有 candidate ID。這個降載 call 不能新增素材、frame、entity、bbox 或 region；本機再從上游 artifact 投影回完整執行計畫，避免為了聽音樂重送或重寫全部視覺證據。
3. 16:9 與 9:16 可有不同排名；橫式最好的 take 不必強迫成為直式首選。
4. 實際 geometry 只按排名逐一驗證，第一個通過者即停止；API／tracker 不會為全部候選預先付費。
5. manifest 會列出 Top-K 是否完整、實際嘗試與換帶次數、rank-1 source reuse，以及重複是否有 typed editorial authority。重用不是一律禁止：同來源不同區間、不同構圖重點，或有意的 montage／前後呼應都可以保留；但 planner 必須標明 `distinct_interval`、`alternate_presentation` 或 `editorial_reprise` 與理由。runtime 會在付費 grounding 與 render 前先排除已可由 source interval 證明的未授權重用，再於渲染後以 presentation fingerprint 做完整稽核；無理由補秒數、把重疊區間冒充不同段，或完全相同 presentation 冒充新構圖時，不得成為 `delivery_eligible`。

### 9:16 不是「全部保留」與「隨便裁掉」二選一

直式重構先把可見內容分成三層，而不是針對手機、人物或特定品牌寫死規則：

- **hard core**：理解畫面不可缺少的實例或區域，例如正在完成關鍵動作的肢體、指定主體、必要 UI／文字。必須完整保留。
- **soft extent**：有助構圖但可有限犧牲的上下文，例如人物非關鍵身體邊緣、桌面、環境與動作方向空間。可由 `auto_bounded_clip_v1` 在明確可見比例下裁切。
- **optional context**：不影響 brief 與動作理解的背景或次要物件，可優先犧牲。

實務 fallback 順序是：

```text
Top-K 直式候選
  → 追蹤 hard core，求整段都可行的滿版 9:16 crop
  → 必要時有限裁切 soft extent／optional context
  → 不可行則換同章下一個 take
  → 多主體先嘗試有共同錨點的虛擬鏡頭或下一個候選
  → 最後才依 brief 選擇滿版待審裁切或 scope-preserving fit／layout
  → 仍不確定就停止並交給真人
```

因此「稍微切到背景、衣服邊緣或非必要環境」可以是正確構圖；「切掉臉、指定產品、關鍵操作手、必要 UI／文字」則不能因滿版好看而合理化。Gemini 成片 QA 也會把 geometry safety 與 portrait composition 分開：沒有切到主體只代表安全，不自動代表畫布利用、視覺焦點與 matte fallback 都理想。

Clip Card 現在分成「不可變 Base Card」與按需產生的 `ClipObservationSupplement`。Base Card 只保存跨 brief 可重用的全片摘要、coarse events、Entities、直接可見／可聽證據與一般用途，不再提前決定直式注意力或運鏡。舊卡的空 `portrait_attention_sequence` 會明確視為 `not_assessed`，不能冒充「已確認沒有注意力轉移」；舊卡若已有非空 phase，只投影其中可觀察 predicate 與 Entity 順序，丟棄當時的 camera suggestion。動作邊界明顯不確定的事件可以先由本機提出必要補件；多主體、快速狀態、UI／文字或音訊角色等較昂貴的補看，則只在素材已進入 brief 的 Top‑K candidate frontier 後才觸發。一次 bounded observation 可同時補齊多個 capability，不會為每個欄位各呼叫一次模型。補件只記錄 action structure、relation carrier、result evidence、shared context、readability、audio role 與 aspect-neutral `observable_beats`，仍不包含 source timestamp、bbox、crop 或 camera behavior。

能力缺口採 claim-scoped fail-closed：缺少 `observable_beats` 只會阻擋「依序移動虛擬鏡頭」這個主張，不會把語意上相關的素材從候選庫排除；缺少 readability 或 audio assessment 也只阻擋對應的可讀性或現場音主張。`assessed_absent` 只有完整影片或含事件前後上下文的 bounded video 才能成立，抽樣影格只能證明「在這些影格中沒有看到」，不能證明整段不存在。Base Card 或 event fingerprint 一旦改變，舊補件立即視為 stale；多份 active 補件若對同一 capability 衝突則停止，只有明確 `supersedes` 關係才能取代舊觀察。

本機先用 `scripts/plan_clip_card_supplements.py` 產生零 API 成本的補件計畫；帶入 `--frontier feature-shortlist.json` 時才會為 Top‑K 候選啟用多人、UI／文字等 soft triggers。`scripts/run_clip_card_supplements.py` 會按事件產生含上下文的 720p bounded proxy、以來源 SHA‑256 重用 File API 物件、每個事件最多一次 paid interaction，並保存 request、raw response、usage、cache key 與最終 `clip-observation-supplement.json`。429／503／network error 不會被冒充 schema repair 自動重送。

只有 brief／比例／音樂已知後，選片 planner 才能把已評估的 `observable_beats` 編譯成 `hold`、帶 deadband 的 `follow`、連續 `follow`、`push_in`、`pull_out`、硬切式 `punch_in_cut` 或其他允許策略。Capability 為 `not_assessed`、`assessed_absent` 或 `not_applicable` 時不得自行發明注意力順序；`simultaneous_required`、`shared_context_required` 與 `relative_scale_required` 也會限制 planner 不得用獨立特寫破壞關係或比較。方向仍不是固定的左→中→右，可以是右→左、人物→結果、整體→細節或完全不動。本機會再次驗證任何 `sequential_focus`／`joint_relation` proposal 是否真的有相對應的 observation evidence；不合法的模型輸出不能進入 Grounding。後續仍須對各 anchor 做 exact-frame Grounding、SAM 追蹤、containment、可讀性與速度／加速度／jerk gate，通過後才會產生可執行 `VerticalVirtualCameraPlan`。

舊式 catalog／feature plan 若缺少可執行構圖意圖，`feature-cut` 仍可補做一次 **selected-clip framing refinement**。新的 `direct-video-edit-plan-v2` 路徑已在同一個 brief＋bounded 候選影片＋音樂 planning call 中完成選片、AttentionObservation、ShotFlowIntent、phase coverage 與相對停留，因此不會再付費重看同一片段；下游只做 exact-frame bbox、SAM、identity checkpoint、本機運鏡與 MusicMap boundary reconciliation。v2 的 phase anchors 不會再被全域 required union 覆寫。舊 refinement 僅保留作 legacy／研究相容路徑。

本機不再把每個 phase 的 anchor 中心直接當成相機 keyframe。它會先從該 phase 的 SAM samples 求出能保留 hard anchors 的裁切中心可行區域，再對整段 shot 求最小移動路徑；一個靜止構圖能跨 phase 成立時便維持不動。Gemini 只說明 attention phase、移動理由與語意順序是否可交換，不輸出左／右方向、座標或速度。小於可感門檻的位移會折疊成 hold，沒有理由的移動、無理由折返或缺乏 settle window 的連續 transition 會轉成有稽核紀錄的 hard cut。需要連續移動時使用 quintic smootherstep，並依實際距離檢查速度、加速度與 jerk；tracker 小幅抖動由 hysteresis／deadband 吸收。scale 仍受來源解析度與最多 1.12× 的保守上限約束。

這也不是把「兩個都必須同時看見」改成任意裁切。規劃器以 `simultaneous`、`sequential`、`relation_core`、`primary_with_context` 與 `independent_detail` 表達語意：失去同框就無法證明的接觸、交接或同時狀態不能被拆鏡；可依序理解的角色、UI 步驟或多個焦點則不再全程 preserve-all；比較畫面只需保留真正承載大小、厚度、方向或接觸的 relation core，不能因追求完整物件而犧牲滿版，也不能分別放大造成尺度誤導。所有 full-bleed 方案與 Top-K 都失敗時，renderer 只可輸出 `full_bleed_center_crop_review` 供人工看問題；只有明確要求保全不可分割寬構圖時才另產生純色 fit review，兩者都不冒充 geometry success。

`feature-cut` 是人工審核用成片，但 tracked crop 已不再只看 seed frame。只要實際執行 SAM tracking，本機便依遮擋後重現、切鏡與幾何狀態規劃有限數量的 identity checkpoint，按保存的 source PTS 抽出 exact frame，再讓 Gemini 只判斷 tracked region 是否仍為鎖定身分；它不能修改 bbox、時間或 crop。`mismatch` 會淘汰候選，`ambiguous`／verifier failure 會保留為待審狀態；若整段沒有任何非 seed checkpoint 的必要性，也必須明示 `not_required_by_policy`，不能把未執行的空值當作成功。preferred／soft extent 未達建議可見比例仍屬 advisory；optional region 無法 Grounding 時也只移除該 optional track，不能拖垮 required 主體。`feature-delivery` 會把 identity、coverage、quality、continuous music mux 與 final QA 串成同一條 hash-bound execution chain；即使全部通過仍只會升級到 `ready_for_human_review`，不會替真人做最終交付核准。研究用的 `--allow-unverified-geometry-preview` 則另允許在上游選擇 `primary_center`、hard core 非 atomic／文字／UI／graphic，且 required 最小可見面積至少 90% 時，輸出略裁 hard-core 邊緣的受控預覽；它不會改變 production 的 100% containment 規則。

### 用到哪些技術

| 技術 | 在這個流程裡負責什麼 | 不負責什麼 |
| --- | --- | --- |
| Python 3.12＋`uv` | 執行整套實驗程式、管理套件與可重現的環境 | 不分析影片內容 |
| FFmpeg／ffprobe | 讀取片長、尺寸、旋轉與影格時間；製作 proxy、偵測切鏡、抽原始影格及輸出 review cut | 不理解人物、物件或故事 |
| Temporal Risk Window scanner | 以本機低解析影格差異找出可能被約 1 FPS 粗取樣漏掉的短暫視覺變化，提出需要加密檢查的時間窗 | 不宣稱時間窗內一定有語意事件，也不產生剪點 |
| ShotQualityMap／QualitySafeInterval | 只對入選 shot 以 source FPS 量測黑白格、freeze、相對失焦／模糊、晃動與 PTS／解碼異常，保存 exact PTS 證據；在分配片長前算出最長連續安全區間 | 不把測量值直接當刪除命令；rack focus、whip pan、locked shot 等意圖仍需語意或人工確認 |
| AttentionProfile／RhythmPlan | 保存每段可見資訊、閱讀、動作、重複與情緒停留的分項理由，將 Gemini 的相對停留建議限制在 QualitySafeInterval 容量內，再決定章節邊界的低／中／高轉場壓力 | 不輸出來源 cut timestamp，不用單一「無聊分數」取代可審核的理由 |
| Gemini File API | 上傳並暫存可重用的影片或圖片，避免同一檔案在有效期內重複上傳 | 不執行內容判斷 |
| Gemini 3.6 Flash＋Interactions API | 看完整 proxy、建立 Clip Cards、提出選片與敘事候選；有音樂時同時聽音樂與讀 brief，判斷各章相對停留；在指定單張影格中找出目標 bbox | 不以固定類型秒數取代剪輯判斷；影片時間只適合語意搜尋，不提供 frame-accurate 剪點；單張 bbox 也不是逐幀追蹤 |
| Pydantic Structured Output | 限制模型輸出欄位與型別，拒絕超界時間、非法 bbox 或不存在的 frame ID | Schema 合法不代表模型的內容判斷一定正確 |
| SHA-256＋不可變 frame ID | 確認素材、proxy、影格與模型結果的來源，並把 AI 選中的畫面映射回原片 | 不判斷畫面好不好 |
| Evidence Proposal／QueryLock | 先讓真人確認目標身分、動作條件與構圖需求，再把這份決定鎖定供後續步驟引用 | 不自動創造新目標，也不取代人工核准 |
| Gemini image Grounding | 在 FFmpeg 抽出的原始單張影格上，找出指定人物或物件的 0–1000 normalized bbox | 不跨影格追蹤，也不能把不可見目標的位置猜出來 |
| SAM 2.1（選配） | 以人工或 Gemini bbox 作為 seed，在同一個 shot 內產生 mask 並向前、向後追蹤 | 不理解剪輯 brief，也不應跨切鏡自行延續物件身分 |
| Identity checkpoint | 在固定預算內挑出追蹤起點／終點、遮擋後重現或幾何異常的 exact frames，再驗證是否仍為鎖定實例 | 不修改 SAM geometry，也不能用未執行的檢查冒充通過 |
| 本機 crop solver | 根據整段 tracking、required regions 與畫面邊界計算 9:16 安全裁切路徑 | 不自行決定哪個人物或物件最重要 |
| Gemini VirtualCamera Proposal＋本機 VerticalVirtualCameraPlan | Gemini 依可見資訊順序提出固定、跟隨、依序交接或 joint relation；本機把 Entity 映射成 region，經 Grounding、SAM、containment 與運動 gate 後才投影成 9:16 crop path | 不使用固定左中右模板、不讓 Gemini 直接輸出像素／時間，也不以運鏡掩蓋髒畫面 |
| VirtualCameraPlan | 將已核准的 `hold`、`follow`、`punch_in_cut`、`push_in`、`pull_out`、`recenter` 等意圖投影成有 containment、速度、加速度、jerk 與來源解析度紀錄的 16:9 運鏡；每段另存 sidecar | 不用運鏡掩蓋髒畫面、不憑單一 target 自動執行雙 anchor `pan_reveal` |
| 本機 MusicMap analyzer | 將音訊解碼成 PCM，提出 beat、accent、energy、section 與 ending-hit 候選 | 不理解歌詞、音樂情緒或剪輯 brief；BPM、第一個 downbeat 與 meter 未經真人核准不可執行 |
| Gemini brief＋music selection planning | 使用者有給音樂時，選片 call 必須實際聽音樂並閱讀 brief，提出素材、順序、相對停留與理由 | 不直接產生精確音樂 sample、cut PTS，也不能因總長要求重複或 freeze 片段 |
| Gemini semantic music pairing（進階選配） | 在完成選片後，再把既有 visual event ID 配對既有 music cue ID，增加語意型卡點線索 | 不重新偵測拍點、不輸出精確時間，也不能創造本機 MusicMap 沒有的 cue |
| VisualSyncMap＋CuePlan | 把畫面的 cut、reveal、action apex、ending pose 等事件，在明確 timing window 內對到已核准的音樂 cue；可把 Gemini 配對當排序加分 | 不會為了卡拍暗中截斷 setup／action／result，也不會直接改寫選片、trim、identity 或 geometry |
| Continuous music assembly（正式安全 fallback） | 從核准音樂中選一段連續、可稽核且優先自然收尾的區間；delivery 排除毛片原音，避免疊音 | 不拼接多個音樂片段、不 time-stretch、不為補長度硬切音樂或 freeze 畫面 |
| Reviewed MusicEditPlan V2（實驗） | 讓語意規劃只挑已核准的 section／cue ID，本機再映射成 exact samples；最多四段、三個 join，可使用 hard cut、5–200 ms micro-crossfade、自然結尾或經核准的樂句淡出，並保存 ducking 區間 | 不接受 Gemini 直接提供 sample；不自動 loop、重播重疊來源、time-stretch 或捏造 ending；尚未取代正式 `feature-delivery` 的單段安全路徑 |
| Final Edit QA | 以一次有聲 16:9 review 檢查 brief、動作完整、資訊停留、重複、轉場與音樂 flow；9:16 另做 crop-only review | 只保存觀察與修正建議，不自動重剪，也不能覆蓋本機 geometry gate 或真人核准 |
| Pillow | 把 bbox 或 mask 畫回原始影格，產生方便人工檢查的 debug 圖 | 不參與辨識或追蹤 |
| 本機 HTML／JavaScript review page | 播放事件、候選片段、debug 圖與裁切結果，供真人核准或退回 | 不會因頁面能正常開啟就宣告模型結果正確 |
| pytest | 驗證 schema contract、時間邊界、座標轉換、cache 與 geometry 規則 | 不取代對真實影片的人工觀看 |

簡單來說，Gemini 負責「理解內容與選對目標」，FFmpeg 負責「精確回到原始媒體時間」，SAM 負責「在同一鏡頭裡延續空間位置」，本機規則負責「檢查與裁切」，最後仍由真人決定結果能不能採用。

### 技術上的對應

最新方法採用「先鎖定證據，再驗證 geometry」：未指定 target 時先提出候選，使用者先審核 `EvidenceQueryProposalV2`，再明確核准成不可變的 `EvidenceQueryLockV2`。V2 把持續物件身分（Identity）、只在特定時刻成立的動作／狀態（Predicate）與構圖義務（Framing）分成三份 contract 與 hash；時間 refinement、單幀 bbox、SAM seed 與 layout 因此可各自重用正確層級的證據。自動直式構圖則在一份 planner response 內保留 Top-K 素材候選，只有實際嘗試的候選才由具名自動政策建立 QueryLock 並進入 exact-frame geometry preflight。完整說明見 [METHODOLOGY.md](METHODOLOGY.md)，毛片 coarse-to-fine 全量流程見 [FULL-VERSION-PLAN.md](FULL-VERSION-PLAN.md)。Gemini polygon 與 bbox seed 的舊 A/B 僅保留為唯讀歷史資料；目前支援路徑只使用 Gemini／人工 bbox → SAM。

## 這個專案要驗證什麼

JasCueVideoLab 是可重跑、可計價、可人工稽核的研究工具，目的是分開量測以下能力，而不是用一次模型回答宣稱「AI 已經會自動剪片」：

- Gemini 是否能完整觀看逐支 proxy，產生可重用的 Clip Cards 與 coarse semantic events。
- 有 brief 或無 brief 時，Gemini 是否能從整批 Clip Cards 選出合適 take、敘事順序、時間段，以及針對 16:9／9:16 應保留或可犧牲的內容。
- 從原片 exact frame 取得的 Gemini bbox，是否能正確指定語意實例並成為 SAM shot-local tracking seed。
- 本機 crop solver 是否能利用整段 track 做雙比例構圖，保存每次候選切換、失敗、fallback、成本與處理時間，供真人決定是否採用。
- 同一素材重跑時，選片、事件、時間與 geometry 是否穩定；模型錯誤、429、不可見與不確定狀態是否能 fail closed，而非靜默補值。

Gemini 負責語意選擇，FFmpeg 負責可回映的媒體時間，Gemini image Grounding 負責單張 seed，SAM 負責同一 shot 內的時序 geometry，本機規則負責驗證與裁切，真人負責最終內容和畫面品質。任何模型 confidence、schema 通過、bbox 或 SAM mask 都不是 production ground truth，也不得直接當成 JasCue 正式 SpatialTrack。

目前的最小垂直切片是：

```text
本機影片 → ffprobe / SHA-256 → Gemini File API
        → Gemini Interactions API Structured Content Map
        → 點選 HTML 事件 → FFmpeg 抽 orientation-corrected 原始影格
        → Gemini Interactions API Structured bbox
        → Pillow debug overlay
```

## 毛片挑帶與雙比例粗剪實驗

一般人版本：把一整個拍攝資料夾交給程式後，本機先做有固定編號的低解析度「看帶影片」；Gemini 只挑它看中的編號與說明用途，不負責猜精確剪輯時間。程式再從編號查回原片位置，用 FFmpeg 的真實時間與切鏡邊界取出乾淨片段，分別組成 16:9 與 9:16 人工審核版。

```text
原始毛片（不上傳整批 4K）
  → 本機 ffprobe／SHA-256／每 2 秒代表幀
  → 烙印 immutable frame ID 的低解析 analysis reel
  → Gemini Structured Output 只選 frame ID、用途與直式構圖意圖
  → 本機 frame ID 映射回原片時間
  → FFmpeg scdet 的 decoded-frame PTS 限制片段不跨硬切鏡
  → 輸出 16:9／9:16 silent rough cuts 與 HTML review page
```

這個設計刻意不把 Gemini timestamp 當 cut point。模型回傳的 frame ID 必須存在於 catalog，Pydantic contract 才會接受；實際 `source_in_ms`／`source_out_ms` 由本機資料生成並 clamp 在單一 shot。舊版 rushes rough cut 的 9:16 仍只有 `left`／`center`／`right` 三種固定構圖意圖，不是逐幀 tracking；需要動態構圖時，應改走下方 exact-frame bbox → shot-local SAM 2.1 mask propagation 路徑。

## Full v1：完整逐片 Clip Card，按需才做 geometry

Full v1 不會把整支毛片切成數百張圖片送入模型。每支影片先建立 720p analysis proxy，讓 Gemini 看完整影片並以 Structured Output 寫一張 `MM:SS` Clip Card；音訊採選配，預設 `auto` 是來源有音軌才保留，沒有音軌就只依視覺分析。FFmpeg shot detection 只保存切點資料與每個 shot 一張 960px 中間 JPEG 供稽核。只有使用者或剪輯 brief 選中事件、且確實需要 9:16 reframe／callout／去背時，才從原始影片抽一張 exact frame 取得 bbox，並選配 SAM 2.1。SAM 現在會在初始化前限制到 `允許區間 ∩ seed shot`，但仍需完成下方其他 production-readiness gate 才能作正式剪輯輸入。

```text
毛片資料夾
  → 每支 720p proxy（音訊 auto／off／required）
  → Gemini 完整觀看 → MM:SS Clip Card
  → 本機驗證事件、Entity、target kind 與片長
  → Clip Cards 可重用於不同剪輯 brief

只有選中的事件需要空間座標時：
  → 建立 Proposal（Identity／Predicate／Framing 分層）
  → 人工核准成 QueryLock；自動流程只能引用具名 policy
  → predicate 存在時，在單一 shot 的 4／8 FPS DF catalog 做一次 frame-ID refinement
  → 本機把 DF ID 查回 PTS，再由 FFmpeg 抽原始 exact frame／hash
  → Gemini image bbox（多候選需人工指定）
  → SAM 2.1 bbox-only、shot-local mask propagation

只有快速 UI／短暫狀態不確定時：
  → 本機 Temporal Risk Window scanner 可先獨立於 Clip Card 找出視覺變化窗口
  → 已知事件內再建立 1–5 秒局部 4／8 FPS frame-ID contact sheet
  → Gemini 只選既有 ID；時間仍由本機映射

只有入選片段需要精修頭尾時：
  → Clip Card coarse event ∩ FFmpeg shot
  → 局部 2／4／8 FPS immutable DF IDs
  → Gemini 標記 setup／action／result／hold／reset 與建議 in／exclusive-out ID
  → 本機將 ID 映射為 decoded-frame PTS、半開區間與安全 handles
  → 產生 proposal preview，真人核准後才可套入 feature cut
```

```bash
# 一支完整毛片；預設不做 dense、不做 bbox/SAM
uv run jascue-video-lab full-clip VIDEO.mp4 \
  --output-dir artifacts/full-v1-clip

# 一個毛片資料夾逐片建立可續跑 Clip Cards
uv run jascue-video-lab full-library /path/to/rushes \
  --output-dir artifacts/full-v1-library

# 已有 feature plan 時，只替實際入選的 source clips 建立 Clip Cards
uv run jascue-video-lab full-selected \
  artifacts/my-rushes-run/catalog.json \
  artifacts/my-feature-cut/gemini-plan/feature_edit_plan.json \
  --prepared-library artifacts/full-v1-library-prepared \
  --output-dir artifacts/my-selected-clip-cards

# 有 predicate 時先做一次局部 frame-ID refinement；輸出只引用既有 DF ID
uv run jascue-video-lab refine-query-predicate \
  artifacts/full-v1-library/clips/ASSET_PREFIX EVENT_ID \
  --query-lock examples/evidence-query-lock-v2.json \
  --query-target-id subject.primary \
  --sampling-fps 8 --window-ms 4000 \
  --output-dir artifacts/query-refinement/EVENT_ID

# 只有被選中的事件才抽原始影格並選配 SAM；DECISION_JSON 由上一步 result.json 指向
uv run jascue-video-lab full-ground-event \
  artifacts/full-v1-library/clips/ASSET_PREFIX EVENT_ID \
  --query-lock examples/evidence-query-lock-v2.json \
  --query-target-id subject.primary \
  --predicate-decision DECISION_JSON \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt

# 明確要求某個短暫事件進入 4／8 FPS 局部 fallback
uv run jascue-video-lab full-clip VIDEO.mp4 \
  --dense-event EVENT_ID --dense-fps 8 --dense-window-ms 4000 \
  --output-dir artifacts/full-v1-clip

# 不依賴既有 Clip Card event，先在本機找可能值得加密檢查的視覺變化窗口
uv run jascue-video-lab scan-temporal-risk VIDEO.mp4 \
  --sampling-fps 4 \
  --output artifacts/temporal-risk.json

# 入選事件的 trim intent；只產生待審 proposal，不會自動核准
uv run jascue-video-lab trim-event \
  artifacts/full-v1-library/clips/ASSET_PREFIX EVENT_ID \
  --sampling-fps 4 \
  --editorial-intent '保留完整動作與結果；標記可疑 hold、reset 與品質風險。' \
  --output-dir artifacts/trim-review/EVENT_ID

# 真人看過 index.html／trim-preview.mp4 後，明確核准或拒絕
uv run jascue-video-lab review-trim \
  artifacts/trim-review/EVENT_ID/trim-decision.json \
  --decision approved --reviewer REVIEWER_ID \
  --notes '已確認動作完整，片尾停留可保留。' \
  --output artifacts/trim-review/EVENT_ID/trim-decision.reviewed.json
```

Trim Intent 不把「畫面變靜」直接等同廢尾或刻意留白。模型只能依畫面提出 `natural_pause`、`intentional_hold`、`title_safe_hold`、`clean_plate`、`reset_or_false_end` 或 `uncertain`，並保存可見證據與不確定性；它不能宣稱知道導演意圖。預設流程讓 Gemini 直接觀看完整 proxy，在指定 Clip Card event／shot 內回傳 coarse `MM:SS` 代表性 select；FFmpeg 只抽入點與 exclusive-out 邊界，將其解析到原始影片的 decoded PTS。Gemini proposal 永遠是 `requires_human_review=true`，因此 schema 通過也不會直接改動正式成片。

4／8 FPS dense DF contact sheet 現在是局部升級手段，不是預設 Trim Intent：只有快速手勢、短暫 UI、本機 risk window 或真人對 coarse 邊界有疑義時，才在小視窗內讓模型從既有 exact frame ID 選擇。`scan-temporal-risk` 只輸出 recall-only 視覺變化窗口，會排除已知硬切鏡且不把畫面差異冒充語意事件；目前仍需由後續流程或真人把該窗口配對到事件，尚未自動改寫 Clip Card。不得把整支毛片拆成大量圖片來取代影片理解。若 Gemini 只回傳 hold 的單側端點，系統不會推測另一端，而會捨棄不完整 hold interval 並把 contract normalization 寫入 uncertainties；若 exclusive out 位於片尾且沒有下一張 decoded frame，則保存明確的 end-of-stream time boundary，而不偽造 frame hash。

`--audio-mode auto` 是預設值：有音軌就保留，無音軌也正常完成；`off` 明確移除音訊；`required` 只適合音訊證據不可缺少的實驗，來源沒有音軌時會保存錯誤並停止該片。artifact 會記錄 `source_has_audio` 與 `proxy_has_audio`，Clip Card 不得為 silent source 捏造 audio evidence。

Clip Card response reuse 會驗證 source hash、proxy hash、模型、schema、prompt fingerprint 與實際保存的 raw request；prompt 改變一定重跑。File API cache 以 exact proxy SHA-256 跨 library 共用，並在每次使用前查詢遠端 `ACTIVE` 狀態；不同編碼／解析度的 proxy、原始 4K 與整批 analysis reel 不會互相冒用。成本報告分成本次新增請求 `execution-pricing.json` 與含歷史的 artifact lifetime `pricing.json`。公開 library index 不含使用者名稱、絕對路徑或原始檔名；這些資訊只保存在 gitignored private manifest。

這不代表所有階段 cache 都已達 production 級。Exact-frame Grounding 與 bbox→SAM 已使用包含 source/frame、target、prompt/schema/model、shot bounds、checkpoint 與處理參數的 variant fingerprint；較早的 proxy、shot／dense catalog 仍有部分沿用「檔案存在」式重用。完成全鏈路 fingerprint 前不要在同一 output directory 偷換來源或參數，也不要把 cache hit 當成內容身分已驗證。

若執行環境禁止批次外傳，可先完全離線準備；此模式不建立 Gemini client：

```bash
uv run jascue-video-lab full-library /path/to/rushes \
  --prepare-only --output-dir artifacts/full-v1-library
```

之後在允許連線的環境移除 `--prepare-only` 重跑同一 output directory，會重用 proxy、shot manifest 與 audit frames，只執行尚未完成的 File API／Clip Card 階段。

若批次上傳被政策阻擋，但已經有一份 feature plan，可以先用 `full-selected --prepare-only` 在本機解析實際入選的 clip IDs。此模式只驗證既有 prepared proxies，完全不建立 Gemini client；之後在使用者自己的允許環境，以相同指令移除 `--prepare-only`，依序處理入選素材，而不是重跑整個資料夾。

### Production-readiness gate

目前最可信的輸出是「可搜尋的 Clip Card」與「exact-frame bbox proposal」。下列狀態不代表已形成 production 自動剪輯器：

1. **已完成核心 contract**：SAM predictor 的實際輸入只含 `允許區間 ∩ seed shot`，不跨切鏡傳播。
2. **已完成核心 contract**：多候選不取最高 model confidence；自動 seed 只接受唯一 `matched` candidate，其餘必須人工指定。
3. **部分完成**：QueryLock v2 已把 temporal（identity＋predicate＋catalog）、Grounding（identity＋exact frame）、SAM（identity＋seed／interval）與 framing lineage 分開；較早的 proxy、shot 與部分 dense cache 仍要補齊全鏈路 fingerprint。
4. **已接入 tracked-crop candidate gate，完整交付仍部分完成**：每個新 SAM sample 可回映原始 decoded source PTS；renderer 會規劃 bounded identity checkpoints、抽出 exact source frames、執行 Gemini identity-only verifier 並保存 cache／raw response。`mismatch` 會淘汰候選，`ambiguous` 或 verifier error 不會轉成成功，沒有必要 checkpoint 時則保存明確的 `not_required_by_policy`。遮擋後自動 re-identification、所有影片型態上的 recall 驗證，以及把這些結果升級為全流程 delivery eligibility 仍未完成。
5. **已完成效率／一致性 contract**：同一 shot 內的多個 bbox target 可共用一次 decoded-frame catalog、predictor 與 SAM inference state；每個 target 仍保存獨立 seed、mask、狀態與 provenance。共享與獨立執行可用逐格 mask agreement 自動比較，但 agreement 不是 ground truth。

另外，silent source 不得生成 audio evidence、失敗但已有 usage 的 API response 仍必須計價、公開匯出需採 allowlist sanitizer。完整測試還要加入 non-zero PTS、VFR、B-frame、rotation/edit-list、快速 UI 命中，以及相似物件跨鏡 identity-switch 等 fixture。

```bash
# 一次完成 catalog、Gemini selects、雙比例粗剪、review HTML 與成本／計時
uv run jascue-video-lab rushes-run /path/to/CLIP \
  --sample-interval-ms 2000 \
  --scdet-threshold 4 \
  --output-dir artifacts/my-rushes-run

# 不呼叫 Gemini，只建立可重用的 catalog／analysis reel
uv run jascue-video-lab catalog-rushes /path/to/CLIP \
  --sample-interval-ms 2000 \
  --output-dir artifacts/my-rushes-run

# 單支影片保存 FFmpeg scdet 的精確 decoded-frame PTS
uv run jascue-video-lab detect-shots VIDEO.mp4 --threshold 4 --output shots.json
```

一批經授權的多片毛片已完成端到端 live test，涵蓋本機 catalog、analysis reel、Gemini frame-ID 選片、雙比例 review cut、成本與時間記錄。測試證明 frame ID 可以穩定回映原片，卻不代表模型選片已達人工剪輯品質；實際費用仍取決於素材長度、輸入 token、重跑次數、方案與當時牌價。

兩秒抽樣只適合舊版第一輪粗看帶，不能當成泛用的唯一視覺取樣。Full v1 已改為完整 proxy Clip Card；0.2–0.5 秒 UI、快速手勢與短暫對焦狀態則可在指定事件與單一 shot 內建立 4／8 FPS immutable dense frame IDs。dense fallback 預設關閉，不會把整支影片或整個資料夾全量抽成圖片。

### Brief-ordered feature cut 與安全 Reframe

固定 `left`／`center`／`right` crop 已被實驗性 9:16 輸出證明不可靠：人物或指定物件移動後仍可能被裁掉。`feature-cut` 改以使用者提供的章節 brief 控制敘事順序，Gemini 分別選橫式／直式 take 與明確 reframe target，再以 exact-frame image Grounding + SAM 2.1 mask propagation 約束 16:9 punch-in 與 9:16 crop。

#### Full Auto v2 executor＋selection planner v3

單一模型首選可能在語意上合理，卻在目標比例下無法安全構圖。Full Auto v2 executor 因此要求每個 `supported`／`partial` chapter 保存 2–4 個不同 evidence frame 的候選，而不是只留下 rank 1。selection planner v3 仍由 Gemini 依 brief 決定 source asset、Clip Card event、immutable RF frame、可見證據、品質風險、橫／直策略、簡短 framing intent，以及有序且互斥的 `required_entity_ids`、`preferred_entity_ids`、`sacrificable_entity_ids`。因此模型沒有失去「要剪哪裡、要保留哪個部位」的判斷。

v3 不再要求 Gemini 重抄 rank-1 asset/event/frame、target description 或 verbose resolved regions。本機會把模型選出的 entity IDs 對回一份 hash-bound `selected-clip-card-evidence.json`，確定性補出 target descriptions、相容欄位與 executable region contracts；projection 可由原始模型輸出和這份證據快照完整重現。相較 v2，送入模型的 Clip Card payload 約縮小 30%，response schema 字元數約縮小 44%，也移除了先前造成付費整批重試的 mirror-field 不一致來源。

目前自動 candidate routing 已接到 9:16 路徑：renderer 依候選順序，先核對 asset／event／frame lineage 與單一 shot 邊界；只有真的要跑 geometry 的候選，才由 `policy:full-auto-topk-lazy-geometry-querylock-v2:v1` 建立具真實 `auto_policy` provenance 的 QueryLock。接著從原始來源抽 exact frame，以 identity-only Gemini bbox 建立 SAM seed，完成 shot-local tracking、identity checkpoints 與實際 crop path，最後才執行本機 preflight。selected-clip framing 若 schema 合法但無法維持上游 coverage obligation，會記為候選級失敗並嘗試下一個候選；HTTP 429／quota 類錯誤則仍立即中止，避免用更多付費呼叫掩蓋服務問題。16:9 的 Top-K 也會保存在 schema 與 provenance 中，但目前仍採投影後的選定候選，尚未執行同等的 runtime geometry switching。

構圖需求使用領域中立的 region contract：

- `hard_core`：語意上必須完整保留的區域；來自 `required`。
- `soft_extent`：有助構圖但可以有限取捨的脈絡；來自 `preferred`，並有明確的最小可見比例。
- `overlay_keepout`：後續圖卡或版面不應遮住的區域；來自 `avoid_overlay`。
- `atomic=true`：局部裁切會破壞意義的單一區域，不論原角色為何都視為 hard core，必須 100% 保留。

同一候選可把多個 region 分別 Grounding，並在同一個 SAM session 內建立獨立 track。Crop solver 以 hard-core tracks 求每個 sample 的合法窗口，soft-extent tracks 只影響構圖中心與可見比例稽核；它們不會擠掉 hard core。Preflight 另外檢查來源 lineage、shot 範圍、Grounding／tracking gate、首尾與中段 coverage、hard-core containment、soft-extent floor、overlay keepout、crop speed、acceleration、jerk，以及 source／track／geometry SHA-256。

自動路徑使用版本化 `auto_bounded_clip_v1`：候選必須先用 `preserve_all` 解出 hard core；只有 soft extent 仍高於明列 floor 時，才可標記為 bounded clip。它不授權裁掉 hard core 或 atomic region。相對地，`controlled_clip` 仍必須來自 content-addressed 的真人 policy sidecar；一旦存在此 binding，renderer 會停用自動換候選，完全依真人核准的候選與 edge priority 執行。

失敗會保存 typed failure code 與 recovery action，例如 shot crossing、coverage 不足、hard core 被裁、soft extent 低於門檻、keepout 違規或 crop motion 過快。現行 executor 會依序嘗試 phase full-bleed、距離過遠時的 hard cut、可控語意裁切與下一個 Top-K 候選。所有證據綁定方案都失敗時只產生滿版但未驗證的 review cut；`fit_with_background` 必須是明確要求的非交付 review layout。兩者都不算 production geometry success。

這個設計也控制成本：Top-K 是同一次 planner 回應中的 2–4 個備選，不是把每支影片重送 K 次。Clip Card 可跨 brief 重用；planner 預設只允許一次 text-only Structured Output request，`--repair-attempts` 預設為 `0`。bbox／SAM geometry 採 lazy evaluation，只有已選 chapter 的候選才依序執行，遇到第一個通過 preflight 的候選就停止。Predicate refinement 也是明確指令才執行的一次局部 image call，不會在 Grounding 內暗中 repair；同一 identity＋exact frame 可在 framing 改變後重用 bbox。每次 model request、raw response、usage、prompt/schema/model fingerprint，以及每個候選的嘗試與 geometry fingerprint 都會保存；重跑時每次 response 另存 immutable attempt，canonical 檔不再覆蓋歷史成本。計價會把 `total_cached_tokens` 依 cached-input 牌價和一般 input 分開計算；若某份 response 沒有 usage，會列為未計價 request 並把總額標成不完整下限，不會當成免費。失敗候選仍可能增加 Grounding 費用與 SAM 時間，因此成本報告須以實際 raw usage 與本機 tracker timing 為準，不能只用候選數乘固定牌價。

429／quota failure 不屬於候選內容問題。為了避免隱藏成本，SDK 明確設成每個 Gemini operation 只嘗試一次；若上游回傳真正的 HTTP 429、`RESOURCE_EXHAUSTED` 或 spending-cap error，geometry executor 會立即寫出 `geometry-model-circuit-breaker.json` 並中止整次 render，不再換候選、不再繼續輸出看似完成但沒有 Gemini geometry 證據的 fallback 成片。一般的 target 不可見、tracking coverage 或構圖不可行才會繼續嘗試下一個候選。

Full Auto v2 目前仍有清楚限制：風險導向、固定預算的 identity checkpoint 規劃器、exact-frame extraction、Gemini verifier 與 candidate gate 已接通，但 verifier 只是 bounded semantic checkpoint，不等於逐幀 re-identification，也尚未在完整的真實影片 corpus 上證明 recall。遮擋後自動重新取得 seed 與自動圖卡避讓尚未完成；`overlay_keepout` 在有字卡但沒有 layout solver 時會 fail closed。獨立的成片 QA 可以提出語意 review，但還不是 feature-cut 必經的單一 orchestrator，也不會替 preflight 補造證據或覆蓋 geometry gate。Safe-fit 只是方便人工觀看的預覽，不是核准構圖；模型 rank、confidence、SAM mask 與 schema validation 也都不是 human ground truth。

每個 tracked 9:16 segment 現在保存 renderer 真正使用的 crop keyframes、required-region union、逐時刻合法 crop interval、containment、可見寬度比例、首尾／中段 tracking coverage、crop speed 與 acceleration。裁切器不再先平滑 target 中心後直接裁切，而是先由每一個 required bbox 算出合法範圍，再把平滑路徑投影回該範圍；這可避免平滑延遲把快速移動主體推出畫面。正式路徑中的 `primary_center` 只會放寬 target 外圍的 8% safety margin，不暗中授權裁掉 target；只有前述明示的 review-only preview gate 可接受有量測下限的有限裁切。

Reframe geometry 使用 FFmpeg 自動旋轉後的 display dimensions 做 aspect-preserving cover，不再假設來源一定是 16:9。4:3、直式、超寬與相同比例素材都共用二維 x／y crop solver；manifest 保存來源／縮放／輸出座標空間、兩軸合法區間與實際 crop keyframes。非方形像素來源會先依 FFmpeg frame SAR 還原顯示比例；在 tracking 尚未建立同一顯示座標系前，只能 fail closed 到已正規化 SAR 的靜態 reframe 並標記人工複核。track seed 尺寸、analysis aspect 或多 track lineage 與來源不一致時同樣不能把 normalized 座標硬套進 renderer。

一個構圖可用 `vertical_regions` 分開表達多個 required、preferred 或 avoid-overlay 區域；kind 只使用泛用的 `subject`、`text_region`、`ui_region`、`graphic`、`other`。多個 required 區域會各自取得 exact-frame Gemini bbox，再共用一個 SAM 2.1 video session，逐 sample 合併 union，避免把兩個人物或「人物＋螢幕」寫成一個模糊 target。文字也沒有品牌特例：brief 應指定「必須完整可讀的語意核心」為 required text region，裝飾或整塊容器則可列 preferred。

預設 `vertical_overflow_policy=preserve_all`；required union 太寬、任一 required track 遺失、首尾 coverage 不足或逐 sample containment 失敗時都 fail closed。Gemini plan 的 schema 也只能輸出 `preserve_all`；模型若認為有限裁切值得考慮，只能留下 `vertical_overflow_proposal`，proposal 不具執行權限。

人工若明確接受有限裁切，才可透過 `scripts/apply_reframe_policy.py` 選 `controlled_clip`，並以 `preserve_start`、`preserve_end` 或 `balanced` 指定優先側。腳本不會重跑選片，而會把原始 catalog、preserve-all brief、feature plan、plan binding、選定 frame IDs 與人工 policy 寫入 content-addressed sidecar；修改其中任一輸入，renderer 都會 fail closed。產生的 revised bundle 必須搭配 `--reuse-feature-plan` 使用，manifest 仍保存可見比例與 review requirement，不能把受控裁切冒充完整保留。

```bash
uv run python scripts/apply_reframe_policy.py \
  SOURCE_BRIEF.json HUMAN_POLICY.json REVIEWED_OUTPUT_DIR \
  --catalog CATALOG.json \
  --feature-plan SOURCE_OUTPUT/gemini-plan/feature_edit_plan.json \
  --reviewer "human-reviewer" \
  --review-note "reviewed required-region tradeoffs"

uv run jascue-video-lab feature-cut \
  CATALOG.json REVIEWED_OUTPUT_DIR/brief.json \
  --sam-checkpoint SAM_CHECKPOINT.pt \
  --output-dir REVIEWED_OUTPUT_DIR \
  --reuse-feature-plan
```

`scripts/build_vertical_crop_audit.py` 可把候選、fallback、風險碼、Grounding debug 與實際 crop 軌跡合成審核頁。

若使用者明示不要背景補邊，而 SAM propagation 又不完整，renderer 可把已驗證的 Gemini seed union 當成靜態 anchor，依同一套 edge policy 產生 `seed_anchor_crop`；它比盲目裁來源正中央更接近指定主體，但不代表已驗證 seed 前後的移動，因此固定標記 `motion_outside_seed_unverified` 並要求人工複核。若 required union 本來就比 9:16 寬，仍應優先換 take、調整 required／preferred、拆鏡或選擇 contain／split／PiP；靜態 anchor 不能把幾何上放不下的內容變成完整可見。

```text
使用者功能 brief（文案事實來源）
  → Gemini 只找每章的可見影片證據與 frame IDs
  → FFmpeg shot PTS 決定 source handles
  → 指定主體 exact-frame Gemini bbox
  → SAM 2.1 在單一 shot 內傳播 mask
  → 16:9：剪輯 zoom intent ∩ mask 安全倍率 ∩ 4K→1080 解析度上限
  → 9:16：平滑 tracked crop；strict 與 primary-center 分開記錄
  → 可選字卡 + 原始現場音 + H.264/AAC review cuts
```

SAM 只提供幾何，不自行決定剪輯美學。16:9 的 `none`／`subtle`／`detail` 由 feature plan 表示 editorial intent，實際倍率不得超過 mask 安全值。9:16 的 `strict` 要求完整保留 required regions；`primary_center` 只表示可犧牲未列為 required 的次要 context。正式輸出若要裁掉 required union，仍必須另以真人 `controlled_clip` 明示；研究 preview 可在前述 90% 可見、非 atomic／文字／UI 的窄條件下先產生受控虛擬鏡頭，供 Gemini 與真人複核，不能冒充正式核准。使用者 brief 的規格文字與模型觀察到的畫面證據分開保存，沒有 ASR 或 transcript。

```bash
uv run jascue-video-lab feature-cut \
  artifacts/my-rushes-run/catalog.json \
  BRIEF.json \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --sam-analysis-fps 2 \
  --output-dir artifacts/my-feature-cut
```

預設 `--aspect both` 會輸出兩種比例。只需要 Shorts 時可傳
`--aspect 9x16`，只需要橫式時則傳 `--aspect 16x9`；未要求的比例不會執行
Grounding、SAM geometry、segment render 或 concat，manifest 會明確標成
`not_requested`，避免把不存在的輸出或未發生的模型成本記成成功。

若同一個 output directory 已保存 feature plan，renderer 不會再因檔案存在就自動假設它仍符合目前 prompt／brief。要做只比較裁切器的 controlled A/B，必須明示 `--reuse-feature-plan`；程式會保存舊 plan、目前 catalog／brief／prompt 的 hash，並只重算 geometry 與成片。想重新選片時則使用新的 output directory，不加此旗標。

若某章已有真人核准的 Trim Intent，可重複傳入 `--trim-decision PATH`。Renderer 只接受 `approval_status=approved` 且帶有人類 review record 的 decision，並再次驗證 source SHA-256 與目前 FFmpeg shot；代表性 select 可以位於同一 source shot 中但不包含較早的 coarse RF anchor。proposed、rejected、跨鏡或同 source shot 多筆造成歧義的 decision 會被拒絕。沒有匹配 decision 的章節仍使用原本「keyframe 中心 ± brief duration、限制在 shot」的粗剪方式，manifest 會分別標示 `human_approved_frame_id_pts` 或 `keyframe_centered_requested_duration`，不會把 fallback 冒充成精修結果。

若目的是先產生影片讓真人整體觀看，可明確加入 `--allow-proposed-trim-preview`。這只接受仍為 `proposed` 的可用 decision，輸出 manifest 會標記 `contains_unreviewed_trim_proposals=true`，每段也標成 `unreviewed_proposed_frame_id_pts`；它不能建立人工 review record、不能接受 rejected decision，也不能冒充正式核准 cut。

```bash
uv run jascue-video-lab feature-cut \
  artifacts/my-rushes-run/catalog.json BRIEF.json \
  --sam-checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --trim-decision artifacts/trim-review/event-a/trim-decision.reviewed.json \
  --trim-decision artifacts/trim-review/event-b/trim-decision.reviewed.json \
  --output-dir artifacts/my-feature-cut-reviewed-trims
```

經授權的多功能產品素材已完成 16:9 與 9:16 live review-cut 實驗：橫式版本只在通過 geometry gate 時套用有限 reframe，直式版本則以使用者明確指定的 `strict` 或 `primary_center` 規則處理動態 crop，不使用模糊背景掩蓋構圖失敗。Grounding schema 通過、bbox/contact sheet 經視覺檢查，仍只代表流程可稽核，不代表每個構圖或選片已獲獨立真人核准。

Feature renderer 同樣接受無音軌來源：有原音時保留並淡入淡出，無音軌時為 review segment 明確合成 deterministic stereo silence，讓所有 segment 維持一致的 A/V concat contract；manifest 會標示 `audio_origin=source` 或 `synthetic_silence`，不把靜音說成來源音訊證據。

這個 segment-level 音軌只為了讓 FFmpeg concat contract 穩定，不是最終混音。指定音樂的 delivery 會明確只 map 核准的 continuous music track，排除所有 rush source audio，因此不會出現毛片原音彼此重疊或和背景音樂疊成突兀的雙重音軌。Picture 與 music 長度超過容差時會 fail closed，不以 `-shortest` 的副作用冒充已完成音樂剪輯。

每章停留時間也不是按內容類別寫死。Gemini 先根據 brief、實際畫面、動作完整性、資訊密度與已提供的音樂提出 `recommended_duration_seconds` 與理由；本機只把這些數值當成**相對權重**，再受單一 shot 的合法 source capacity、核准總長與 MusicMap cue 約束。若某個 selected shot 太短，系統只能把剩餘時間分配給其他有足夠合法素材的章節，不能未經授權地重播、freeze 或跨 shot 偷延長；全部容量不足時會先寫出 `editorial-duration-capacity-shortfall.json` 再停止。有剪輯作用的 montage、不同時間區間、不同觀看重點或前後呼應可以作為成片輸出，但必須由 typed reuse policy 明示並在實際 PTS audit 中保存；它們不會被冒充為新增的獨立原始素材。舊 brief 的逐章秒數只作缺少模型建議時的 legacy fallback，不能被解讀成「某類畫面一律停留幾秒」。

#### Quality-safe interval 規劃

`quality_risks: list[str]` 只能提醒人「可能有問題」，無法安全驅動剪輯。新的 P0 路徑會在 Top-K 候選縮小後，才以本機 source-FPS scanner 建立 `ShotQualityMap`：

```text
shortlisted source shot
  → source-FPS 本機品質量測
  → exact-PTS QualityRiskWindow
  → 排除 hard block 與尚未確認為刻意的 trim candidate
  → QualitySafeInterval
  → 各比例最長連續 CandidateCapacity
  → Gemini 相對停留建議 × 本機安全容量
  → exact trim／geometry／render
  → 成片再跑一次 technical QC
```

品質風險分成三種影響：

- `hard_block`：解碼／PTS 完整性等不可安全執行的問題，直接 fail closed。
- `trim_candidate`：黑白格、持續 freeze 等高可信疑點；`intent=unknown` 時不進入自動容量，只有人工或語意證據確認為 `intentional` 才保留。
- `review`：相對失焦、motion blur、camera shake 等可能是 rack focus、whip pan 或刻意手持的訊號；保留片段但標記人工複核。

容量使用「最長連續安全區間」，不會把被髒畫面隔開的 4 秒與 3 秒偷偷相加成可連續使用的 7 秒。例如一顆 12 秒 shot 經本機 source-FPS 掃描後，發現 `00:01–00:02` 失焦、`00:06.800–00:07.200` 被遮擋，安全區間是 `00:00–00:01`、`00:02–00:06.800`、`00:07.200–00:12`，最長連續容量就是 4.8 秒。小數時間來自 decoded frame ID 對回原始 PTS，**不是 Gemini 輸出的小數秒**；Gemini 只提供 `MM:SS` coarse window 或選擇本機建立的 immutable frame ID。

真人已核准的 Trim Intent 會在片長分配前鎖定為 exact duration；若穿過被排除的品質區間，renderer 會拒絕，而不是事後縮短、停格或延長其他髒尾巴。若 runtime 候選無法承接已規劃的長度，目前 v1 會換下一個能完整承接的 Top-K 候選或停止；尚不會在 renderer 裡靜默做全域縮時重排。

```bash
# 先保存 FFmpeg shot PTS，再只掃描入選 shot
uv run jascue-video-lab scan-shot-quality SOURCE.mp4 \
  --shot-id shot-0003 \
  --shot-manifest-output artifacts/quality/shot-manifest.json \
  --output artifacts/quality/shot-0003.quality-map.json

# 由同一份 evidence 建立雙比例的連續安全容量
uv run jascue-video-lab build-candidate-capacity \
  artifacts/quality/shot-0003.quality-map.json \
  --candidate-id candidate-003 \
  --preferred-duration 5.5 \
  --minimum-duration 2.0 \
  --output artifacts/quality/candidate-003.capacity.json

# feature-cut 要求所有可能實際嘗試的 source shot 都有 quality map
uv run jascue-video-lab feature-cut CATALOG.json BRIEF.json \
  --sam-checkpoint SAM_CHECKPOINT.pt \
  --execution-profile production_review \
  --shot-quality-map artifacts/quality/shot-0003.quality-map.json \
  --output-dir FEATURE_OUTPUT
```

這個 scanner 是 deterministic guardrail，不是通用美學判官。focus／shake 目前仍是 shot-relative heuristic；目標是否可見與各比例是否放得下，要等 QueryLock、tracking 與 geometry preflight 才能加入 capacity。`scan-temporal-risk` 仍只負責低成本 visual-change recall，不能替代這條品質路徑。

#### Attention、Rhythm 與 Virtual Camera

P1 不再把 Gemini 的 `recommended_duration_seconds` 當成孤立數字。每章可另外保存一份 `attention_observation`：

- `minimum_dwell_seconds`：辨認主體、讀完必要文字、完成動作或保留必要情緒所需下限。
- `recommended_duration_seconds`：模型基於本次可見證據提出的偏好停留。
- `maximum_dwell_seconds`：在資訊開始重複前仍合理的上限。
- semantic novelty、action progress、reading load、unresolved tension、emotional hold、repetition pressure 與 music transition opportunity 等分項證據。

本機先將 maximum clamp 到 QualitySafeInterval 的連續容量，再建立 `AttentionProfile` 與 `RhythmPlan`。RhythmPlan 只提供章節時長上下限和 boundary pressure；它沒有 source timestamp 欄位，不能自行產生 frame-accurate cut。舊 plan 沒有完整 attention vector 時，未知欄位保持 `null`，不以規則偽造模型分數。

P2 將 reframe 從固定倍率升級為可稽核的 `VirtualCameraPlan`。16:9 與 9:16 都能使用 Gemini 基於素材證據提出的 `hold`、`follow_deadband`、`follow`、`punch_in_cut`、`push_in` 或 `pull_out`；多 anchor 直式鏡頭另可依 phase 順序交接注意力。Gemini 另需標記 `movement_motivation`，並以 `semantic_order_locked`、`spatially_optimizable` 或 `no_continuous_traversal` 說明本機可否優化構圖位置；任何 policy 都不能交換影片中的事件時序，也不指定字面方向。實際 keyframe scale、可行 center region、全段最小移動路徑、containment、hysteresis、速度、加速度與 jerk 都由本機 track 和 geometry compiler 決定。微小位移只有在跨區間共同靜態構圖仍合法時才會折疊成 hold；自動改成 hard cut 則另需 `cut_admissible` 證明不會切斷關係、動作、UI 狀態或閱讀。每個執行結果保存 sidecar、track fingerprint、實際 crop movement episode 與 traversal audit。這套 16:9 剪輯運鏡與 9:16 版型 reframe 共用 tracking evidence，但分屬不同 editorial contract。

若使用者授權的交付範圍是 60–90 秒，而 Gemini 的各章 AttentionProfile 在 QualitySafeInterval 內的最大連續容量總和略短於 brief 偏好秒數，可明示 `--allow-shorter-within-delivery-range`。程式只會把 project duration 降到仍在交付範圍內的 attention maximum，並保存 `project-duration-resolution.json`；它不會重播、停格、穿過髒畫面或偷偷延長任何章節。容量依 requested aspect、候選與 evidence anchor 分開計算：只計算包含該 anchor 的連續 safe interval，並以至少一個可執行候選能承擔的容量規劃；runtime 換候選時仍須重新確認該候選能承擔已配置時長。若已核准的同一首音樂長於縮短後的 project timeline，scheduler 可在使用者明示允許縮短交付時使用其合法 prefix cues 來調整畫面章節邊界；`feature-delivery` 接著強制由 assembly 產生單一連續區段與自然淡出，不切碎、交疊或 time-stretch 音樂，再對 final mux 執行成片 QA。任何長度或 lineage 不一致都會 fail closed。

```bash
uv run jascue-video-lab feature-cut CATALOG.json BRIEF.json \
  --sam-checkpoint SAM_CHECKPOINT.pt \
  --rhythm-style standard \
  --shot-quality-map artifacts/quality/shot-0003.quality-map.json \
  --output-dir FEATURE_OUTPUT
```

`calm`、`standard`、`energetic` 只調整 boundary-pressure 分級，不會改寫 Gemini evidence、突破 min/max dwell、跳過動作完整性或把所有剪點吸到 beat。

若 geometry 與片段已經渲染，只想比較另一種敘事順序，不需要再呼叫 Gemini。`scripts/resequence_segments.py` 讀取明確的 trim/sequence JSON，重新編排現有編號 A/V segments，並輸出包含每段來源、trim 與新時間軸的 manifest。這只適合可稽核的 picture-edit A/B；它不會把既有片段描述冒充成新 Full Clip Card，也不能取代原片層級的 take selection。

完整的 Clip Card-driven A/B 則分成兩次 Gemini 任務：第一輪逐片產生 Clip Cards；第二輪只讀已驗證 Clip Cards 與使用者 brief，輸出 Structured narrative plan。`scripts/plan_selected_clip_cards.py` 實作第二輪，`scripts/render_clip_card_narrative.py` 只從通過 evidence gate 的 source/event/MM:SS 建立 16:9 review cut。第二輪仍可能產生規格換算錯誤，Clip Card 也可能把局部可見的數字或型號字元誤判成另一個相似值。因此任何 OCR／身分衝突只能觸發 `needs_human_review`，必須回查 orientation-corrected 原始影格後才能採用或排除；schema validation 不能取代 claim validation，也不能把模型 OCR 當成 ground truth。

`scripts/plan_clip_card_feature_cut.py` 將這個方法延伸到完整 feature cut：模型可閱讀整個已驗證 Clip Card library，但只能選 catalog 中既有的 asset／event／entity／RF frame ID；本機會再次驗證影格確實屬於該素材、位在事件區間，且每個 brief-specific entity priority 都能回溯到 event，再以 hash-bound Clip Card evidence 投影出 `feature-cut` 可使用的 target 與 region contract。選片階段不產生 bbox 或剪點，只有真正入選、需要動態構圖的區間才執行 exact-frame Grounding 與 SAM。新版保留每章 2–4 個候選，9:16 renderer 會先試可驗證的 tracked candidates，再考慮 planner 明列的 safe-fit；所有候選均失敗時只輸出待審 preview，不會把中心裁切冒充成成功追蹤。

當素材庫大到無法在單次 narrative planner request 中穩定放入所有 Clip Card 時，先以 `scripts/shortlist_clip_card_feature_candidates.py` 做一次 text-only 階層召回：每個 brief chapter 只保留可回溯至原始 Clip Card 的少量候選，再交給完整 planner 決定順序、framing intent 與 Top-K。這不是只取 rank 1 的捷徑，也不會跳過後續 evidence／geometry gate；它把「高召回找素材」和「跨章節敘事與構圖決策」拆成兩個可稽核任務，避免一個超大回應失敗後整批重送。

正式的簡化路徑不要求 Gemini 預先替所有素材補寫動作節點、閱讀負荷、關係或運鏡資料。Clip Card 只作為可重用的素材索引；text-only shortlist 完成後，`--candidate-video-evidence` 只把每章前幾名候選裁成有上下文的短代理片段，連同實際音樂交給**同一次**剪輯規劃 call。Gemini 的回應只負責：

- 挑選各章候選並說明可見證據與風險；
- 建議相對停留時間和章節順序；
- 指出橫式是否值得推近；
- 指出直式必須保留、可犧牲的可見 entity，以及注意力應固定、跟隨或依序轉移。

Gemini 不回傳 bbox、mask、逐幀座標、精確剪點、運鏡速度或 easing。本機仍以 decoded PTS 決定真正 trim，僅對入選片段執行 exact-frame Grounding、bbox-seeded SAM、containment 與 virtual-camera motion gate。若 bounded videos 超過預設總秒數，程式會在上傳和付費規劃前停止；重新規劃會依 proxy SHA 重用 File API object，但必須建立新的 model response，不能重播不相容的舊 raw output。

```bash
uv run python scripts/plan_clip_card_feature_cut.py \
  CATALOG.json BRIEF.json PREPARED_CLIP_CARDS PLAN_OUTPUT \
  --shortlist FEATURE_SHORTLIST.json \
  --music MUSIC.wav \
  --candidate-video-evidence \
  --candidate-video-depth 2 \
  --maximum-candidate-video-seconds 360
```

這條 production path 的付費語意階段預期是「一次 shortlist、一次看 shortlisted videos＋聽音樂的 edit plan、必要時一次 final QA」；`clip_card_observations.py` 與 supplement runner 保留給研究和針對性診斷，不是正式剪輯前的全庫必經步驟。

Clip Card plan 轉成 renderer plan 時會另寫不可變的 external-projection sidecar，保存來源 catalog、brief、模型 request／raw response、projection contract 與輸出 plan 的 hash。candidate override 也必須接續並驗證這條 provenance；任一上游內容改變就 fail closed。早於此 contract 的舊 artifact 不可手動複製 plan 冒充可重用結果，必須從仍保存的原始 artifact 重新投影。

`scripts/plan_clip_card_open_edit.py` 是沒有內容 brief 的對照實驗：只給 60–90 秒與雙比例等操作限制，讓 Gemini 從完整 Clip Card library 自行推論主題、時間軸位置與每格 2–4 個候選。新版 evidence payload 也保留 Entity kind、required／optional／avoid-overlay 關係，讓模型可產生泛用 `vertical_regions`，而不是把多個獨立主體合寫成一個 bbox target。局部 Trim Intent 可能為保留完整動作而使成片超過模型原先配置的秒數，因此 `scripts/reconcile_open_edit_budget.py` 另讀實際 segment durations，只以 keep／drop／reorder 完整片段把全片拉回 duration contract；它不會在動作中間靜默截短。

Planner 的 JSON Schema 無法完整表達所有跨欄位 invariant；模型若同時填入互斥但可保守消解的欄位，本機只允許不增加執行權限的 canonicalization，例如明示 `original` 時清除 zoom／focus，或將 required／atomic region 收緊成完整可見。原始付費 response 與原始 request 永遠保持不變，逐 JSON path 的 before／after／rule、canonical output 與兩邊 hash 另存；無法安全消解的矛盾仍 fail closed。`--reuse-raw-output` 只會重投影完全配對的 request／interaction／raw-output，fresh paid run 也拒絕覆寫既有 artifact namespace。

當 9:16 audit 證明 hard-core union 不可容納或 tracking coverage 不足時，自動路徑會先換同一 chapter 的下一個 evidence-bound candidate，而不是立刻套背景補邊。`scripts/apply_open_edit_candidate_overrides.py` 仍可接受人工審查過的 `feature_id + aspect + candidate_id + reason` patch，保留原始 OpenEditPlan 與兩邊 hash，再重新投影 brief／feature plan／trim plan；候選不存在或同一 aspect 重複覆寫會 fail closed。人工 override 與自動 candidate routing 是不同權限層，不能互相冒充。

成片 Gemini QA 採成本分級，不是每次 render 都重看全部。所有片段先跑零 API 成本的本機 geometry／coverage／media gate；只有文字／UI、多 required targets、controlled clip、tracking risk 或 fallback 段落需要語意複核。`scripts/verify_feature_cut.py` 可把完成版 9:16 壓成 720×1280 proxy，以一次 `gemini-3.6-flash`、`thinking_level=low` Structured Output call 檢查主體身分、重要文字、語意是否符合與重複／突兀問題；它不回時間戳、不驗證 frame-accurate geometry，也不會自行改剪。相同 render／manifest／prompt／schema／model 會重用結果，最終仍由真人核准。若 schema 驗證失敗或重試，每一次 request、raw response、錯誤、timing 與 pricing 都保存於不可覆寫的 attempt 目錄，總成本會聚合所有實際有 usage 的請求。

Gemini 的成片 `pass` 不可覆蓋本機幾何證據。QA validator 會把 required-region coverage、完整 containment、controlled clip、fallback 與 source-edge 診斷一起納入本機最終狀態；只要其中一項需要複核，即使模型認為主體「看得見」，validated status 仍固定為 `review`。feature-cut 另把本次新增或改變的 raw interactions 記在 `pricing.incremental.json`，與包含歷史快取的 `pricing.json` 分開，避免把舊請求重複算成本輪花費。

主要影片／圖片辨識請求另使用 Interactions API `system_instruction` 建立 evidence-only 邊界：本次媒體與明確 metadata 是唯一證據，禁止以模型記憶、常見名稱、相似外觀或「最可能答案」補完品牌、型號、數字與 UI 文字。Full Clip Card prompt 也要求任一關鍵字元不清楚時改用泛稱並保存 uncertainty。控制 A/B 曾觀察到舊 prompt 以先驗補完一個相似但錯誤的型號；改用 domain-neutral 規則後該欄位在重跑中恢復正確，模型卻又把另一處模糊小字補成畫面不存在的規格。這證明 prompt guardrail 不是 ground truth：單一正確 claim 不代表整張 Clip Card 都正確，衝突與重要文字仍需 exact-frame 驗證及人工核准。

`scripts/verify_clip_card_text.py` 實作不覆蓋原始 Clip Card 的文字驗證：從原片抽多張 exact frames、保存 PTS／hash、裁出文字證據，以 `resolution=high` 分別做 blind transcription，再以明列 `other`／`unreadable` 的候選式請求交叉檢查。方法不一致時輸出 `needs_human_review`；只有人工核准後才能另外產生 reviewed Clip Card。

## 音樂卡點 MVP

音樂卡點採獨立 evidence chain。有指定音樂時，正式順序是 **music-first**：先建立經核准的音樂網格與暫定敘事節點，再選片；成片後的 visual-event 對齊只負責 QC 與局部 refinement，不再用來補救一條完全沒按音樂規劃的固定時間軸。

```text
本機音樂檔
  → FFmpeg 解碼單聲道 PCM
  → MusicMap Proposal：beat／accent／energy／section／ending-hit 候選
  → 真人核准 BPM、第一個 downbeat、meter
  → immutable MusicMap Lock

editorial brief（尚未選片）
  → Brief VisualSyncMap：章節意圖與可調整時間窗，不冒充影片證據
  → 有音樂時：Gemini 選片 call 必須同時聽音樂、讀 brief、看 catalog
  → Gemini 提出選片、順序、相對停留與音樂角色理由
  → 進階選配：再配對既有 visual intent ID 與 music cue ID
  → 全局 CuePlan scheduler
  → 真人核准成 immutable CuePlan Lock
  → CuePlan 的章節長度與音樂策略進入 feature-cut
  → Gemini 才從 catalog 選擇符合內容、動作與節奏的素材

完成 feature-cut 後的 render manifest
  → VisualSyncMap：cut／chapter start／ending pose／已執行的虛擬鏡頭交接
  → 可另外加入經證據確認的 reveal／action apex／UI change
  → 選配：Gemini 聽音樂並把既有 visual ID 配對既有 cue ID
  → 全局、順序保持的 CuePlan scheduler
  → CuePlan Proposal＋HTML 人工審核
  → 真人核准成 CuePlan Lock
```

零成本 baseline 全部在本機執行，不呼叫 Gemini。分析器只提出聲學候選，不把 `section_001` 冒充為 verse、chorus 或 drop；human review 之前，beat grid 不具執行權限。`narrative`、`balanced`、`montage` 三種 preset 只改變 section／downbeat／accent／一般 beat 的排序權重，不改變素材語意。

有指定音樂時，Gemini 的主要選片規劃請求不再是選配：它必須實際取得音樂、brief 與 catalog，否則不得宣稱做了 music-aware edit。若要進一步減少規則式卡點的機械感，才選配第二次 `gemini-3.6-flash` 音樂語意配對。第二次 call 會取得已核准的 MusicMap cue IDs 與 Clip Card／render manifest 衍生的 visual event 語意，只回答「哪個 visual event 適合哪些既有 cue IDs」，不能自己發明秒數。程式只傳送至少落在一個 visual event 合法 timing window 內的 cue，避免把數百個永遠不可能採用的拍點塞入 prompt。音樂以 SHA-256 cache，不會對每個鏡頭重送；最終 sample-accurate 位置、合法 timing window、全局順序與 hard gate 仍由本機決定。

Music-aware delivery 預設保留一段連續音樂區間，優先選擇 phrase-aligned start 與自然 ending；`join_count` 必須為零，`internal_music_edits` 必須為空。畫面章節邊界以及 geometry 已驗證的虛擬鏡頭交接，可在合法窗口內配對 MusicMap 的 section／downbeat／accent；它們是可排序的 VisualSync 候選，不是「每拍都移動」的命令，也不能為卡拍截斷 setup／action／result，或把音樂切碎後交疊拼接。成片完成後，canonical 16:9 會以一次有聲 Structured Output QA 檢查 brief delivery、停留、重複、轉場與 music flow；9:16 另用靜音 proxy 做 crop-only QA，避免把音樂好聽誤判成構圖正確。Gemini QA 只提出 observation／revision suggestion，不會自動覆蓋本機 gate 或直接重剪。

```bash
# 1. 本機分析音樂；輸出 proposal，不會自動核准
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab analyze-music MUSIC.wav \
  --output-dir artifacts/music-demo

# 2. 真人確認；可覆寫 BPM、第一個 downbeat 與拍號
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab review-music-map \
  artifacts/music-demo/music-map.proposal.json \
  --reviewer "human-editor" \
  --decision approved \
  --bpm 120 \
  --first-downbeat-ms 240 \
  --meter 4 \
  --output-dir artifacts/music-demo/reviewed

# 3. music-first：尚未選片前，先從 brief 建立 provisional visual intents
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab build-brief-sync-map \
  FEATURE_BRIEF.json \
  --aspect 9:16 \
  --default-flex-ms 3000 \
  --target-duration-ms 81150 \
  --output artifacts/music-demo/brief-sync-map.json

# 接著以既有 plan-semantic-music → plan-music-cues → review-cue-plan
# 建立 approved CuePlan Lock，再把它放在 feature-cut 的選片之前：
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab feature-cut \
  CATALOG.json FEATURE_BRIEF.json \
  --sam-checkpoint SAM_CHECKPOINT.pt \
  --music-first-cue-lock artifacts/music-demo/preselection/reviewed/cue-plan.lock.json \
  --output-dir FEATURE_OUTPUT

# 4. 成片後 QC：從 render manifest 建立 exact visual events。
# 預設 flex=0，因此只做唯讀稽核。
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab build-visual-sync-map \
  FEATURE_OUTPUT/render-manifest.json \
  --aspect 9:16 \
  --output artifacts/music-demo/visual-sync-map.json

# 若操作者明確允許 boundary 前後各移動 250 ms，才可另建有 window 的 proposal：
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab build-visual-sync-map \
  FEATURE_OUTPUT/render-manifest.json \
  --aspect 9:16 \
  --default-flex-ms 250 \
  --output artifacts/music-demo/visual-sync-map.flex-250.json

# 5. 產生成片後 CuePlan 與可播放的 HTML review；尚未修改影片
# 選配：先讓 Gemini 做一次音樂—畫面語意配對
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab plan-semantic-music \
  MUSIC.wav \
  artifacts/music-demo/reviewed/music-map.lock.json \
  artifacts/music-demo/visual-sync-map.flex-250.json \
  --output-dir artifacts/music-demo/semantic-pairing

UV_CACHE_DIR=.uv-cache uv run jascue-video-lab plan-music-cues \
  artifacts/music-demo/reviewed/music-map.lock.json \
  artifacts/music-demo/visual-sync-map.flex-250.json \
  --preset balanced \
  --semantic-pairing artifacts/music-demo/semantic-pairing/semantic-music-pairing.proposal.json \
  --music MUSIC.wav \
  --video FEATURE_OUTPUT/renders/feature-cut-9x16-clean.mp4 \
  --output-dir artifacts/music-demo/cue-plan

# 6. 真人核准 hash-bound CuePlan
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab review-cue-plan \
  artifacts/music-demo/cue-plan/cue-plan.proposal.json \
  --reviewer "human-editor" \
  --decision approved \
  --output-dir artifacts/music-demo/cue-plan/reviewed
```

目前 MVP 的正式路徑完成卡點分析、排程、稽核與 lock；`scripts/render_music_cue_preview.py` 另可產生清楚標示為 **unapproved review preview** 的 A/B 影片，方便真人直接聽看 Gemini 語意配對是否改善節奏。preview 只在已授權的小窗口內對既有 segment 做有限 retime 並替換音樂，audit 會保存每個邊界的目標 cue、實際位移與變速比例；它不是 production RenderPlan，也不得冒充經 source-handle-aware re-trim 的正式成片。

這是刻意的 fail-closed 邊界：render manifest 只有既有 segment duration，不能證明把 cut 移動 250 ms 仍保留完整 setup／action／result、合法 source handle、同一 shot、可用構圖與片尾 hold。正式下一階段必須讓經核准的 Trim Intent 提供 action-safe timing window，通過 geometry preflight 後才可將 CuePlan 套入新的 RenderPlan；不會直接用 `setpts` 變速或裁掉動作來製造「有卡拍」的假象。

正式測試建議直接用單一交付命令。它會強制使用 `production_review`，自動補齊入選候選的 ShotQualityMap，完成 picture→continuous music→final mux→FinalEditQA；輸出最高只會是 `ready_for_human_review`：

```bash
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab feature-delivery \
  CATALOG.json BRIEF.json \
  --sam-checkpoint CHECKPOINT.pt \
  --music MUSIC.wav \
  --music-map-lock MUSIC_MAP.lock.json \
  --aspect both \
  --output-dir artifacts/my-delivery
```

### 音樂不是硬截一段：MusicEditPlan V2

正式 `feature-delivery` 目前仍以單一連續音樂區間作為 fail-closed 預設；這能避免第一次自動剪輯就產生不自然的拼歌。實驗性的 `MusicEditPlanV2` 則補上實務剪輯需要的下一層語法：

1. Gemini 或剪輯師只能選擇 reviewed `section_id`、`locked cue ID`、段落語意角色與能量關係。
2. 本機依 `MusicMapLock` 將 ID 映射成 exact samples，不接受模型自行提供 sample 或毫秒。
3. 相鄰 passage 明確選擇 hard cut 或 5–200 ms micro-crossfade；join 數量上限為三個。
4. 結尾只能保留原曲自然結尾、在核准邊界做短 fade-out，或使用已核准 `ending_hit`。
5. 對白、旁白、UI 重點需要讓位時，可保存 typed ducking region；不把任意音量自動化藏在 FFmpeg command 裡。
6. V2 plan 永遠標記 `requires_human_review=true`。在完成聽感 A/B、final mux lineage 與 FinalEditQA integration 前，不會自動升級成正式交付路徑。

這裡的 micro-crossfade 只用來消除接縫 click 或柔化已核准的樂句交界，不是把兩段不相容的音樂「糊」在一起。能量、樂句與 ending 不成立時，正確結果是改選 passage 或退回連續音樂，不是增加更長的 crossfade。

### 虛擬鏡頭的 cut 不是高速平移

多主體 phase 若沒有足夠時間完成自然平移，系統會將轉場改成 hard cut。Motion gate 現在會分別量測 cut 兩側的連續攝影機路徑，不再把兩個構圖間的瞬間跳變算成超高速 velocity／acceleration／jerk。這能避免「本來可用兩個滿版構圖硬切，卻因假性速度超標退回黑邊 fit layout」。

若滿版連續運鏡仍不可行，泛用 fallback 順序是：已驗證的構圖 hard cut → establishing／共同 context → phase-aligned sequential views → 經 contract 允許的 controlled clipping → 下一個候選 → 最後才是 fit/layout。不能證明必要關係時則 fail closed，不會用漂亮運鏡掩蓋證據缺失。

完整的剪輯語法與加速策略記錄在 [EDITORIAL-GRAMMAR-V2.md](EDITORIAL-GRAMMAR-V2.md)。

需要逐層研究時，仍可用下列獨立工具組裝連續音樂與執行唯讀 QA。組裝器會驗證 MusicAssembly plan／binding／render manifest、實際音樂 hash 與 picture／music 各自的 stream 長度；任何證據不一致或超出容差都會拒絕，不會用 `-shortest` 暗中截斷。QA 會保存影片、manifest、brief、prompt、schema、raw response、usage 與成本 hash，重跑相同輸入不會再次付費：

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/assemble_music_delivery.py \
  FEATURE_OUTPUT/renders/feature-cut-16x9-clean.mp4 CONTINUOUS_MUSIC.wav \
  --music-assembly-artifacts MUSIC_ASSEMBLY_ARTIFACT_DIR \
  --aspect 16:9 \
  --output DELIVERY/feature-cut-16x9.mp4 \
  --manifest DELIVERY/feature-cut-16x9.manifest.json

UV_CACHE_DIR=.uv-cache uv run python scripts/run_final_edit_qa.py \
  canonical_16x9 DELIVERY/feature-cut-16x9.mp4 \
  FEATURE_OUTPUT/render-manifest.16x9.json QA/16x9 \
  --brief FEATURE_BRIEF.json

UV_CACHE_DIR=.uv-cache uv run python scripts/run_final_edit_qa.py \
  crop_only_9x16 DELIVERY/feature-cut-9x16.mp4 \
  FEATURE_OUTPUT/render-manifest.9x16.json QA/9x16
```

## 重要界線

- `start_ms`、`end_ms` 與 `recommended_keyframe_ms` 是 **coarse semantic time**，只用於搜尋與人工瀏覽，不是 frame-accurate cut point。
- 對 Gemini 原生影片理解索取少量截圖候選時，API contract 使用官方文件慣例 `MM:SS`，不要求模型計算毫秒。程式只把合法且未超界的 `MM:SS` 換算成 FFmpeg seek 值；換算結果仍不是精確 frame time。
- `frame_pts` 與 `frame_time_ms` 是 FFmpeg 實際抽到之原始影格的媒體時間；每張 `frame.png` 都另外保存 SHA-256。
- Gemini bbox 是單張影格的人工審核 proposal，不是 pixel mask，也不是 production-ready tracking data。
- `main` baseline 沒有 ASR、transcript、字幕、temporal tracker、SAM/EdgeTAM/Apple Vision、逐幀追蹤、自動裁切、NLE timeline、FCP/Motion/FxPlug 或成片輸出。
- `experiment/dynamic-tracking` branch 另有一條明確隔離的 optional CSRT bbox propagation 實驗。它不屬於 baseline，也不得把輸出稱為 Gemini 原生 tracking 或正式 SpatialTrack。
- `experiment/sam21-video-segmentation` branch 把 Gemini／人工 bbox 當語意 seed，交由 SAM 2.1 產生並傳播 mask；原始 seed、SAM prompt box、mask 與 mask-derived bbox 分開保存。
- `experiment/gemini-segmentation-seed` 曾測試 Gemini 原生 polygon 作為 SAM mask seed。A/B 後已從目前主路徑退休：執行入口會拒絕 polygon，歷史 artifact 只用於說明為何選擇 bbox seed。
- Live 成本與時間資料會分開記錄 analysis proxy、Gemini raw usage 牌價估算、API latency 與 tracker geometric drift。成本只依官方 Standard list price 估算；free tier 與沒有 usage response 的失敗請求不得假裝成已知帳單金額。
- Interactions API 的影片視覺處理預設約 1 FPS；官方目前未在 Interactions API 開放 `video_metadata` 自訂 FPS。因此 0.2–0.5 秒 UI 狀態可能漏掉。本實驗以完整影片 Content Map 對照「抽出的原始單幀 Grounding」量測這個限制，不把未觀察到的狀態靜默補上。

官方依據：

- [Interactions API overview](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Video understanding / File API](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Image understanding / Gemini 原生 bbox 座標順序](https://ai.google.dev/gemini-api/docs/image-understanding)
- [google-genai Python SDK](https://googleapis.github.io/python-genai/)

## 環境

需求：Python 3.12、`uv`、FFmpeg/ffprobe，以及 Gemini API key。

```bash
cd ~/Experiments/JasCueVideoLab
UV_CACHE_DIR=.uv-cache uv sync --python 3.12
export GEMINI_API_KEY='...'
```

若執行環境不會繼承 terminal export，可在專案根目錄建立已被 `.gitignore` 排除的 `.env.local`，內容只放 `GEMINI_API_KEY=...`，執行前先 `source .env.local`。不要把 key 貼進 issue、artifact 或 commit。

只使用官方 `google-genai` SDK；預設模型是穩定版 `gemini-3.6-flash`，需要可重現歷史 A/B 時才以 `JASCUE_GEMINI_MODEL` 明確覆寫。3.6 請求不送出已淘汰的 `temperature`、`top_p` 或 `top_k`；純 geometry 目前維持 `low`，exact-frame semantic identity checkpoint 使用 `medium`，較複雜的少數規劃實驗才使用 `high`。不同 task profile 必須進入 request 與 cache fingerprint，且不能假設較高 thinking 一定改善 bbox。模型 ID 會進入 request、cache identity、provenance 與逐模型計價；切換模型不會誤用舊 response cache，但仍可重用相同 File API 上傳。程式不依賴已淘汰的 `google-generativeai`，也沒有舊 Gemini 1.5／2.0 model ID。

## 本機 Blind Review Web App

不想透過 Codex 代為判讀時，可直接啟動 human-first 審核介面：

```bash
cd ~/Experiments/JasCueVideoLab
set -a; source .env; set +a
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab serve-review
```

瀏覽器開啟 `http://127.0.0.1:8765`，直接拖入影片。預設只綁定本機 loopback，沒有登入系統；除非完全理解風險，請勿使用 `--allow-network` 對區網開放。

App 的固定順序是：

1. 影片串流寫入本機，ffprobe 與 SHA-256 驗證後建立 session。
2. 可選擇建立 1080p／30fps analysis proxy；Gemini 語意分析使用 proxy，bbox 仍從原始影片抽幀。
3. 沒有 target 時只顯示候選卡，且不預選、不顯示候選 confidence。
4. 使用者選擇候選或自行輸入精確 target 後，先檢查完整 Identity／Predicate／Framing Proposal 與 hash，再明確核准 QueryLock，才允許產生 target-locked `MM:SS` coarse candidates。
5. Identity-only QueryLock 可選一個時刻，由 FFmpeg 保存原始 frame PTS，再執行單幀 Grounding。若 QueryLock 含 predicate，Web 只顯示 coarse candidates 並鎖住 Grounding；必須先匯出、執行 `refine-query-predicate` 取得正式 DF frame-ID evidence，不能把 `MM:SS` 冒充為 predicate 已驗證。
6. Blind review 只顯示 `Candidate A/B` 框；提交「正確／錯物件／太大／太小／不可見／無法判斷」前，reveal API 會拒絕提供模型 label、confidence 與理由。
7. 可在畫面拖曳人工修正框；人工判定寫入後才能揭露完整 Gemini proposal。
8. `匯出完整 JSON` 包含 media identity、human annotations、已揭露 proposals 與尚未審核清單。

每個時刻提供兩個隔離模式：A 是預設且可驗證的「FFmpeg exact frame → image Grounding」；B 是實驗性的「完整影片 → 指定 `MM:SS` → bbox」。Google 官方明確文件化的是 image object detection bbox，而 File API 影片預設以 1 FPS 保存／處理；官方沒有提供 B 模式實際採用 frame 的 PTS 或 hash。因此 B 的 contract 永遠標記 `unknown_gemini_video_sample`，投影到 FFmpeg frame 的圖只供 A/B 診斷，不能成為 production geometry。兩種方法都經獨立盲審後，export 才會計算第一候選 bbox IoU 與 center distance。

一個經授權的真實短片測例曾讓 A、B 兩種模式選中相同指定實例，但兩個 bbox 仍有明顯幾何差異。公開文字不揭露原始檔名與私人路徑；媒體本身仍可能含人物、品牌與活動場景，不能稱為已去識別化。這是模型輔助視覺檢查，不是獨立 human ground truth；B 模式的 reference frame 仍不可知，因此不能用來建立正式 tracking seed。

持久資料位於被 Git 排除的 `artifacts/blind-review-app/<session-id>/`；跨 session 的 Gemini File API cache 依 analysis source SHA-256 位於 `artifacts/blind-review-file-cache/`。同一 upload identity 在官方 48 小時保存期內會重用。App 不會把 API key 傳到瀏覽器，也不以 browser storage 當實驗資料來源。

### 從實驗工具變成可丟素材的 App

目前核心流程已可包成簡單 App，不需要把分析、選片、Grounding、SAM 與 renderer 重寫成另一套。產品介面可以固定為：

```text
建立專案並拖入一批素材／brief／可選音樂
  → 背景工作建立 proxy、Clip Cards、MusicMap 與 Top-K
  → 使用者審核選片、Identity／Predicate／Framing Lock 和 trim
  → 只對入選且需要重構的片段執行 Gemini bbox＋SAM
  → 顯示 16:9／9:16、節奏、虛擬鏡頭與失敗原因
  → 使用者核准後 deterministic render
```

真正還需要產品化的是 upload／job queue、進度與取消、artifact DAG／stale propagation、API 配額與成本上限、File API cache 管理、登入與媒體權限、人工 review UI，以及失敗後從哪個 stage 重跑。這些是服務與介面工程，不是要模型一次直接吐出不可稽核的完整時間軸。自動化程度可以很高，但 target identity、重要文字、主觀 hero take、受控裁切、trim 與最後節奏仍應保留 human-in-the-loop 核准。

## 產生四種真實影片 fixture

Fixture 是由 Pillow 畫面經 FFmpeg 編碼而成的真實 MP4，不是 API mock。A 是 30 秒無旁白手機 UI 操作；B 是人物與手機同時移動的 16:9 畫面；C 含 0.3 秒快速按鈕狀態；D 含硬切鏡與兩支相似手機。

```bash
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab make-fixtures
```

每支影片都會生成 `.media.json`，包含 duration、coded/display dimensions、rotation、frame rate、time base、stream/format metadata 與 SHA-256。參考標註在 `fixtures/annotations/`；每份標註必須聲明 author type、方法與是否經獨立真人確認。`real_continuity_public.json` 目前由 Codex 視覺檢查後手動輸入，**不是 independent human ground truth**。

## 實跑垂直切片（三次）

```bash
UV_CACHE_DIR=.uv-cache uv run jascue-video-lab run \
  fixtures/generated/A_silent_phone_ui.mp4 \
  --runs 3 \
  --ground-per-event 1 \
  --annotations fixtures/annotations/A_silent_phone_ui.json
```

若要依建議測試 B 對同一關鍵幀分別框手機、手機螢幕、手、臉與另一支手機，可把 `--ground-per-event` 提高；實際數量仍取決於 Content Map 中該事件建立的 Entity。

其他命令：

```bash
# 只做媒體探測
uv run jascue-video-lab probe VIDEO.mp4 --output media.json

# 抽取 >= 2.8 秒的第一張原始影格，旁邊會寫 frame.png.json
uv run jascue-video-lab extract VIDEO.mp4 2800 frame.png

# 從既有結果重建 timeline
uv run jascue-video-lab timeline ARTIFACT/run-01 ARTIFACT/source.mp4

# 比較任意多次執行
uv run jascue-video-lab compare ARTIFACT/run-01 ARTIFACT/run-02 ARTIFACT/run-03 \
  --output ARTIFACT/comparison.json --annotations fixtures/annotations/A_silent_phone_ui.json

# 對同一張已抽出的原始影格重跑 Grounding
uv run jascue-video-lab ground-repeat ARTIFACT FRAME.png.json \
  --event-id EVENT --event-description DESCRIPTION --entity-id ENTITY \
  --target-description TARGET --runs 5 --output-dir OUTPUT

# 不讓 Gemini 產生時間數字：FFmpeg 建立 PTS 網格，Gemini 只選 frame ID
uv run jascue-video-lab storyboard-temporal ARTIFACT \
  --interval-ms 4000 --output-dir ARTIFACT/storyboard-pts-grid-4s-live

# 讓 Gemini 推薦少量官方 MM:SS 截圖時刻，本機驗證、抽幀並 Grounding
# 若未提供 target，這個命令只會提出候選並停止，不會自行挑物件 Grounding
uv run jascue-video-lab direct-moment-repeat ARTIFACT \
  --runs 3 --ground-runs 1 --output-dir ARTIFACT/direct-mmss-3runs-live

# 明確的候選階段；沒有 bbox，也不做 tracking
uv run jascue-video-lab suggest-targets ARTIFACT \
  --output-dir ARTIFACT/target-candidates

# 使用者選定候選後，鎖定 target 才找時間與 Grounding
uv run jascue-video-lab direct-moment-repeat ARTIFACT \
  --candidate-map ARTIFACT/target-candidates/run-01/target_candidates.json \
  --candidate-id selected-subject \
  --runs 3 --ground-runs 1 --output-dir ARTIFACT/selected-subject

# 僅限 experiment/dynamic-tracking branch；用 Gemini GroundingProposal 當 seed
uv sync --extra tracking
uv run jascue-video-lab track-csrt VIDEO.mp4 \
  --grounding-json ARTIFACT/events/EVENT/groundings/ENTITY/grounding.json \
  --target-description '審核者指定的前景實體；排除背景圖像與相似實例' \
  --output-dir ARTIFACT/tracking-selected-subject

# 僅限 experiment/sam21-video-segmentation branch；官方 checkpoint 不進 Git
SAM2_BUILD_CUDA=0 UV_CACHE_DIR=.uv-cache uv sync --extra segmentation
mkdir -p artifacts/models
curl -L -o artifacts/models/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Gemini bbox → SAM seed mask → 向前／向後 mask propagation
uv run jascue-video-lab track-sam21 VIDEO.mp4 \
  --checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --grounding-json ARTIFACT/grounding.json \
  --target-description '審核者指定的前景實體；排除背景圖像與相似實例' \
  --analysis-fps 2 --output-dir ARTIFACT/sam21-selected-subject

# 同一 shot 的多個 bbox seed 共用一個 SAM predictor／inference state
# targets.json 必須含兩個以上、target_id 唯一的 bbox-only targets
uv run jascue-video-lab track-shared-sam21 VIDEO.mp4 \
  --checkpoint artifacts/models/sam2.1_hiera_tiny.pt \
  --targets-json ARTIFACT/targets.json \
  --analysis-fps 15 --device cpu \
  --output-dir ARTIFACT/sam21-shared
```

`targets.json` 必須把每個 bbox 鎖到 upstream exact-frame Grounding 所聲明的 decoded source PTS，而不是只給可被重新四捨五入的毫秒：

```json
{
  "asset_id": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "targets": [
    {
      "target_id": "subject-a",
      "target_description": "審核者選定的第一個前景實體",
      "seed_source": "exact-frame Gemini bbox",
      "seed_time_ms": 5739,
      "seed_frame_pts": 344344,
      "seed_frame_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "seed_source_width": 3840,
      "seed_source_height": 2160,
      "seed_box_2d": [410, 370, 490, 570]
    },
    {
      "target_id": "subject-b",
      "target_description": "審核者選定的第二個前景實體",
      "seed_source": "exact-frame Gemini bbox",
      "seed_time_ms": 5739,
      "seed_frame_pts": 344344,
      "seed_frame_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "seed_source_width": 3840,
      "seed_source_height": 2160,
      "seed_box_2d": [520, 330, 575, 505]
    }
  ]
}
```

範例中的全零 hash 必須替換為實際值。原始 Grounding PNG 與 SAM 的 analysis JPEG 具有不同編碼，所以 hash 不同；dimensions 也可能因縮放而不同。追蹤器會把要求的 source PTS 強制加入固定間距抽樣序列，並以該 PTS 精確選 seed，不使用 nearest frame。此命令能驗證 PTS 與重解碼影格對齊；`seed_frame_sha256`、原始 dimensions 與 `seed_source` 則是呼叫端提供並保存的 upstream provenance，仍須由上游 Grounding bundle 驗證，不能由追蹤命令單獨證明 Gemini 看過該 PNG。

```bash
# 將共享 session 的 per-target tracks 合成一支正常播放速度的人工審核影片
uv run jascue-video-lab render-multi-sam21 \
  ARTIFACT/sam21-shared/targets/subject-a/segmentation-track.json \
  ARTIFACT/sam21-shared/targets/subject-b/segmentation-track.json \
  --label 'Selected subject A' \
  --label 'Selected subject B' \
  --analysis-frames-dir ARTIFACT/sam21-shared/analysis-frames \
  --display-fps 30 \
  --output-dir ARTIFACT/multi-track-review

# 比較兩條完全對齊的 SAM tracks；只量 agreement，不宣稱其中一條是真值
uv run jascue-video-lab compare-sam21-tracks \
  ARTIFACT/reference/segmentation-track.json \
  ARTIFACT/candidate/segmentation-track.json \
  --output ARTIFACT/sam-track-agreement.json

# 與 CSRT 比較只稱為 agreement；兩者都不是 human ground truth
uv run jascue-video-lab compare-trackers \
  ARTIFACT/sam21-selected-subject/segmentation-track.json \
  ARTIFACT/csrt/tracking.json --output ARTIFACT/tracker-agreement.json
```

`track-shared-sam21` 要求所有 seed 都落在同一個 FFmpeg shot，且不接受 Gemini polygon。它只共用重複的影片解碼、predictor 與 inference state；每個物件仍有自己的 object ID、bbox seed、mask 與 drift 狀態。每個 propagation output 會立即縮減成 binary mask 與小型統計並落盤，不會把 `影格數 × 物件數` 份 full-resolution float logits 全留在 RAM；缺格、重複格、越界格或缺物件則直接失敗，不會偽裝成 `lost`。`render-multi-sam21` 只會合併來自同一 asset、同一區間與完全相同 decoded-source PTS 取樣序列的軌跡；共享 session 必須顯式提供並驗證 immutable frame manifest，任一對齊資料不同就拒絕輸出。`--display-fps 30` 只控制審核影片播放時間，不表示 tracker 已在 30 FPS 上推論。產出為含來源音軌的 H.264/yuv420p MP4 與可追溯 manifest，只供人工審核，不是準確率證明或 production SpatialTrack。

## macOS SAM 2.1 runtime 狀態

目前可採用的 reference path 是 Meta 官方 SAM 2.1 PyTorch video predictor。它具備影片 memory、bbox prompt、多物件與雙向傳播；在 macOS 上可跑 CPU 或 MPS。實測顯示 MPS 能產生幾乎相同的 mask，但在目前的 Apple Silicon／PyTorch 組合上不一定更快，因此 `device=auto` 的結果不能取代目標機實測，正式預設仍以可重現 benchmark 決定。

Apple 發布在 Hugging Face 的 `coreml-sam2.1-tiny` 與 `coreml-sam2.1-large` 都是**單張圖片 segmentation** package。它們沒有可直接取代 SAM video predictor 的 temporal memory pipeline；Large 仍然只是較大的 image-only 模型，不能因名稱含 SAM 2.1 就當成影片 tracker。

EdgeTAM 的官方 PyTorch video predictor 同樣具備 temporal memory、bbox prompt、多物件與雙向 propagation。本機單一 fixture 的 MPS propagation 約為官方 SAM 2.1 Tiny MPS 的 2.87 倍速，但其中一個目標曾連續五格輸出空 mask，且目前依賴組合需要隔離的 tensor 相容修正；因此只列為需要人工 golden set 驗證的 experimental candidate，不取代 reference path。

EfficientTAM-Ti 也以相同的兩個 bbox、一個共享 state 與雙向 propagation 跑通；速度介於 EdgeTAM 與官方 SAM 2.1 Tiny CPU 之間，而且不需修改 upstream source。不過 MPS 無法使用其 CUDA-only 小孔洞後處理，所以同樣必須以人工 mask fixture 驗證邊緣品質，不能只看 throughput 或 peer IoU。

MLX 原生的完整 SAM 2.1 video predictor 也是值得繼續比較的 macOS 方向：它可保留 bbox、多物件與 memory propagation，且初步測速較 PyTorch 快。不過目前找到的社群實作仍很新，repository 的程式碼授權標示也尚未完整；因此只列為隔離的研究候選，不加入預設依賴。另一個僅支援 point、單向 propagation 或不能可靠共用多物件 state 的實作，不會用不等價條件加入 benchmark。詳細證據、MPS／CPU 實測、benchmark 限制與採用 gate 見 [MACOS-SAM21-EVALUATION.md](MACOS-SAM21-EVALUATION.md)。

## 產出

每次 `run` 會建立唯一 artifact 目錄：

```text
artifacts/<asset-sha-prefix>/<UTC timestamp>/
├── media.json
├── source.mp4 -> 原始影片
├── upload/
│   ├── file_upload_initial.json
│   └── file_upload_final.json
├── run-01/
│   ├── run.json
│   ├── content_map.request.json
│   ├── content_map.attempt-01.*              # 每次失敗／修正皆獨立保存
│   ├── content_map.attempt-02.*
│   ├── content_map.raw_interaction.json
│   ├── content_map.raw_output.json
│   ├── content_map.schema_validation.json
│   ├── content_map.json
│   ├── errors.json                         # 有錯才出現，不靜默吞掉
│   ├── index.html                          # 每個事件可點選播放
│   └── events/<event-id>/
│       ├── frame.png
│       ├── frame.json                      # requested time 與真實 PTS 分開
│       └── groundings/<entity-id>/
│           ├── grounding.request.json      # 不含 base64 圖片，只記 hash
│           ├── grounding.raw_interaction.json
│           ├── grounding.raw_output.json
│           ├── grounding.native.json         # Gemini 官方 y-first 座標
│           ├── grounding.coordinate_transform.json
│           ├── grounding.schema_validation.json
│           ├── grounding.json
│           └── debug.png
├── run-02/...
├── run-03/...
├── comparison.json
└── result.json
```

Gemini File API 物件依官方文件保存 48 小時。命令會先以已保存的 file name 查詢：仍為 `ACTIVE` 就重用；只有明確收到 `404/NOT_FOUND` 才重新上傳，其他不確定錯誤會保存並停止。`upload/file_cache.json` 記錄是否 reuse，舊 metadata 在重傳前移到 `upload/history/`；只有明確傳入 `upload --force-reupload` 才無條件重傳。參考：[Files API](https://ai.google.dev/gemini-api/docs/files)、[File input methods](https://ai.google.dev/gemini-api/docs/file-input-methods)。

API Structured Output 仍會由本機 Pydantic 再驗證。原始 Interaction response 與原始 `output_text` 都先保存；若 JSON 或語意 contract 失敗，錯誤、類型與 traceback 會寫入 `errors.json`，不會以假資料補值。請注意 request 使用 `store=false`，以本機 artifacts 作為實驗紀錄。

Gemini 官方 object detection 格式為 `[ymin, xmin, ymax, xmax]`，而本專案 canonical contract 依需求固定為 `[xmin, ymin, xmax, ymax]`。API boundary 因此使用明確命名的 `box_2d_yxyx`，通過 native schema 後再以純軸序重排轉成 `box_2d`；兩份 JSON 與 transform record 都保存。不得用框的長寬比例猜座標順序。

`comparison.json` 包含：Event 數量差、label 相似度、start/end 差、keyframe 差、第一候選 bbox center distance（0–1000 空間）、IoU、每份 schema validation 結果及 reviewer-reference 對照。歷史 JSON key `human_annotation_comparison` 目前為相容性保留，不代表已由真人標註。無候選或不可見 proposal 會保留為不可比較，不會捏造 bbox。

## 測試

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

Contract tests 驗證 schema、entity reference、半開事件區間、禁止 `frame_accurate`、不可見目標空候選與 bbox 範圍；geometry tests 驗證 normalized-to-pixel、center distance、IoU、mask-derived bbox、碎片 gate 與跨 shot latch；media test 會實際呼叫 FFmpeg 產生短片並確認推薦毫秒與真實 PTS 分離。這些測試不會 mock 或宣稱 Gemini/SAM live call 成功；live 成功只能由 artifacts 中的 raw response、validated JSON、mask 與 debug overlay 證明。

## Live 實驗結論

已使用多支經授權的真實素材完成重複 live run。公開文件中的素材描述已去識別化，但媒體本身仍可能含可辨識人臉、品牌與活動場景。原始素材、可辨識檔名、活動資訊與逐次報告只保存在 Git 排除的本機 artifacts；公開 README 只記錄可泛化的方法學結論：

- Content Map、Grounding Proposal 與 HTML timeline 的垂直資料流程可以完成，但 schema 合法不代表事件時間、物件身分或剪輯判斷正確。
- 初版曾把 Gemini 官方 y-first bbox 當成專案 x-first bbox，造成正確物件被畫成錯誤形狀。現行 API boundary 固定使用明確命名的 `box_2d_yxyx`，再由本機 deterministic conversion 轉成 canonical x-first；不以長寬比例做 heuristic auto-swap。
- 在目標層級、實例特徵及排除條件鎖定後，單幀 Grounding 的重跑可以相當穩定；這只能證明它適合作為待審核 bbox seed，不能取代獨立真人 ground truth 或成為 production tracking data。
- 完整 Content Map 曾在正確媒體 duration 下產生超界時間。縮小 schema 或以 contract-error feedback 修成合法數值，仍不能保證推薦影格具有正確語意。
- 少量顯著時刻可先採 Gemini `MM:SS`，經本機片長驗證後再由 FFmpeg 抽幀並保存真實 PTS；若時間非法、目標不可見或需要全片 coverage，改用帶 immutable frame ID 的本機 storyboard，讓模型只選既有 ID。
- 固定間距 storyboard 只提供 coarse coverage；快速 UI、短暫手勢與瞬時對焦狀態必須在候選事件及單一 shot 內使用更密的局部 frame-ID 網格，不能把固定抽樣結果當 frame-accurate cut point。
- 多個相似實例同框時，即使 bbox 重跑一致，也只能稱為模型穩定度。必須先由人工或 QueryLock 鎖定目標，且在不知道模型框的情況下建立 reference，才能計算有意義的準確率。
- OCR 實驗曾出現兩類錯誤：把模糊字元以既有知識補成相似名稱，以及在另一處小字捏造畫面沒有的規格。evidence-only system instruction 能降低但不能消除此問題；重要名稱、數字與 UI 狀態仍需 exact-frame 驗證與人工核准。

上述 IoU、重跑一致性與模型輔助抽查只能作為實驗診斷。完成獨立真人 blind review 前，不得宣稱人工 ground-truth 驗收通過。可攜式 HTML 的 build/schema 與瀏覽器載入檢查也只證明報告工具可用，不代表其中的模型判斷正確。

## 未來與 JasCue 的資料邊界

在人工審核與多次穩定度門檻通過後，可考慮轉成 JasCue **fixture** 的只有：

- 去識別化的測試影片及其 SHA-256/media metadata。
- 人工確認後的 coarse Content Map 事件、Entity 描述與不確定性案例。
- 人工確認的單幀 bbox 測試案例、schema contract 與 geometry 測試向量。
- 多次執行的比較報告，用於建立未來 regression threshold。

下列資料不得直接成為正式 SpatialTrack：

- Gemini 的 semantic timestamps 或推薦 keyframe。
- 未經人工確認的 bbox、不可見物件推測或相似物件選擇。
- `debug.png`、模型 confidence 或 label similarity。
- 單幀 bbox 串接、內插或任何假裝為逐幀 tracking 的衍生資料。
- 本實驗的 HTML timeline；它是審核工具，不是 NLE timeline。

任何移入 JasCue 的 fixture 都應經明確人工審核與獨立變更流程；本 repository 不提供也不執行合併回 JasCue 的命令。
