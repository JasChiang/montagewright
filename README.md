# montagewright

丟一個資料夾的毛片進去，出來一支剪好的片，還有一份說明每個決定是怎麼來的報表。

```bash
montagewright render RUSHES/ --brief BRIEF.md --music TRACK.mp3 \
  --aspect 9:16 --review --output CUT/

montagewright transcribe VIDEO.mp4        # 只要字幕，不重剪
montagewright-web                         # 同一件事，網頁上看得到過程
```

![網頁介面](docs/images/web.jpg)

## 做法

如果有人想研究 AI 剪輯但沒什麼頭緒，這裡是我把雲端跟地端接起來的做法。整套東西的判斷只有一句話：看得懂才答得出來的問題交給模型，要量才知道的問題交給程式。

畫面這邊，先把毛片壓成 proxy 讓 Gemini 整支連聲音看完，寫成一張 Clip Card。實際長這樣（省略了幾筆）：

```jsonc
{
  "summary": "Cameras pan horizontally back and forth showing standing foldable phones (white and purple) against a white background.",
  "composition": "horizontal",        // 內容橫向鋪開，會跟直式裁切打架
  "usable_from_seconds": 0.0,
  "usable_to_seconds": 27.2,          // 開頭在甩、結尾鏡頭移開的部分不算
  "speech": "ambient",                // 不是 content，所以不會去做逐字稿
  "camera_moves": true,
  "camera_motion": "左右平移展示白與紫色兩款對折手機機背鏡頭模組與鉸鏈細節",
  "subjects": [
    { "label": "the white foldable phone",
      "centre_x": 0.65, "centre_y": 0.52, "width": 0.45, "height": 0.9,
      "moves": true, "at_seconds": 2.0 },
    { "label": "the purple foldable phone", ... }
  ],
  "action": [
    { "what": "the camera pans right showing the purple phone",
      "starts_seconds": 2.5, "ends_seconds": 4.5 }, ...
  ],
  "needs": [
    { "what": "crop", "why": "畫面左側有大量留白，若要進行直式構圖或聚焦產品需重新裁切置中" }
  ]
}
```

有幾個欄位是踩過坑之後才加的。`camera_motion` 是因為素材自己在動的時候，再疊一層數位運鏡兩邊會打架，不如框住不動讓它演完。`speech` 決定要不要為這支付一次逐字稿的錢，判準是「這顆的意思靠不靠聲音成立」，不是「有沒有人在講話」。主體的 `label` 一定要能跟旁邊長得像的東西分開（「左邊那台白色的」而不是「那台手機」），`at_seconds` 則是記下這個框是看第幾秒說的 —— 東西在動的時候，位置只在那一刻成立。

卡片刻意不看 brief，所以只要素材沒變就一直有效，同一批素材想剪成別的主題不用再重看一遍。要重新構圖成 9:16 的話，再用 Gemini 的物件偵測抽幾張靜幀問「左邊那台深色手機」在哪，然後把那個框交給 SAM 2.1 往前後傳播，追出每一幀的位置。「哪一台是深色的」要看得懂畫面才答得出來，所以交給模型；「它在第幾格的哪個位置」則是量出來的，所以交給程式。

聲音那邊的分工其實一樣，只是換成時間跟文字。Apple SpeechTranscriber（macOS 26 內建）每個字都有自己的起訖時間碼，還附一個信心值，這種東西只有真的去量音訊才會知道；但它中文常常聽錯，會掉字，也會出現同音字。Gemini 剛好反過來，它聽得懂在講什麼，可是你問它某句話是第幾秒到第幾秒，它就給你一個看起來很合理的數字。時間戳麻煩的地方在於錯的跟對的長得一模一樣，沒辦法用看的檢查出來。

所以我的做法是讓 Gemini 獨立看一次影片先給一份逐字稿，再把正確的字回填到辨識器量到的時間上去。文字取自模型，時鐘完全來自辨識器，模型自己報的秒數一律不採用。

還有一條是決定跟執行要分開。執行層碰到做不到的事，要降級並且記錄下來，但不能放棄素材，也不能自己改掉模型的決定。有一次它自作主張把靜止鏡頭換成橫向掃描，結果下一輪審查回報「掃過去了，字還是被切」，整個迴圈在跟自己打架。所以報表跟影片一樣是交付物，片子不好看的時候，才知道是哪一層判斷錯了。

## 流程

十二個階段，其中八個要付錢給 Gemini。行首的 `$` 是付費呼叫、`-` 是本機執行，`◈` 表示有內容定址快取，`〔〕` 是跑這一階的條件。

```
-   毛片 → proxy  ◈
    640px；已經剪過的長片先照場景切點拆回鏡頭
    |
$   Clip Card  ◈
    每支素材一次，跨專案共用
    |
-   Apple ASR  ◈  〔有人講話才跑〕
    每字的起訖時間，整條線唯一的時鐘
    |
$   逐字修正  〔同上〕
    看影片改同音字；它自己報的秒數不採用
    |
$   定調
    片長、比例、哪些素材直接排除
    |
$   選片
    用哪幾顆、進出點、每顆什麼運鏡
    |
$   節奏  〔有配樂才跑〕
    聽音樂重寫長度，並記下哪些本來就不打算對拍
    |
$   主體定位  〔有運鏡才跑〕
    抽靜態格問框，回傳 0..1000 座標
    |
-   SAM 逐幀追蹤  〔同上〕
    本機 propagation，算出每一格的裁切路徑
    |
-   渲染
    ffmpeg 分段 → 串接 → 混音
    |
$   審查  〔--review〕
    看單顆 + 看整片；沒交付的那幾顆送回「主體定位」重跑
    |
-   交付
    mp4、字幕、report.json、FCPXML
```

有一點跟直覺不太一樣：定調跟選片並不是只讀卡片上的文字，它們會把所有 proxy 一起附上去重看一次，卡片在這裡只是索引。選片原本只讀文字摘要，改成附上影片之後，它從挑十一顆變成挑十七顆，而且講出了摘要裡根本沒有的東西。摘要畢竟是在還沒人知道這支片要講什麼的時候寫的。

畫面配音樂的片，切點會對上實際量到的重音；以講話為主的片則是一顆鏡頭一句話，講完就切，長度完全由內容決定。不用先告訴它是哪一種，卡片會自己判斷。

## 怎麼用

```bash
# 需要 Python 3.12 跟 ffmpeg
uv sync                                    # 或 pip install -e .
echo 'GEMINI_API_KEY=...' > .env && set -a && . ./.env && set +a

# 選配：逐幀跟拍要一份 SAM checkpoint（不放也能跑，只是不逐幀追）
#   https://github.com/facebookresearch/sam2 → sam2.1_hiera_tiny.pt

# 選配：逐字稿需要 macOS 26，編一次 Swift 工具
swiftc -parse-as-library -O -o tools/transcribe/transcribe \
  tools/transcribe/Transcribe.swift
```

最短的一次，素材夾進去、9:16 出來：

```bash
montagewright render ~/rushes --aspect 9:16 --output ~/cut
```

`--brief` 給的是意圖，不是分鏡表。寫「開頭三秒決定一切，每一顆都要有笑點或共鳴點」會比寫「第一顆用 A、第二顆用 B」有用得多，因為前者它會拿去當判斷依據，後者只是把你的工作抄一遍。

| 選項 | |
|---|---|
| `--review` | 逐顆對照它自己的計畫，沒做到的重新規劃再跑一次。最值得開，也最貴，大概多三到四成 |
| `--budget` | 總上限，包含重跑。碰到就停並交出當下最好的版本 |
| `--timeline` | `premiere` / `finalcut` / `both`，預設不產生。指回原始素材，所以每顆前後都還能往外拉 |
| `--subtitles` | 預設 `sidecar` 寫一份 SRT；`burn` 另外輸出燒好字的版本；`none` 不做 |
| `--subtitle-look` | `plain` / `speakers`（每人一色）/ `spoken`（講到哪亮到哪）/ `plate`（加底色塊） |
| `--speech` | 預設 `auto`，卡片說「聲音是內容」才做逐字稿，b-roll 不會被收費 |

實測花費（Gemini 3.6 Flash）：

| 這一輪 | 顆數 | 片長 | 花費 |
|---|---|---|---|
| 74 支毛片，第一次跑 | 10 | 34.9s | **$1.43** |
| 同一批素材再剪一支 | 22 | 71.0s | **$0.55** |
| 5 分半街訪，含逐字稿與重新規劃 | 13 | 48.6s | **$0.81** |
| 同一支再剪一次 | 16 | 46.1s | **$0.46** |

第一次最貴，之後就便宜很多，因為卡片跟逐字稿都是用檔案內容的 hash 快取的，素材沒變就整個階段跳過，proxy 也不用重新編碼。

上傳的檔案另外用 Gemini File API 存 48 小時，同樣以內容 hash 當 key，所以第二次規劃呼叫不用重傳（實測 76 秒對上第一次的 600 秒）。不過省下來的是傳輸時間而不是 token —— 定調跟選片每次都會把那些 proxy 重看一遍，input token 照算。

## 網頁介面

`montagewright-web` 開在 `127.0.0.1:8765`，不用記參數，而且整個過程看得到。跑完會得到一個時間軸，點一顆就看到它原本打算怎麼拍、驗收又看到什麼，可以改進出點、換順序、刪掉再重新輸出，這一步不用錢，因為決定都還在，只是重排而已。

比較實用的是可以檢查裁切跟追蹤是不是真的：切到「原素材＋裁切框」，它會用 proxy 播放並且把 9:16 的框畫在上面跟著時間軸走。字幕也可以在這裡改完再燒。

## 會產出什麼

- **`report.json`** —— 調性、每顆為什麼被選、為什麼是這個長度、每一筆降級跟造成它的量測數字、每個階段花多少錢
- **`work/crops.json`** —— 每顆實際用的裁切路徑，逐個 keyframe。這是「它真的有跟拍」跟「報表說它有跟拍」的差別
- **可編輯的時間軸** —— 指回原始素材檔案，每顆掛著 marker 寫明為什麼選它、哪裡降級、驗收看到什麼
- **字幕 SRT** —— 時間已對齊成片，並標出每句是誰講的

## 程式結構

```
src/montagewright/
  clipcard.py     每支素材的卡片
  transcript.py   逐字稿（本機辨識 + Gemini 修正）
  backfill.py     把校正過的字回填到辨識器的時間上
  planner.py      定調、選片、節奏、主體定位、重新規劃
  capabilities.py 執行層做得到跟做不到的事，同時餵給 prompt 跟 dispatcher
  reframe.py      裁切路徑
  executor.py     EDL → 渲染計畫
  renderer.py     ffmpeg
  review.py       逐顆驗收 + 成片審查
  timeline.py     FCPXML / Premiere XML
  subtitles.py    安全區、斷行、把字燒進畫面
  webapp.py       網頁介面
  measure/        量測：SAM 追蹤、音樂分析、場景偵測、幾何
  prompts/        所有 prompt，繁體中文
```

`measure/` 以外的都在做決定，`measure/` 裡的都在算數字。有一個測試守著這條線：任何 `measure/` 裡的程式 import 上層都會失敗。

撞出來的坑寫在 [`docs/lessons.md`](docs/lessons.md)，大部分屬於「東西沒壞但答案是錯的」那一類。舊版設計文件在 `docs/history/`。
