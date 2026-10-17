# High-dimensional update audit

審查日期：2026-10-13。只讀取 Git、程式及既有結果，沒有執行訓練、載入 checkpoint 執行推論或修改 model。本輪工作檔案僅為本 audit 與 `HIGH_DIMENSION_ANALYSIS_PLAN.md`。

## 1. Git 狀態與比較範圍（本機更新後重新核對）

- 原審查文件已按要求 commit：`ad230ef`，message `Add high-dimension update audit`。
- fetch 最新 `group/main` 後仍是 `e3f17c01246b8a345fc92bd2787257a8abd79b12`，沒有比前次 audit 更新的 remote commits。
- 為保留文件 commit 並讓 main 精確對齊 remote，原分支保存為 `codex/audit-saved-ad230ef`；本機 `main` 從 `group/main` 建立並追蹤它。
- 已確認 `git rev-parse HEAD group/main` 輸出完全相同的 `e3f17c01246b8a345fc92bd2787257a8abd79b12`。切換後 working tree clean。
- 本輪已直接讀取更新後本機 script、兩組 config/log/results，重新核對下面內容。相關新程式和結果現在已在本機。
- 比較基點仍為 `5d3fa48adcbd89a489b91ea2ddb4e2d6588f6411`，比較終點為以上 shared HEAD。
- 沒有 push、改 model 或 training。隨後建立／更新兩份 analysis documents，屬本輪預期的未提交文件；不改變 HEAD 相等這個事實。

實際檔名為 **`HighDimensionMatrix-PINN_Combination.py`**，不是問題中的 `HighDimensionMatrix_PINN_Combination.py`。

最近新增的 10 commits（新至舊）：

| Commit | 內容 |
| --- | --- |
| e3f17c0 | 新增 high-dimensional 結果文字說明 |
| e606262 | 上傳額外結果圖 |
| 4a97b96 | 將 matrix 腳本改成目前名稱 |
| ca65977 | 訓練包內 loader 增加 `.astype(float)` |
| 0214adf | 匯出完整訓練包 |
| c54ae4b | 增加繪圖及發佈圖片 |
| 4642a20 | 發佈 MIT/HUST 短測試結果 |
| f2d6e97 | Zhang evaluation 關閉 gradient tracking |
| 12ecc95 | 新增 Wu matrix + Wang + Zhang 整合腳本 |
| 0300ae3 | 上傳文獻 |

## 2. 隊友新增／修改了哪些 files？

基點到最新 tree 共 **456 個路徑**有變動（455 新增、1 修改；唯一修改是 `.gitignore`）。主要分組：

| 路徑／群組 | 數量 | 性質 |
| --- | ---: | --- |
| `.gitignore` | 1 | 忽略訓練包的 `运行输出/` |
| 根目錄文獻 PDF | 7 | Severson、TimesNet、Wang、BatLiNet、Time-MoE 等參考資料，不是 model implementations |
| `HighDimensionMatrix-PINN_Combination.py` | 1 | 全新的整合實驗程式，1170 行 |
| `results/wu_wang_zhang/` | 22 | 兩次實驗各 5 個 artifacts、README、11 張圖 |
| `Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/` | 11 | 10 張另外上傳的圖和 `Explanation` |
| `训练要用的/` | 414 | 387 個 CSV、11 個 pretrained weights，以及模型／工具／入口／依賴／說明副本 |

原本根目錄 `Model/Model.py`、`dataloader/dataloader.py`、原始 `data/` 及 original main scripts 在這段 diff 中**沒有被修改**。但新增程式自建模型及資料流程，所以「原檔沒改」不代表「新實驗架構沒變」。

訓練包內包含 `02_模型/Model/Model.py`、`03_读电池数据/dataloader/dataloader.py`、六個原訓練／比較／微調入口、`开始训练.py` 和 `说明.md`。loader 副本相較根目錄的差異是 `df.iloc[:,:-1] = f_df.astype(float)`。這個修改不是 matrix 腳本的 preprocessing；matrix 腳本有自己獨立的 `Battery`／`load_battery`。

附帶 runtime 問題：`开始训练.py` 的 `check_files()` 仍要求 loader 副本與根目錄逐 byte 一致；以上已知差異會被它判為 problem，然後在啟動訓練前 return 1。這是靜態碼審查結論，未執行或修正 launcher。

## 3. 新腳本實際做什麼？

主要呼叫路徑：

```text
list_batteries → load_battery → delete_3_sigma → Battery
  → split_batteries（按 condition 留出、再按 battery 分 validation）
  → PairTable
      → causal_window → dominant_periods → matrices_from_window
  → SolutionU + dynamical_f（Wang 改寫版）
  → ZhangSOH（BatLiNet-inspired SOH 改寫版）
  → 同一個 loop 各自訓練；依 Wang validation MSE 存共同 checkpoint
  → 測試、metrics.json、test_predictions.npz、圖片
```

兩個 predictor 並行比較，不是 Wang 的 output 再送入 Zhang；也不是兩者組成 ensemble。`--plot-from` 是獨立的既有結果繪圖路徑。本次未執行任何入口。

## 4. Dataset 如何變成 matrix？Transformation 前後 shape

每個 CSV 已是每 cycle 一列的 **16 個充電摘要特徵 + capacity**，不是 raw voltage/current waveform。

1. `load_battery` 選取這 17 欄，移除 inf/NaN，按全電池每欄 mean/std 移除三個標準差以外的列，包含 capacity。
2. `Battery.features` 保留 16 個**未 normalized**特徵；`Battery.xt` 另組 16 features + cycle index，逐電池 min-max 到 [-1,1]。SOH = capacity/1.1。
3. 對第 i 個保留列，`causal_window` 取截至 i 的最近 T=16 列；早期不足 16 列時在左邊重複第一列。
4. `dominant_periods` 沿 cycle 軸做 real FFT，取絕對振幅，跨 16 channels 平均；將 DC 振幅設零，取 k=2 個最大頻率 bin，令 period `p = T // frequency`。
5. 將**原本的特徵值**（不是 FFT 複數係數）在右側補零至長度 L 可被 p 整除，reshape 成 `[L/p, p, 16]`。
6. 換軸至 `[16, L/p, p]`，bilinear resize 到 `[16,8,8]`；兩個 period 疊成 `[2,16,8,8]`。

| 位置 | Default shape | 意義 |
| --- | --- | --- |
| 清洗後整顆電池 features | `[N,16]` | N 個保留 cycles |
| 原 PINN 單 cycle／batch | `[17]`／`[B,17]` | 16 特徵 + cycle |
| 單 cycle 歷史 window | `[16,16]` | time × feature |
| 單 period reshape | `[16,ceil(16/p),p]` | channel × period repetitions × position |
| 單 cycle 最終 matrix | `[2,16,8,8]` | period rank × channel × resized row × resized column |
| matrix batch | `[B,2,16,8,8]` | B 是當次 minibatch 大小，不固定等於 64 |
| Wang flatten | `[B,2048]` | 2×16×8×8 |
| Wang concat 原向量後 | **`[B,2065]`** | 2048+17 |
| Zhang CNN input | **`[B,32,8,8]`** | 合併 k×channel=32 |
| 兩模型 SOH output | `[B,1]` | 一個 cycle 的 SOH |

高維不等於新增了 2048 個獨立物理量。它加入歷史資料，將同一批數值按選出的週期重排、插值，部分資料重複使用。

TimesNet 的官方實作確實用 FFT 選週期、padding 和 2D reshape，但還有 learned embedding、Inception blocks、period weighting 及 residual 等。這個腳本只借用重排思路，並額外 resize；不是完整 TimesNet。官方還在 batch 層面平均選頻，而本腳本逐 window 選頻。[官方 TimesNet 程式](https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/TimesNet.py)

## 5. 每個 dimension／feature 代表什麼？

最外面的 k 軸是該 window 中由振幅大小排序的兩個候選 period，不是固定兩種物理機制。同一 k 在不同 cycle／battery 可能代表不同 period；整數除法亦可能讓不同頻率得到相同 period。

原 reshape 的 row 是第幾次週期重複，column 是週期內的位置。resize 到 8×8 後變成插值座標，不能將某個 pixel 直接解讀成某個固定 cycle 或固定秒數。column 的時間單位是保留列，不是充電內 waveform sampling time。

16 channels 與原特徵順序一致：

| index | feature | 物理／統計含義 |
| ---: | --- | --- |
| 0 | voltage mean | 所選 CC 電壓段平均電壓 |
| 1 | voltage std | 電壓標準差 |
| 2 | voltage kurtosis | 電壓值分布峰度 |
| 3 | voltage skewness | 電壓值分布偏度 |
| 4 | CC Q | CC 選定段累積充電量 |
| 5 | CC charge time | 該段時間 |
| 6 | voltage slope | 該段電壓變化斜率 |
| 7 | voltage entropy | 電壓曲線統計 entropy，不是熱力學 entropy |
| 8 | current mean | CV 選定段平均電流 |
| 9 | current std | 電流標準差 |
| 10 | current kurtosis | 電流分布峰度 |
| 11 | current skewness | 電流分布偏度 |
| 12 | CV Q | CV 選定段累積充電量 |
| 13 | CV charge time | 該段時間 |
| 14 | current slope | 電流衰減斜率 |
| 15 | current entropy | 電流曲線統計 entropy |

Cycle index 是另一路 `xt` 的第 16 欄，**不是第 17 個 matrix channel**；capacity／SOH 亦不是 matrix channel。matrix 仍混合 V、A、Ah、時間及統計量等原尺度，時間等大數值 channel 可能主導平均 FFT 振幅。不能把最高振幅 period 當成已辨識的電池物理週期。

## 6. Wang 2024 與 Zhang 2025 對應哪個 model？

| 名稱 | Repo 對應 | 正確定位 |
| --- | --- | --- |
| Wang et al., 2024 原版 PINN4SOH | `Model/Model.py` 的 `PINN`、`Solution_u`、`MLP`；由 `main_MIT.py` 等載入 | 原 SOH baseline |
| 新腳本的 Wang | `HighDimensionMatrix-PINN_Combination.py` 的 `SolutionU`、`wang_forward`、`dynamical_f` | 接收 matrix 的改寫版，不是直接 import 原 `PINN` |
| 新腳本的 Zhang | 同檔 `MatrixEncoder`、`ZhangSOH`、`reference_batch`、`collect_predictions` | 借用 BatLiNet intra/inter-cell 思路的簡化 SOH 模型 |

Zhang 論文為 **Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning**，原任務是早期資料預測 lifetime，不是本腳本每 cycle 的 SOH。原文以 normalized capacity 對齊 electrical signals，建立 per-cycle feature maps；本腳本使用 Wang 的 CSV 摘要再做 period matrices，input 定義也不同。不能稱它為「原 BatLiNet 不變、只加 matrix」。[Zhang 等，2025，原論文](https://www.nature.com/articles/s42256-024-00972-x)

沒有在這次新增檔案中找到另一个完整、原始 BatLiNet implementation 或 original Zhang prediction baseline；PDF 不等同於 implementation。

## 7. Input、output、architecture 比較：有沒有同時改 model？

**有。**

| 項目 | 原 Wang PINN | 新 Wang-matrix | 新 Zhang-inspired SOH |
| --- | --- | --- | --- |
| input | `[B,17]` | `[B,2065]` | intra matrix 及 inter-cell matrix difference，CNN `[B,32,8,8]` |
| 主網絡 | encoder 17→60→60→32，predictor 32→32→1 | encoder **2065→60→60→32**，相同 predictor 形狀 | 兩個獨立 encoder，各 Conv2d(32→32,3×3,pad1)→ReLU→global average pool；共用 Linear(32→1) head |
| dynamics | 35→60→60→1 | 維持 35→60→60→1 | 沒有 PDE/dynamics network |
| output | SOH | SOH | intra SOH、delta SOH；推論組成最終 SOH |
| loss | data + PDE + measured-trend penalty | 代數形式沿用 current-code 三項 | intra MSE + lambda_delta×delta MSE |

Wang 第一層多了 2048×60 = **122,880 個 weights**。這是容量改變，並非單純 reshape。新腳本也將 MLP biases 全部初始化為零；原 dynamics MLP 的 bias 沒有相同的顯式歸零。初始化流程和 RNG consumption 不能視為完全一致。

`wang_forward` 將 matrix detach，僅對另一路 16 features 和 cycle 求偏導。因此 PDE 只限制「固定 matrix 時，向量 input 改變」的偏導，不包含歷史 matrix 隨 cycle／features 改變的完整影響。loss 公式看似相同，物理約束作用的 function 已改變。

Zhang 的運算為：

```text
D_i = matrix(i) - matrix(first retained cycle)
intra = shared_head(h_theta(D_i))
delta = shared_head(h_phi(D_i - D_reference))
inter = delta + reference SOH
prediction = 0.5 × intra + 0.5 × inter
```

每個 sample 隨機抽 training reference，預設做 2 次 prediction 再平均（可能重複抽到同一顆，不保證兩顆不同）。reference cycle 對齊的是清洗後 row index，超過 reference 長度時 clamp 到最後一列，不是按 SOH 或真實 cycle number 對齊。這個固定 0.5 mixture 不是我們之前規劃的 learned PIMOE router。

不同 cycle 各自選 period、resize，再相減；`D_i` 的相同 pixel 不保證對應相同 period/phase。第一列的常數 window 會有 FFT tie，DC 設零也不能保證 tie 時 DC 永不入選，之後 frequency 被 clamp 到至少 1。這些都是特徵語義的限制，不是已驗證的物理對齊。

## 8. Original 和 high-dimensional experiments 是否公平對照？

以下 original 指已有 current-code MIT Run 01；不是將不同資料集的 paper metrics 混在一起。

| 檢查項目 | 原 MIT Run 01 | 已存 matrix MIT 短測試 | 相同？ |
| --- | --- | --- | --- |
| split | ID %5 留 test，其餘 pair-level 80/20 validation | 整個 2018-04-12 condition 留 test；其餘按 battery 80/20，再 cap | **否** |
| batteries | 102 train/valid +23 test | 3 train、1 valid、1 test | **否** |
| target 定義 | capacity/1.1 | capacity/1.1 | **定義相同**，樣本不相同 |
| input normalization | 全軌跡 per-battery [-1,1] | xt 類似；matrix 未 normalized | **不完全相同** |
| cycle index | 清洗前插入，保留被刪行留下的 gap | 清洗後重新 0…N−1 | **否** |
| constant feature | 原 loader 直接除 max−min | 新 minmax 將零 span 換成 1 | **否** |
| pairing／sampling | 每相鄰 pair，stride 1 | 取 i=0,5,10…，各配 i+1；stride5 | **否** |
| epoch／batch | 上限200，實跑86；batch512 | 2 epochs；batch64 | **否** |
| global seed | 未固定；split420 | Python/NumPy/torch420；NumPy Generator shuffle/reference | **否** |
| Wang loss weights | alpha0.5 beta0.01 | alpha0.5 beta0.01 | 值相同，但 summed loss3 在不同 batch size 下相對影響不同 |
| scheduler | 原 linspace/cosine，epoch 後 step | 新 `solution_lr` epoch 前計算；短測試固定0.002 | **否**；即使跑200 epochs，warmup 分母及 timing 仍不同 |
| checkpoint | 依自身 validation MSE | Wang validation MSE 同時決定 Wang、Zhang checkpoint | Zhang **不是依自己最佳 validation** |
| environment | Python3.12.14、torch2.14.0+cpu 等 | 記錄 Python3.12.3、torch2.14.0+cu130，CPU 執行；NumPy2.4.4、pandas3.0.6 | **否** |

原 loader 在清洗時另含清洗前的 cycle column；新 loader 清洗的選欄不含 cycle。一般均勻 index 可能不造成额外刪除，但不能說兩者 preprocessing 逐項完全一樣。

同一個新腳本內，Wang/Zhang 確實共用 train/valid/test batteries、pair labels 及 matrix construction；但架構、loss、reference access 及 checkpoint-selection 公平性不同。**沒有同 split／settings 下的「無 matrix vs 有 matrix」ablation**，因此不能把任何 accuracy 差異歸因於 matrix。

## 9. 已存 batteries、shapes 與 prediction results

原結果仍存在：`results/reproduction/MIT/run_01/{true_label.npy,pred_label.npy,metrics.json}`；historical baseline 另存 `historical_run_01/`；作者 archive 亦保留。沒有重跑或改寫。

新結果：

- `results/wu_wang_zhang/MIT_2018-04-12_20260926_142820/`
- `results/wu_wang_zhang/HUST_10_20260926_142824/`

各有 `config.json`、`log.txt`、`metrics.json`、`test_predictions.npz`、`best_wang_valid.pt`。NPZ 已以 `allow_pickle=False` 從 Git blob 讀取 shape；checkpoint 沒有反序列化。

| Run | train | valid | test | pair 數／prediction shape |
| --- | --- | --- | --- | --- |
| MIT | 2017-05-12_battery-10、2017-06-30_battery-1、2017-05-12_battery-11 | 2017-05-12_battery-1 | 2018-04-12_battery-1 | train410、valid332、test182；三個 NPZ arrays 各 `[182,1]` |
| HUST | 1-1、2-2、3-1 | 1-2 | 10-1 | train1199、valid535、test336；三個 NPZ arrays 各 `[336,1]` |

NPZ keys 為 `y_true`、`wang_pred`、`zhang_pred`。兩次皆 2 epochs、window16、k2、8×8、stride5、batch64、cap3/1/1；best epoch=1 是 **zero-based 第二個 epoch**。不是 full reproduction。

| 已存結果 | MAE | MAPE % | RMSE | MSE |
| --- | ---: | ---: | ---: | ---: |
| MIT Wang-matrix | 0.584366 | 61.4374 | 0.645352 | 0.4164798 |
| MIT Zhang-inspired | 5.026066 | 531.9764 | 6.109314 | 37.32371 |
| HUST Wang-matrix | 0.189353 | 19.1884 | 0.241890 | 0.05851075 |
| HUST Zhang-inspired | 1.822809 | 189.0067 | 2.362972 | 5.583635 |

SOH 是 fraction；MAPE 是百分比，以 true SOH 作分母。這些誤差很大，但短測試只能證明程式可跑，不能證明模型不行或 matrix 無效。README 將 Zhang 大誤差歸因於原始尺度與兩個 epochs；這是合理假設，沒有 controlled experiment 證明因果。

新增 `...result after high dimension matrix/Explanation` 宣稱可看出 matrix 對 prediction quality 的 contribution；**目前 artifacts 不支持此因果結論**。該上傳圖目錄没有配套新 config／numeric prediction arrays，不能自動當成另一組可追溯完整實驗。`results/wu_wang_zhang/figures/` 才有相應 numeric artifacts 可追溯。

結果 config 記錄 commit `12ecc95`，但之後還有 evaluation fix、plotting 及 rename；沒有完整 dirty diff，不能斷言 stored run 與最新腳本完全相同。shape／metrics 可核查，精確執行程式 provenance 不完整。

## 10. Target leakage／future-test information 審查

| 部位 | 判斷 | 原因 |
| --- | --- | --- |
| window 本身 | **未發現直接讀取後續 feature rows** | `features[start:cycle+1]`；左補第一列，右補零 |
| FFT／reshape 的直接 input | **未直接包含 capacity／SOH channel** | 只用 FEATURE_COLUMNS 16 欄 |
| 全軌跡三 sigma | **確認有 target-dependent selection 與 future/test-trajectory 使用** | capacity 參與 outlier 刪除；各顆包括 test 的全軌跡 mean/std 影響目前保留 rows、matrix 及評估樣本 |
| xt min-max | **確認有 future/test-trajectory information** | 每顆（包含 test）用全生命 max/min；normalized cycle 更包含完整保留軌跡長度 |
| 清洗後 causal window | **不能稱嚴格 online causal** | 過去哪幾行保留，本身已由未來／target-informed cleaning 決定 |
| train／validation／test batteries | **程式上分離** | split battery objects 後建 PairTable；cap 不把同一顆放兩邊 |
| Zhang reference labels | **不是 test-target 直接洩漏**，但需聲明 reference-label availability | validation/test 的 pool 固定 train_b；加的是已知 training reference SOH，不是被預測電池的真值 |
| Zhang reference 時間 | **部署假設待確認** | 可以查完整 training reference trajectory 對應 row；只有當 reference batteries 是已完成歷史資料時才有合理可用性 |
| checkpoint selection | **未發現依 test metric 選 checkpoint** | 用 Wang validation MSE；test 在 restore 後執行 |

結論：**沒有發現直接把 test SOH 數值當 model feature，但不能說沒有 leakage。** target-informed 清洗與整條 test trajectory normalization 已足以否定「完全不看 future/test information」的說法。這部分原 baseline 亦有類似限制；新 causal window 並沒有消除它。

## 11. 接下來應如何理解／報告結果（本次不實作）

目前可說「成功建立 period-matrix preprocessing，並接到兩個改寫的 predictor 做 smoke test」。不能說「驗證 high-dimensional matrix 提升 Wang／Zhang paper accuracy」。

若日後要測 matrix 的貢獻，需要先固定同一 battery split、相同有效 rows／labels、normalization、training budget、seeds 與 checkpoint rule，再做有／無歷史 matrix 的對照；另控制模型容量差異。Zhang 應依自己的 validation criterion 存最佳 checkpoint。若主張 online inference，另訂不用 test labels 清洗、也不用完整 test trajectory 的 causal preprocessing protocol，並重新建立可比的 baseline。以上只是問題清單，沒有啟動訓練或修改任何流程。

## 12. 最簡單的總結

**以前：** 每個 cycle 的 16 個充電特徵，加 cycle index，共 17 個數字，交給 Wang PINN 預測 SOH。

**現在：** 隊友新增一個程式，把最近 16 個保留 cycles 的特徵整理成兩組 16-channel 的 8×8 矩陣，再分別送進加寬的 Wang PINN 和簡化 Zhang-style 模型；也換了 split，並存下 MIT/HUST 兩個短測試。

**Matrix 到底做了什麼：** 它把「最近一段歷史的同一批特徵」，按 FFT 找出的候選週期折成小格，再縮放成固定尺寸。它沒有憑空產生新的物理量；目前也還沒有公平實驗證明它提升了預測。

## Appendix A: Source evidence / remote line references

All line numbers refer to commit `e3f17c01246b8a345fc92bd2787257a8abd79b12`, now also local HEAD.

- [Cleaning / min-max / Battery, line 141](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L141)
- [FFT periods / causal window / reshape, line 230](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L230)
- [PairTable, line 289](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L289)
- [Battery split, line 350](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L350)
- [Wang SolutionU, line 423](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L423)
- [Zhang encoders and shared head, line 439](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L439)
- [Wang derivatives, line 478](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L478)
- [Scheduler, line 493](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L493)
- [Training reference selection, line 522](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L522)
- [Losses and training loop, line 563](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L563)
- [Prediction mixture, line 623](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L623)
- [Defaults, line 916](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L916)
- [Checkpoint selection and saved outputs, line 1060](https://github.com/GeorgeHeYiJie/CapstoneD07-LeungHoiTung-HeYiJie/blob/e3f17c01246b8a345fc92bd2787257a8abd79b12/HighDimensionMatrix-PINN_Combination.py#L1060)

Read-only Git blob checks confirmed all 387 bundled CSVs, 11 bundled pretrained weights, three Model files and two utils files match their root counterparts. Stored NPZ MAE/MSE were recalculated and agree with the stored metrics; no inference or training was performed.

## Appendix B: Complete changed-path inventory

Generated from `git diff --name-status -z 5d3fa48 group/main`; A = added, M = modified. These are net tree changes, so intermediate script renames are represented by the final path.

```text
M	.gitignore
A	2019 - Data-driven prediction of battery cycle life before capacity degradation.pdf
A	2023 - TimesNet Temporal 2D-Variation Modeling for General Time Series Analysis.pdf
A	2024 - Generative learning assisted state-of-health estimation for sustainable battery recycling with rando.pdf
A	2024 - Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis.pdf
A	2025 - Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning.pdf
A	2025 - TIME-MOE BILLION-SCALE TIME SERIES FOUN-.pdf
A	HighDimensionMatrix-PINN_Combination.py
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/Explanation
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/HUST_10_abs_error.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/HUST_10_metrics.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/HUST_10_soh_curve.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/HUST_10_training.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/HUST_10_true_vs_pred.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/MIT_2018-04-12_abs_error.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/MIT_2018-04-12_metrics.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/MIT_2018-04-12_soh_curve.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/MIT_2018-04-12_training.png
A	Wang et al., 2024 & Zhang et al., 2025 result after high dimension matrix/MIT_2018-04-12_true_vs_pred.png
A	prediction of second-life battery degradation trajectory using interpretable mixture of experts.pdf
A	results/wu_wang_zhang/HUST_10_20260926_142824/best_wang_valid.pt
A	results/wu_wang_zhang/HUST_10_20260926_142824/config.json
A	results/wu_wang_zhang/HUST_10_20260926_142824/log.txt
A	results/wu_wang_zhang/HUST_10_20260926_142824/metrics.json
A	results/wu_wang_zhang/HUST_10_20260926_142824/test_predictions.npz
A	results/wu_wang_zhang/MIT_2018-04-12_20260926_142820/best_wang_valid.pt
A	results/wu_wang_zhang/MIT_2018-04-12_20260926_142820/config.json
A	results/wu_wang_zhang/MIT_2018-04-12_20260926_142820/log.txt
A	results/wu_wang_zhang/MIT_2018-04-12_20260926_142820/metrics.json
A	results/wu_wang_zhang/MIT_2018-04-12_20260926_142820/test_predictions.npz
A	results/wu_wang_zhang/README.md
A	results/wu_wang_zhang/figures/HUST_10_abs_error.png
A	results/wu_wang_zhang/figures/HUST_10_metrics.png
A	results/wu_wang_zhang/figures/HUST_10_soh_curve.png
A	results/wu_wang_zhang/figures/HUST_10_training.png
A	results/wu_wang_zhang/figures/HUST_10_true_vs_pred.png
A	results/wu_wang_zhang/figures/MIT_2018-04-12_abs_error.png
A	results/wu_wang_zhang/figures/MIT_2018-04-12_metrics.png
A	results/wu_wang_zhang/figures/MIT_2018-04-12_soh_curve.png
A	results/wu_wang_zhang/figures/MIT_2018-04-12_training.png
A	results/wu_wang_zhang/figures/MIT_2018-04-12_true_vs_pred.png
A	results/wu_wang_zhang/figures/summary_mae_rmse_mape.png
A	训练要用的/01_环境依赖/requirements-reproduction.txt
A	训练要用的/02_模型/Model/Compare_Models.py
A	训练要用的/02_模型/Model/Model.py
A	训练要用的/02_模型/Model/__init__.py
A	训练要用的/03_读电池数据/dataloader/__init__.py
A	训练要用的/03_读电池数据/dataloader/dataloader.py
A	训练要用的/04_训练工具/utils/__init__.py
A	训练要用的/04_训练工具/utils/util.py
A	训练要用的/05_训练和运行脚本/四个数据集从零训练/main_HUST.py
A	训练要用的/05_训练和运行脚本/四个数据集从零训练/main_MIT.py
A	训练要用的/05_训练和运行脚本/四个数据集从零训练/main_TJU.py
A	训练要用的/05_训练和运行脚本/四个数据集从零训练/main_XJTU.py
A	训练要用的/05_训练和运行脚本/对比基线训练/main_comparision.py
A	训练要用的/05_训练和运行脚本/用已有权重微调/main_adaptation - fine-tuning.py
A	训练要用的/06_四个数据集/data/HUST data/1-1.csv
A	训练要用的/06_四个数据集/data/HUST data/1-2.csv
A	训练要用的/06_四个数据集/data/HUST data/1-3.csv
A	训练要用的/06_四个数据集/data/HUST data/1-4.csv
A	训练要用的/06_四个数据集/data/HUST data/1-5.csv
A	训练要用的/06_四个数据集/data/HUST data/1-6.csv
A	训练要用的/06_四个数据集/data/HUST data/1-7.csv
A	训练要用的/06_四个数据集/data/HUST data/1-8.csv
A	训练要用的/06_四个数据集/data/HUST data/10-1.csv
A	训练要用的/06_四个数据集/data/HUST data/10-2.csv
A	训练要用的/06_四个数据集/data/HUST data/10-3.csv
A	训练要用的/06_四个数据集/data/HUST data/10-4.csv
A	训练要用的/06_四个数据集/data/HUST data/10-5.csv
A	训练要用的/06_四个数据集/data/HUST data/10-6.csv
A	训练要用的/06_四个数据集/data/HUST data/10-7.csv
A	训练要用的/06_四个数据集/data/HUST data/10-8.csv
A	训练要用的/06_四个数据集/data/HUST data/2-2.csv
A	训练要用的/06_四个数据集/data/HUST data/2-3.csv
A	训练要用的/06_四个数据集/data/HUST data/2-4.csv
A	训练要用的/06_四个数据集/data/HUST data/2-5.csv
A	训练要用的/06_四个数据集/data/HUST data/2-6.csv
A	训练要用的/06_四个数据集/data/HUST data/2-7.csv
A	训练要用的/06_四个数据集/data/HUST data/2-8.csv
A	训练要用的/06_四个数据集/data/HUST data/3-1.csv
A	训练要用的/06_四个数据集/data/HUST data/3-2.csv
A	训练要用的/06_四个数据集/data/HUST data/3-3.csv
A	训练要用的/06_四个数据集/data/HUST data/3-4.csv
A	训练要用的/06_四个数据集/data/HUST data/3-5.csv
A	训练要用的/06_四个数据集/data/HUST data/3-6.csv
A	训练要用的/06_四个数据集/data/HUST data/3-7.csv
A	训练要用的/06_四个数据集/data/HUST data/3-8.csv
A	训练要用的/06_四个数据集/data/HUST data/4-1.csv
A	训练要用的/06_四个数据集/data/HUST data/4-2.csv
A	训练要用的/06_四个数据集/data/HUST data/4-3.csv
A	训练要用的/06_四个数据集/data/HUST data/4-4.csv
A	训练要用的/06_四个数据集/data/HUST data/4-5.csv
A	训练要用的/06_四个数据集/data/HUST data/4-6.csv
A	训练要用的/06_四个数据集/data/HUST data/4-7.csv
A	训练要用的/06_四个数据集/data/HUST data/4-8.csv
A	训练要用的/06_四个数据集/data/HUST data/5-1.csv
A	训练要用的/06_四个数据集/data/HUST data/5-2.csv
A	训练要用的/06_四个数据集/data/HUST data/5-3.csv
A	训练要用的/06_四个数据集/data/HUST data/5-4.csv
A	训练要用的/06_四个数据集/data/HUST data/5-5.csv
A	训练要用的/06_四个数据集/data/HUST data/5-6.csv
A	训练要用的/06_四个数据集/data/HUST data/5-7.csv
A	训练要用的/06_四个数据集/data/HUST data/6-1.csv
A	训练要用的/06_四个数据集/data/HUST data/6-2.csv
A	训练要用的/06_四个数据集/data/HUST data/6-3.csv
A	训练要用的/06_四个数据集/data/HUST data/6-4.csv
A	训练要用的/06_四个数据集/data/HUST data/6-5.csv
A	训练要用的/06_四个数据集/data/HUST data/6-6.csv
A	训练要用的/06_四个数据集/data/HUST data/6-8.csv
A	训练要用的/06_四个数据集/data/HUST data/7-1.csv
A	训练要用的/06_四个数据集/data/HUST data/7-2.csv
A	训练要用的/06_四个数据集/data/HUST data/7-3.csv
A	训练要用的/06_四个数据集/data/HUST data/7-4.csv
A	训练要用的/06_四个数据集/data/HUST data/7-5.csv
A	训练要用的/06_四个数据集/data/HUST data/7-6.csv
A	训练要用的/06_四个数据集/data/HUST data/7-7.csv
A	训练要用的/06_四个数据集/data/HUST data/7-8.csv
A	训练要用的/06_四个数据集/data/HUST data/8-1.csv
A	训练要用的/06_四个数据集/data/HUST data/8-2.csv
A	训练要用的/06_四个数据集/data/HUST data/8-3.csv
A	训练要用的/06_四个数据集/data/HUST data/8-4.csv
A	训练要用的/06_四个数据集/data/HUST data/8-5.csv
A	训练要用的/06_四个数据集/data/HUST data/8-6.csv
A	训练要用的/06_四个数据集/data/HUST data/8-7.csv
A	训练要用的/06_四个数据集/data/HUST data/8-8.csv
A	训练要用的/06_四个数据集/data/HUST data/9-1.csv
A	训练要用的/06_四个数据集/data/HUST data/9-2.csv
A	训练要用的/06_四个数据集/data/HUST data/9-3.csv
A	训练要用的/06_四个数据集/data/HUST data/9-4.csv
A	训练要用的/06_四个数据集/data/HUST data/9-5.csv
A	训练要用的/06_四个数据集/data/HUST data/9-6.csv
A	训练要用的/06_四个数据集/data/HUST data/9-7.csv
A	训练要用的/06_四个数据集/data/HUST data/9-8.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-1.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-10.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-11.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-12.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-13.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-14.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-15.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-16.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-17.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-18.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-19.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-2.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-20.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-21.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-22.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-23.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-24.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-25.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-26.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-27.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-28.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-29.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-3.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-30.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-31.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-32.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-33.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-34.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-35.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-36.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-37.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-38.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-39.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-4.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-40.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-41.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-42.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-43.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-44.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-45.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-46.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-5.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-6.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-7.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-8.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-05-12/2017-05-12_battery-9.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-1.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-11.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-12.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-13.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-14.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-15.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-18.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-19.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-2.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-20.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-21.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-22.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-23.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-24.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-25.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-26.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-27.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-28.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-29.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-3.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-30.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-31.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-32.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-33.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-34.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-35.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-36.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-37.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-38.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-39.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-4.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-40.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-41.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-42.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-43.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-44.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-45.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-46.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-47.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-48.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-5.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-6.csv
A	训练要用的/06_四个数据集/data/MIT data/2017-06-30/2017-06-30_battery-7.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-1.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-10.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-11.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-12.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-14.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-15.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-16.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-18.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-19.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-2.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-21.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-22.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-24.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-25.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-26.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-27.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-29.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-30.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-31.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-33.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-34.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-36.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-37.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-38.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-39.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-4.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-40.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-41.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-42.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-43.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-44.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-45.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-46.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-6.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-8.csv
A	训练要用的/06_四个数据集/data/MIT data/2018-04-12/2018-04-12_battery-9.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#4.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#5.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#6.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-025_1-#7.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#10.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#11.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#12.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#13.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#14.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#15.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#16.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#17.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#18.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#19.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#4.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#5.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#6.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#7.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#8.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-05_1-#9.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#4.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#5.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#6.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#7.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#8.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY25-1_1-#9.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY35-05_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY35-05_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY35-05_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#10.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#11.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#12.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#13.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#14.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#15.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#16.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#17.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#18.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#19.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#20.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#21.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#22.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#23.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#24.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#25.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#26.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#27.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#28.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#4.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#5.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#6.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#7.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#8.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_1_NCA_battery/CY45-05_1-#9.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#10.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#11.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#12.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#13.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#14.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#15.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#16.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#17.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#18.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#19.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#20.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#21.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#22.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#23.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#4.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#5.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#6.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#7.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#8.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY25-05_1-#9.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY35-05_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY35-05_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY35-05_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY35-05_1-#4.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#10.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#11.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#12.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#13.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#14.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#15.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#16.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#17.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#18.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#19.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#20.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#21.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#22.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#23.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#24.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#25.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#26.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#27.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#28.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#4.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#5.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#6.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#7.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#8.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_2_NCM_battery/CY45-05_1-#9.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_1-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_1-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_1-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_2-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_2-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_2-#3.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_4-#1.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_4-#2.csv
A	训练要用的/06_四个数据集/data/TJU data/Dataset_3_NCM_NCA_battery/CY25-05_4-#3.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-1.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-2.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-3.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-4.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-5.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-6.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-7.csv
A	训练要用的/06_四个数据集/data/XJTU data/2C_battery-8.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-1.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-10.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-11.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-12.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-13.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-14.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-15.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-2.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-3.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-4.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-5.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-6.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-7.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-8.csv
A	训练要用的/06_四个数据集/data/XJTU data/3C_battery-9.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-1.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-2.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-3.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-4.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-5.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-6.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-7.csv
A	训练要用的/06_四个数据集/data/XJTU data/R2.5_battery-8.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-1.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-2.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-3.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-4.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-5.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-6.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-7.csv
A	训练要用的/06_四个数据集/data/XJTU data/R3_battery-8.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-1.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-2.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-3.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-4.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-5.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-6.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-7.csv
A	训练要用的/06_四个数据集/data/XJTU data/RW_battery-8.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-1.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-2.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-3.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-4.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-5.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-6.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-7.csv
A	训练要用的/06_四个数据集/data/XJTU data/Sim_satellite_battery-8.csv
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_HUST.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_MIT.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_TJU_0.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_TJU_1.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_TJU_2.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_XJTU_0.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_XJTU_1.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_XJTU_2.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_XJTU_3.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_XJTU_4.pth
A	训练要用的/07_直接运行和微调用的权重/pretrained model/model_XJTU_5.pth
A	训练要用的/开始训练.py
A	训练要用的/说明.md
```
