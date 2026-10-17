# High-dimensional matrix 公平比較計畫

日期：2026-10-13。依據本機／shared commit `e3f17c01246b8a345fc92bd2787257a8abd79b12`。

**本文件是分析及實驗設計，尚未實作新 trainer、修改 model 或執行 training。** 已完成本機更新、靜態 audit 和既有 prediction 數值驗證。本階段到此停止；以下 training 是下一步建議，不是已完成結果或自動執行指令。

## 1. 研究問題與能回答的範圍

> Does converting the battery input into a high-dimensional matrix improve prediction performance for Wang et al. (2024) and Zhang et al. (2025)?

分兩層回答：

1. **主要四組比較：** 在相同 dataset protocol 下，original-input model 與 high-dimensional-input model 的 SOH 誤差差多少？這是 pipeline-level comparison。
2. **較強的因果問題：** 改善是否來自 FFT period reshape，而不是多看歷史、插值增加維度或增加參數？需要另加 representation controls，不能只靠四組下結論。

重要命名：repo 裡 Zhang 是 **BatLiNet-inspired SOH adaptation**，不是原論文 lifetime predictor 的完整復現。四組的 target 全部統一為同 cycle SOH；報告不得稱為忠實重現 Zhang paper 的 original model。若要比較真正的 BatLiNet lifetime model，需另立 raw-curve/lifetime 任務，超出本計畫。

## 2. 四組實驗定義及 input shape

第一階段只用 **MIT 全部可用 batteries**；四組使用同一份資料 manifest。令 B 為 minibatch、T=16、C=16、k=2、H=W=8。

| ID | 報告名稱 | Input／特徵處理 | Architecture／output | 必須重跑？ |
| --- | --- | --- | --- | --- |
| W-O | Wang + original input | 當前 cycle 的 normalized 16 features + cycle index，`[B,17]` | 原 `Solution_u` 形狀、35-input dynamics；output `[B,1]` SOH | 是 |
| W-H | Wang + high-dimensional input | 同尺度歷史 window `[B,16,16]` → `[B,2,16,8,8]`；flatten2048，再 concat17，共 `[B,2065]` | 同 hidden widths／activation／dropout／loss；第一層加寬，dynamics 維持35；output SOH | 是 |
| Z-O | Zhang-inspired + original sequence input | 不做 FFT 或 period reshape。16-cycle standardized feature sequence；減第一 cycle 的 padded sequence，送 CNN 前排成 `[B,16,16,1]` | 兩個 intra/inter encoders：Conv2d(16→32,3×3,pad1)→ReLU→pool→shared Linear32→1；output SOH | 是，repo 尚無這個 control |
| Z-H | Zhang-inspired + high-dimensional input | 同一 history 經 period matrix；減第一 cycle matrix，`[B,2,16,8,8]` → CNN `[B,32,8,8]` | 維持新 script 的兩個 encoders／shared head／mixture；output SOH | 是 |

Z-O 的「original」在這個實驗只表示**不經 period transformation 的現有 CSV history**；不是原 Zhang raw Q–V feature map。用 width=1 的 CNN tensor 是計算格式，沒有 FFT 選週期或 bilinear resize。第一層 input channels 與 Z-H 不同，必須公開這個架構差異。width=1 也使部分 kernel 邊界權重不能接觸非零資料，不能宣稱完全容量相等。

W-H 額外增加 122,880 個第一層 weights，而且 W-O 只看當前 cycle，W-H 看 T 個 cycles；Z-H 也因 channels 翻倍增加 4,608 個 weights／encoder，即兩 encoder 共9,216。四組對照是有用的應用比較，但不是純 matrix 的單一因素實驗。

**不改現有 baseline 檔案。** 下一步如獲指示，使用獨立 analysis runner/adapters，從原類別 import 或獨立定義新增 input layers；舊 `Model/Model.py`、整合 script、reproduction artifacts 全部保留。

## 3. 固定 batteries、split 與 sample identities

採用與隊友方向一致的 **leave-one-condition**，不沿用原 PINN 的 ID %5 rule：

- MIT test：完整 `2018-04-12` 批次，36 batteries。
- 其餘兩批共89 batteries，按排序後 ID list，用 `train_test_split(test_size=0.2, random_state=420)` 按 battery 劃分，預期71 train、18 validation。
- 不 cap batteries；train/valid/test identity 永遠不因 training seed 改變。不得因某模型表現不好移除 battery。
- 在計畫實作時產出含**確切 battery IDs**的共同 `split_manifest.json`；本輪沒有生成或使用新 split 訓練。若缺檔／不可用導致數量不符，停止並記錄，不能默默換資料。
- 每顆保留原 CSV row number 作 cycle coordinate，不能在 outlier handling 後重新編號。
- 以 `(battery_id, original_row_index)` 作唯一 evaluation key。stride=1；pair 是原始相鄰 cycle i、i+1，不跨 battery，遇不合格 pair 不偷偷跳過 gap 接成新相鄰。
- 比較兩個 input representations 時，使用完全相同 train pairs 和 test first-member samples；test 最後一列不被當 first member，與既有 pairing convention 一致。
- 初期歷史不足 T 列：四組的 history control 均左補第一個已觀測有效 feature row；不得補未來數值。

若後續加入 HUST：test group10共8 batteries，其餘69按 battery 80/20，預期55 train／14 valid，四組全部重跑 HUST。不能把 MIT original 與 HUST matrix 指標配對。本階段先完成 MIT，HUST 是第二個獨立 validation dataset。

## 4. 前處理：公平性及 leakage 防護

現有 matrix script 的全軌跡三 sigma／min-max 不能直接作為無 leakage protocol。建議以下**新的、明確命名的 `HD-SOH-v1` protocol**；所有四組同步採用，這是科學流程改動，不是 compatibility fix。

1. 先按 battery ID split，然後才估算 scaler。原始 CSV 不改寫。
2. 不根據 capacity/SOH 值做三 sigma 或好壞樣本篩選。Feature outlier 也不再用 test battery 全軌跡 mean/std 刪除。先保留所有 finite feature values，避免一開始引入額外 outlier threshold。
3. 非 finite feature：只可用同 battery 過去有效值 forward-fill；首筆尚無有效值時使用 training-only feature median。保存 imputation flags 和比率，不能 backward-fill。
4. 只有 missing/nonfinite target 無法計算 supervised loss；該 sample 標記為不可評估，四組共同排除。不得因 target 是高／低 SOH 或某模型誤差大而剔除。缺 target 的 feature row 仍可存在 history，不能用 target 可用性重排時間。若 SOH 為零則保留 MAE/MSE/RMSE，MAPE 對該點標為 undefined，報 count 並在四組使用相同有效 MAPE mask。
5. 在 train batteries 的有效 feature rows 上 fit 16-column min/max。四組共用 scaler：`2*(x-min_train)/(max_train-min_train)-1`；constant column span設1。Validation/test 只 transform，不 fit、不 clip 到[-1,1]。超界是可報告的 distribution shift。
6. 原 cycle index 用 training max cycle 作固定尺度，例如 `2*cycle/max_train_cycle-1`，不使用 test battery 的最終壽命或最後 cycle 來 normalize。零 max 時明確報錯。
7. matrix 使用**同一 normalized features**，不是現有 script 的 raw heterogeneous units。FFT 只讀目前及過去 T rows，不把 capacity、SOH、完整 test trajectory statistics 放進 input。
8. T=16、k=2、8×8固定；padding及插值在四組適用處一致。明確排除 DC 後只在正頻率取 top-k；相同振幅按 frequency index 排序，令 tie 可重現。這是新 protocol 的明確修正，不宣稱與舊 smoke tensors 相同。
9. 對每種 preprocessing／representation 保存 source revision、scaler、參數及 tensor/sample-key hashes，防止原始／matrix 版本實際讀到不同資料。

這個 protocol 解決的是 repository loading/transform 層的已知 leakage。CSV 的早期 raw-feature extraction provenance 不完整，不能因此宣稱原始特徵的全部前處理已被獨立驗證。研究定位仍是給定當前充電 features 的 SOH estimation，不是無需未來 features 的 lifetime forecast。

**必做無訓練檢查：** 只改某顆 validation/test 電池的未來 feature rows，較早 cycle 的 input 不應改變；只改 test labels，所有 inputs、scalers、split、reference bank 必須不變。Label metrics 本身會變，這是正常的。以此檢查 causal window 和 target independence。

## 5. Zhang reference 與 validation 必須獨立控制

- Reference bank 僅 training batteries。其 SOH 可作 reference label，前提是在部署故事中它們是已完成的歷史資料；不能借用 validation/test battery label。
- Train pair reference 使用獨立 RNG，排除自身；Z-O、Z-H 每個 seed 使用相同 reference-ID／row-index schedule。
- Validation/test 固定每個 sample 的兩個 training references，盡可能不重複；保存 reference manifest。使用 original row index 對齊，超過 reference 已知長度時用最後可用 row，記錄 clamp 次數。這不是按真實 SOH matching。
- 同一 sample 先各自算 reference prediction，再平均；不先平均 matrices。
- alpha_mix=0.5，lambda_delta=1.0，n_refs=2固定。
- **每個 model 依自己的 validation SOH MSE 選 checkpoint。** Zhang 必須用最終 mixed prediction MSE；不能再用 Wang 最好 epoch 決定 Zhang。
- 獨立訓練／存檔每個 arm，避免新增另一模型後改變 shared RNG stream 或 early stopping。

## 6. 固定 training settings

先用同一 CPU、目前鎖定環境；不安裝歷史版本、不追 archived accuracy。以下是新 protocol 的預定值，尚未跑：

| 項目 | 固定設計 |
| --- | --- |
| 資料／batch | 同 manifest、stride1、batch512；同一 seed 同一 batch-order schedule；最後 batch 不丟棄 |
| Epochs | 四組皆完整200 epochs；正式比較不 early stop，避免不同模型實際訓練量不一。可因 nonfinite／資源失敗中止，標記 failed，不補跑到較有利數字 |
| Checkpoint | 在200 epochs內，各模型用自身最低 validation MSE 選最佳；另存 final epoch metrics 作診斷 |
| Wang Adam | solution與dynamics兩 optimizer；betas(0.9,0.999)、eps1e-8、weight_decay0 |
| Wang LR | 兩 arm 共用同一原 `LR_Scheduler` 序列與 after-epoch timing：warmup0.002→0.01、30 entries；cosine170 entries至目標0.0002；dynamics0.001固定 |
| Wang loss | current-code data +0.5 PDE +0.01 measured-trend sum；W-O/W-H相同，batch一致；不使用 historical label-only defect |
| Wang architecture | hidden60、sine、dropout0.2、predictor32；只做 input必需的差異，记录初始化方式／參數數量；W-H matrix對偏導detach的語義限制明確報告 |
| Zhang Adam | lr0.001固定，betas(0.9,0.999)、eps1e-8、weight_decay0；Z-O/Z-H相同 |
| Zhang loss | intra MSE + delta MSE；兩 arm 相同；不添加額外 PDE |
| Randomness | Split seed420不變。初步配對 seed42；正式主結果 seeds42、43、44。Python／NumPy／torch各自設定，shuffle與reference用獨立seeded streams |
| Determinism | CPU float32；workers0，固定並記錄 thread count。相同seed不代表不同架構的所有dropout masks相同，不作此承諾 |
| Tuning | 第一輪不做 hyperparameter search。若有必要，只依 validation、對同模型兩 representations 給相同搜尋預算，並在 test 前鎖定 |

Wang 與 Zhang 是不同方法，不要求兩者 optimizer schedule 相同；重點是 **W-O/W-H 相同、Z-O/Z-H 相同**。不得只延長 matrix 組訓練或只為它調 learning rate。

## 7. 現有結果哪些能直接使用？

| Artifact | 可直接做的事 | 不可做的事 |
| --- | --- | --- |
| `results/reproduction/MIT/run_01/` | 既有 PINN reference，重畫既有 prediction、方法背景及已知 runtime | 不能當新 protocol 的 W-O：split、normalization、樣本不同 |
| `historical_run_01/` 和 author archives | reproduction 附錄／歷史差異說明 | 不加入新比較、不再跑02–10 |
| `results/wu_wang_zhang/MIT_2018-04-12_20260926_142820/` | 182點 smoke sanity plots；四個 metric 已重算相符 | 不能當正式 W-H／Z-H |
| `results/wu_wang_zhang/HUST_10_20260926_142824/` | 336點 smoke sanity plots；四個 metric 已重算相符 | 不能當完整 HUST performance |
| `results/wu_wang_zhang/figures/` | 可追溯到 NPZ 的既有圖 | 不能當 original-vs-HD improvement 圖 |
| 額外上傳的 `...result after high dimension matrix/` | Illustrations；標記缺 config/arrays 的 provenance 限制 | 不反推圖上數值成正式實驗結果 |
| `训练要用的/` | 程式／CSV／預訓練權重副本，供參考 | 不是新 HD dataset，也不是配對原始預測結果 |

**最少重跑：MIT 四組各一次，共4個新 runs。** 所有新組都從零訓練，不給某一組獨享預訓練權重。四個單 seed runs 是探索性比較；建議正式數字四組×3 seeds=12 runs。若 MIT和HUST都要正式比較，合計24 runs。任何舊結果都不能直接湊成這個 four-arm table。

## 8. 若要聲稱「matrix 本身有效」，需增加的最小 controls

四組同時涉及不同 history／參數數量。至少增加以下兩組，與 W-H／Z-H 使用完全相同 input shape、network 和 history：

- **W-NF：** 相同16-cycle history，不用FFT。按固定 chronological grid `[16 channels,16 rows,1 column]` resize到8×8，再複製為2 views，flatten2048並concat17。直接用與W-H相同architecture／相同初始化權重。比較 W-NF vs W-H，固定 history、參數量、插值與training budget，測的是adaptive period representation相對固定history layout。
- **Z-NF：** 同一固定 chronological-grid2 views，亦減第一cycle表示，再交與Z-H完全相同的network及reference schedule。比較 Z-NF vs Z-H。

NF 是明確設計的 no-FFT control，不是 paper 原版；複製view不是新增資料。這兩組不能隔離每一項 interpolation effect，但比只比較17維和2065維更接近研究問題。

若要有較強的 matrix-contribution 結論，MIT 至少6 arms×1seed=6 runs；三seeds則18 runs。可先完成四組再決定是否值得追加，但在缺control時必須限制結論措辭。另有shuffle-history/random-reshape control屬後續研究，不列入最低成本方案。

## 9. Metrics、improvement 與 per-battery analysis

令真值 y、預測 p，SOH 均為 fraction：

```text
MAE  = mean(abs(p-y))
MAPE = 100 × mean(abs(p-y)/abs(y))   # 同一非零target mask
MSE  = mean((p-y)^2)
RMSE = sqrt(MSE)
Improvement (%) = 100 × (error_original - error_HD) / error_original
```

正值表示HD較好，負值表示HD較差；分母為零則記 N/A，不加任意epsilon或報無限進步。MAPE的相對improvement不能與MAPE百分點差混淆。

1. 先按 `(battery_id,row_index,seed)` 對齊兩 representations；assert沒有重複、缺漏及true-label差異，不按array位置猜測一致。
2. 每顆 test battery 算四個metrics及四個improvement；報n、cycle範圍、imputation／reference clamp情況。不得將跨battery連接的sample index畫成一條物理degradation trajectory。
3. 報 pooled sample-weighted metrics，另報各battery metrics的等權mean。Macro RMSE是mean(per-battery RMSE)，與sqrt(macro MSE)分開，不能混稱。
4. 每seed做paired improvement；跨seed報每項error與paired improvement的mean±sample SD，保留個別runs。另可報ratio-of-mean-errors，但需分列，兩種計算不等價。
5. 呈現battery勝／負／持平數、最差battery、signed residual對cycle的變化；分析哪種trajectory變好，不只列一個平均數。
6. 若報confidence interval，用battery為cluster的paired bootstrap，不將同battery cycles當獨立樣本。單一held-out condition仍限制對其他conditions的推論；三seed亦不足以宣稱普遍勝出。

## 10. 預計 tables／figures／artifacts

| 輸出 | 內容 |
| --- | --- |
| `protocol_table.csv` | 每arm input、參數數、normalization、split、training設定；清楚標識adaptation |
| `split_manifest.json`／`sample_manifest.csv` | 全部battery IDs、row IDs、seed-independent split、共同sample mask |
| `metrics_by_run.csv` | dataset、arm、seed、四metrics、best epoch、runtime、param count |
| `summary_metrics.csv` | 四組mean±SD；existing baseline分開列為reference |
| `per_battery_metrics.csv` | 每battery每arm每seed四metrics及sample count |
| `paired_improvements.csv` | Wang O-vs-H／Zhang O-vs-H，以及可選NF-vs-H；paired improvements |
| SOH trajectory panels | 每battery真值／original／HD，共用cycle與axes，含最差案例 |
| Error及improvement圖 | paired scatter、per-battery improvement bars、error vs cycle、四metrics分開units |
| Training curves | train data/PDE/physics或intra/delta losses、validation MSE，標best epoch |
| Representation figure | 同一cycle raw history→period selection→reshape→resize；channel與period annotation |
| Runtime／capacity table | preprocessing time、training time、inference time、memory estimate、參數數量 |

每run另存config、scaler、checkpoint、seed/RNG記錄、source revision/diff、prediction檔（含battery/row IDs）、logs、hashes。所有新輸出建議放 `results/high_dimension_analysis/HD-SOH-v1/<dataset>/<arm>/seed_<n>/`；no overwrite，不寫入既有 reproduction/smoke folders。

## 11. 執行順序與品質檢查（尚未執行）

1. 明確建立獨立protocol/runner；保護原script與baselines。
2. 不更新weights的檢查：split disjoint、sample-key equality、scaler train-only、future/target perturbation tests、finite values、shape、參數counts。
3. 小範圍同split的3-epoch smoke；確認兩個model各自checkpoint、matrix shapes、reference restriction、gradient連接。輸出標smoke，不計入正式比較。
4. 以實測per-epoch timing、matrix生成時間及RAM推算正式預算，檢查四組皆可負擔。不要直接使用新script預設cap3/1/1代表正式資料。
5. 鎖定protocol後，按前述4-arm最低方案或3-seed方案完成訓練。每arm最後只用自身best validation checkpoint測試；不依test改設定。
6. 同一analysis程式讀四組prediction、核對identity再算metrics、improvement、tables/figures；任何missing run都顯示missing，不能拿smoke補位。

## 12. 時間與資源估計

以下是**排程估算，不是已benchmark或保證**；本次沒有新training測速。舊PINN約219秒完成86 epochs，但新資料split、2065維layer及matrix cache不同，不能直接乘算。現有smoke logs沒有可用總runtime。

主要成本除了training，還有window FFT／resize及所有pair的matrix materialization。每個`[2,16,8,8]` float32 matrix為8192 bytes。現有PairTable同時存mat1/mat2/intra1/intra2，光這四組約32768 bytes/pair；例如80,000 pairs約2.44 GiB，還未計matrix cache、Python lists、stack temporaries和models。正式前應測量RAM，必要時使用同值的lazy/on-demand cache，不能為省RAM只在其中一組縮資料。

| 工作 | 初步預留 |
| --- | --- |
| 整理protocol、獨立runner、leakage與identity檢查 | 約0.5–1個工作天 |
| 3-epoch smoke、runtime/RAM估算、必要runtime修正 | 約0.5個工作天 |
| MIT四組×1seed訓練 | 暫留數小時至1個工作天，須由smoke timing確認 |
| MIT四組×3seeds | 訓練量約單seed的3倍；若slow/memory-bound可跨天 |
| Tables、per-battery plots、結果解釋及核查 | 約0.5–1個工作天 |

**整體預留：MIT第一版正式四組分析約2–3個工作天；三seed與兩個NF controls約3–5個工作天。** HUST extension額外增加，不包含在最低方案。這是含實作／檢查／作圖的人工作業預算，不代表電腦要連續training幾天。

Smoke完成後更新實測預估：每arm總時間≈preprocessing時間+200×實測epoch時間+evaluation/output時間，再對所有arms／seeds加總；不要假定原PINN timing能代表CNN或matrix版本。

## 13. 本輪完成狀態

- 本機main已更新且HEAD與group/main一致；audit commit另存安全分支。
- Audit已更新，existing predictions的四個metrics重新計算一致。
- 四組研究設計、必要重跑、leakage防護、outputs及時間預算已記錄。
- 沒有修改model code，沒有full training，也沒有新的smoke training。
- 現在停止，下一步應是按本計畫實作獨立比較流程及有界smoke；不自動開始正式training。
