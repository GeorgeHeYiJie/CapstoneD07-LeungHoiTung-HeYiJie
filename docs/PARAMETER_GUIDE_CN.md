# 降噪與綜合模型：參數作用和相互關係

整理日期：2026-11-03。先 fetch 並 fast-forward 至共享 `group/main` 的 `874a69f2bd45e07f796f39cb0167afe272d2417c`，再整理本文。本次沒有修改模型或重新訓練。

已完成實驗的來源版本為 `8952464066c66844e624ae5c3d2142296e41b7ba`；本次同步後，三個根目錄算法檔案相對該版本沒有變更。新增的 `运行用的Oct29/` 是另一套執行入口，不代表已完成的 CPU 實驗使用了 GPU。

**閱讀重點：以下是程式的作用解釋，不是參數調優結果。未做單一參數消融，因此不能把誤差變化歸因於某個參數。**

## 1. 先分清三個程式

| 檔案 | 在今次實驗中的角色 |
| --- | --- |
| [DenoisingAlgorithm.py](../DenoisingAlgorithm.py) | 對每顆電池的 16 個輸入特徵降噪；不改 capacity |
| [HighDimensionMatrix-PINN_Combination.py](../HighDimensionMatrix-PINN_Combination.py) | Test 1：Wang 和 Zhang-inspired 預測器分開訓練／評分，沒有融合成同一 SOH 輸出 |
| [New1HighDimensionMatrix-PINN_Combination.py](../New1HighDimensionMatrix-PINN_Combination.py) | Test 2：Wang 分支與 Zhang-inspired 分支加權融合成一個 SOH 預測 |
| [run_denoising_analysis.py](../scripts/run_denoising_analysis.py) | 已批准的凍結評估適配器：固定資料列、訓練設定、記錄和 checkpoint 選擇；原始算法類別不改 |

Test 2 現有訓練沒有提供 `feat_ref`，所以 `zhang.head_delta` 不參與訓練／預測。不能稱為已完整實現兩篇論文的所有跨電池學習方法。

## 2. 降噪參數

下列值來自 `DenoisingAlgorithm.py` 的常數和函數，並用於本次降噪資料。

| 參數 | 值 | 工程意義與關係 |
| --- | --- | --- |
| `DATA_DIR` / `OUTPUT_DIR` | 原始腳本是佔位輸入／`./denoised_for_matrix` | 路徑參數，不是模型超參數。本次適配器逐日期目錄處理，輸出到 `prepared/denoised_csv/`，沒有覆蓋原資料 |
| `FEATURE_COLS` | 16 欄 | 決定哪些特徵被處理；欄名與模型一致 |
| `CAPACITY_COL` | `capacity` | 保留原值，SOH 標籤不降噪 |
| `SG_WINDOW` | 11 | Savitzky–Golay 局部窗口，涵蓋連續資料列。窗口加大通常更偏向平滑，但可能削弱短期特徵；未實驗驗證最佳值 |
| `SG_POLY` | 2 | 窗口內擬合二次多項式。必須小於窗口長度；與窗口共同決定局部曲線的彈性 |
| `GCL_SIGMA` | 0.08 | 殘差權重的尺度。相同殘差下，增大 sigma 通常令權重衰減較慢 |
| `GCL_ALPHA` | 1.5 | 控制殘差權重曲線形狀；不是 PINN 的 PDE loss 權重 |
| `GCL_ITER` | 3 | 殘差修正次數。增加次數不等於必然更平滑，因為更新會把部分原始殘差加回去 |
| 函數內修正係數 | 0.4 | `filtered += 0.4 * w * residual`；控制每次加回原訊號殘差的幅度 |
| `SCALES` | `[2,4,8,16]` | 分別用不同大小的連續區塊取平均，再插值回原長度 |
| `FUSION_MODE` | `weighted` | 使用加權融合；另一選項 `average` 是等權平均 |
| `SCALE_WEIGHTS` | `[0.4,0.3,0.2,0.1]` | 依序對應尺度 2、4、8、16；程式會把權重正規化。偏重大尺度會更偏向其粗略趨勢，但不保證更準 |

實際次序：**原始特徵 → SG 平滑 → GCL 殘差修正 → 多尺度平均／插值 → 加權融合 → 訓練集正規化 → 高維矩陣**。

特別注意：降噪發生在正規化之前，16 欄的物理單位／數值範圍不同，但共用 `GCL_SIGMA=0.08`。所以它對不同欄位並不代表相同的相對修正尺度。這是需要研究的可能因素，不是已證實的誤差原因。

多尺度函數使用 `L // scale` 個完整區塊，尾端不足一塊的資料不直接進入區塊平均；之後用全長等距座標插值回 L 筆。輸出列數不變不代表每個原始時間位置完全不受影響。

**OFFLINE 限制：**中心平滑及全軌跡池化／插值使用未來週期資訊。曲線更平滑不等於更接近未知真實訊號，也不等於 SOH 預測更準。

## 3. 高維矩陣與輸入設定

| 項目 | Test 1 | Test 2 | 作用／關係 |
| --- | --- | --- | --- |
| 原始特徵 | 16 | 16 | 每一筆已有充電特徵，不是原始電壓波形 |
| `xt` | 17 維 | 17 維 | 16 個正規化特徵，加正規化資料列時間座標 |
| `WINDOW` / `--window` | 16 | 16 | 建矩陣的歷史窗口；不是降噪 `SG_WINDOW=11` |
| `K_PERIODS` / `--k` | 2 | 3 | 保留的週期表示數；改動會影響表示及部分網絡輸入通道 |
| 矩陣高、寬 | 8 × 8 | 8 × 8 | 將不同形狀週期矩陣調整至固定尺寸，便於模型處理 |
| 每筆矩陣形狀 | `[2,16,8,8]` | `[3,16,8,8]` | 批次時前面再加 B；依序為週期、特徵、高、寬 |

矩陣是對既有特徵歷史的重新組織，不是新增感測量測。兩個腳本的週期選擇邏輯也不同；不能只因窗口同為 16，就認為它們使用相同表示。因為原始／降噪在同一 Test 內固定設定，主比較仍是該模型的整套資料管線有無降噪。

## 4. Loss 與混合權重

### Test 1：兩個預測器分開計算

Wang：`L = L_data + 0.5 * L_PDE + 0.01 * L_trend`。

- `alpha=0.5`：PDE 殘差的重要程度。
- `beta=0.01`：預測變化方向與標籤變化方向相反時的懲罰權重。
- `L_trend = sum(ReLU((u2-u1)*(y1-y2)))`；u 是預測、y 是標籤，1、2 是相鄰資料列。
- data/PDE 使用平均 MSE，trend 使用總和。因此 batch size 改變時，三項相對尺度可能改變；不能單看 beta 數字判斷約束強弱。

Zhang：`L = L_intra + 1.0 * L_delta`。

- `lambda_delta=1`：跨電池 SOH 差值學習的 loss 權重。
- `alpha_mix=0.5`：輸出為 `0.5*y_intra + 0.5*(y_delta+y_ref)`。
- 這裡混合的是 **Zhang 的單電池預測及參考電池差值預測**，不是 Wang 與 Zhang。
- `n_refs=2`：驗證／測試對兩個固定訓練參考電池的結果取平均；不讀取測試參考標籤。

### Test 2：融合輸出

`y_hat = sigmoid(mix)*u_Wang + (1-sigmoid(mix))*y_Zhang`。

| 參數 | 值 | 意義 |
| --- | --- | --- |
| `MIX_INIT` | 0.5 | 是可學習參數 mix 的初始值。經 sigmoid 後，初始 Wang 比重約 62.25%、Zhang 約 37.75%，不是各半 |
| `ALPHA_PHYS` | 0.1 | PDE loss 權重，與輸出混合比例不同 |
| `BETA_MONO` | 0.01 | Wang 分支的相鄰趨勢懲罰權重 |
| 矩陣 correction 係數 | 0.1 | `u_Wang = 主路徑輸出 + 0.1*矩陣分支輸出`；不是 PDE 權重，雖然數值相同 |
| `HIDDEN` | 60 | Wang encoder 隱藏寬度；改變會改模型容量／架構 |
| Dropout | 0.2 | 訓練時隨機遮蔽部分表示；評估時關閉，受隨機種子影響 |

Test 2：`L = MSE(y_hat,y) + 0.1*mean(f²) + 0.01*L_trend`。
資料 loss 約束融合輸出；PDE 和 trend 作用於 Wang 分支，其中 trend 使用 Wang 的 u，並非融合輸出 y_hat。mix 與各分支透過訓練相互影響，因此不能只以最後混合比例判斷某個分支的獨立因果貢獻。

## 5. 正式實驗實際設定（不是原始入口預設）

| 設定 | Test 1 | Test 2 |
| --- | --- | --- |
| 模型 seed | 42 | 42 |
| Epochs | 200，沒有 early stopping | 200，沒有 early stopping |
| Optimizer | 三個 Adam：Wang solution、dynamics、Zhang | 一個 Adam：combined model |
| Learning rate | solution：30 epoch warmup，0.002 → 0.01，之後 cosine 趨向 0.0002；dynamics/Zhang：0.001 | 固定 0.001，無 scheduler |
| 更新單位 | 64 組相鄰資料對／批次 | 一組相鄰資料對，即 2 筆／更新 |
| Training stride | 1 | 2，不重疊相鄰對 |
| Best checkpoint | Wang、Zhang 各自最低完整驗證 SOH MSE | 融合預測最低完整驗證 SOH MSE |
| CPU threads | 1 | 1 |

共同固定：MIT 71/18/36 顆電池；29,083 測試樣本；SOH = capacity / 1.1 Ah；電池順序、row IDs、標籤與評估樣本一致；不重新做 outlier filtering。split seed 是 420，模型 seed 是 42，兩者用途不同。參考抽樣 seed：訓練 10042、驗證 20042、測試 30042；Test 2 未啟用參考電池分支。

兩個 condition **採相同訓練集正規化方法，但分別擬合 scaler 數值**。因此本次比較的是「降噪＋對應訓練集 scaler」整套管線，不是固定同一組 scaler 係數的消融。

原始 Test 1 入口預設 2 epochs、少量電池、stride 5；Test 2 預設 50 epochs；兩者預設 seed 420。直接執行原始入口不會重現本次正式協定。完整偏差與 runtime 修正見 [PROTOCOL.md](../results/denoising_analysis/PROTOCOL.md)，以各 run 的 `configuration.json` 為實際設定依據。

## 6. 已有說明與解讀界線

- [原始 README](../README.md)：安裝、資料與執行方式，沒有完整參數相互關係手冊。
- Test 1 程式頂部和 `argparse`：流程、公式和參數用途較詳細。
- 降噪／Test 2：「可調參數區」有中文短註解，但隱藏的固定係數和實際分支使用情況需看函數。
- [PINN 詳解](PINN_EXPLAINED.md)、[學生版](PINN_EXPLAINED_FOR_STUDENT.md)：原始模型及 loss 意義。
- [Loss 權重差異](LOSS_WEIGHT_DISCREPANCY.md)：歷史版本與現行公式差異。不同公式下的權重數字不能直接等價比較。
- [最終降噪結果](../results/denoising_analysis/ANALYSIS_SUMMARY.md)：已完成的 Test 1／Test 2；各自附逐電池和早期週期分析。

目前可說：seed 42 下，Test 1 Wang 和 Test 2 combined 的整體誤差上升，Test 1 Zhang 的整體誤差下降。不能說已證明某個窗口、GCL 核、FFT 或混合係數是原因。要識別原因，未來需另行批准控制單一變數的實驗；本文不啟動任何新訓練。
