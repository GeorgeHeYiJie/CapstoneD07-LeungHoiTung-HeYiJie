# MIT high-dimensional comparison — 預檢後停止

日期：2026-10-14。狀態：**未開始正式 training；超過使用者的一小時估算上限。**

## 各組估算（不是正式訓練結果）

| 組別 | Training（分鐘） | 每 epoch validation 合計（分鐘） | 含25%緩衝（分鐘） |
| --- | ---: | ---: | ---: |
| W-O | 5.5 | 0.3 | 7.3 |
| W-H | 10.8 | 7.0 | 22.3 |
| Z-O | 12.2 | 3.0 | 19.0 |
| Z-H | 39.5 | 9.7 | 61.5 |

四組 training 合計約 **68.0 分鐘**；validation及緩衝合計約 **110.0 分鐘**。最終test inference、完整matrix首次建立及圖表輸出仍可能增加時間。
估算方法：8個計時training batches的第90百分位，乘完整train batch數及200 epochs；5個validation batches同樣外推。另加25%緩衝。此為單次CPU診斷，非精確保證。
Validation timing包含部分首次matrix cache建構，因此外推可能偏高；但training本身的外推已超過60分鐘，依使用者規則先停止，不以樂觀估算直接開跑。

## 已完成與保留的設計

- 新增獨立預檢工具 scripts/high_dimension_comparison.py；直接import原模型classes，不改原model、dataloader、dataset或舊results。
- 按既有計畫HD-SOH-v1：MIT 71 train、18 validation、36 test batteries；split seed420、training seed42；相同train-only min-max、capacity/1.1 SOH、原row index、stride1、batch512。
- 有效pair數：train 42,784、validation 13,585、test 29,083；四組共用相同pair keys。
- W-O/W-H使用計畫中的17/2065 input；Z-O/Z-H使用計畫中的16/32 CNN input channels。Z-O以既有ZhangSOH(k=1,channels=16)建構，Z-H用k=2；沒有新增層或更改activation。
- 四組各9個診斷minibatch optimizer updates（第一個作warm-up，後8個計時），不是完整epoch或正式run。另每組5個validation batches；沒有test評估。
- 確認預測shape及finite值、參數gradient存在且finite；battery split互斥。對一顆test battery做未來features擾動，較早window/matrix保持不變；更改其labels不影響features。這些是局部檢查，並非完整正式runner驗證。
- scaler、split及tensor/sample-key hashes已保存。診斷權重未保存且已隨process結束丟棄；正式run需重新seed/初始化。
- 沒有利用既有2-epoch smoke predictions產生正式表格。

## 輸出狀態

- preflight/runtime_estimates.json：每組計時、參數數量、診斷loss與估算。
- preflight/split_manifest.json、scaler.json、dataset_manifest.json：資料identity和尺度记录。
- preflight/checks.json、config.json：檢查結果和環境。
- comparison_metrics.csv、per_battery_metrics.csv、overall_comparison.png及prediction_vs_true plots：**未產生**，因為尚無正式predictions；不填入smoke數字或虛構結果。
- MAE、MAPE、RMSE、MSE、improvement、best epoch、stopping epoch：**正式值尚不存在**。200-epoch設計沒有early stopping；未因時間上限自行減少epochs或batteries。

## 下一步

如接受約兩小時級的比較預算，才繼續完成正式輸出runner、完整smoke/checkpoint與metric驗證，然後重新初始化四組、各跑seed42一次。不可將本次9-batch診斷接續成正式訓練。
若要求保持一小時內，需先另外同意調整實驗預算；本輪沒有擅自減epochs、改split、改架構或改dataset protocol。
