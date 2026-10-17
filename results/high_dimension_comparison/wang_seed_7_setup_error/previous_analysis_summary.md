# Wang MIT 公平比較結果

只完成 W-O / W-H，各seed42一次；Zhang沒有執行。四組預檢的smoke numbers沒有用作正式結果。

共同protocol：HD-SOH-v1；71/18/36 train/validation/test batteries；42,784/13,585/29,083 pairs；SOH=capacity/1.1；train-only min-max，batch512，各200 epochs，不early-stop。
模型input與loss均沿用計畫及預檢；沒有改原模型檔案。W-H第一層較寬並使用16-cycle history，因此本結果是pipeline比較，不能單獨歸因於FFT matrix。

| Metric | Wang original | Wang matrix | Improvement % |
| --- | ---: | ---: | ---: |
| MAE | 0.00640118137 | 0.00771128018 | -20.467 |
| MAPE_percent | 0.688277523 | 0.824349891 | -19.770 |
| RMSE | 0.00802262448 | 0.0125325443 | -56.215 |
| MSE | 6.43625035e-05 | 0.000157064667 | -144.031 |

Improvement = (original error - matrix error) / original error × 100；負值表示matrix較差。MAPE以真值為分母，單位%；MAE/RMSE是SOH fraction，MSE是其平方。

| Arm | Training s | Validation s | Inference cold / warm s | Best epoch | End epoch |
| --- | ---: | ---: | --- | ---: | ---: |
| W-O | 439.98 | 22.05 | 0.25 / 0.25 | 111 | 200 |
| W-H | 920.47 | 43.03 | 6.70 / 0.41 | 60 | 200 |

Training runtime包含batch準備及matrix cache misses；validation另外列出。Inference是完整test資料pipeline耗時，不是純network kernel時間。冷／暖cache是同一checkpoint，已確認predictions完全相同。

每個模型按自己的validation MSE選best checkpoint，reload後validation predictions逐元素完全一致。Test sample keys／labels已逐元素確認一致。Initial seed、source snapshots、split、scaler、hashes和設定在wang_shared/；每arm checkpoint、loss_history、predictions、metrics在各seed_42/。

comparison_metrics.csv是pooled sample-weighted metrics；per_battery_metrics.csv含36顆電池各兩組的metrics及paired improvement。未把cycles當獨立統計replicates；單seed不代表普遍優勝或顯著性。

輸出包括overall_comparison.png、prediction_vs_true.png及36張per-battery SOH trajectory plots。未啟動Zhang或其他seed。

## 補充核查與結果解讀

- Pooled四項誤差全部較差：MAE improvement -20.47%、MAPE -19.77%、RMSE -56.22%、MSE -144.03%。負值是誤差上升，不是改善。
- 36顆test batteries中，matrix有21顆MAE/MAPE較好、15顆較差；RMSE/MSE則18顆較好、18顆較差。因此不能說每顆都退步，pooled結果反映部分電池較大的誤差與sample weighting。macro_battery_metrics.csv另列等battery權重平均，與pooled結果分開。
- Matrix在epoch61出現明顯loss spike（validation MSE約0.1867），之後到epoch200仍未恢復epoch60最佳值。最佳checkpoint在spike之前。training_comparison.png呈現這個穩定性限制；本輪沒有為了結果調參或重跑。
- 此為seed42一次對照；不能證明matrix方法普遍無效，也不能證明惡化單獨由FFT造成。架構容量和history也不同，這點沿用既定計畫。
- 獨立以sklearn和相同float64精度重算pooled metrics與per-battery MSE，確認CSV一致；兩組200 epochs的LR序列完全相同；29,083個evaluation keys唯一且兩組key/label逐元素相同。
- Model、dataset、original training scripts和reproduction artifacts未修改；本輪只新增獨立runner及結果。沒有Zhang正式experiments。
