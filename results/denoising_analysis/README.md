# Offline denoising results — saved experiment bundle

All four seed-42 conditions completed on 2026-10-29. Start with [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md), [Test 1](test1/RESULT_ANALYSIS.md), and [Test 2](test2/RESULT_ANALYSIS.md). The English parameter guide is [here](../../docs/PARAMETER_GUIDE.md).

The experiment source was commit `8952464066c66844e624ae5c3d2142296e41b7ba`. Publication preparation first synchronized to shared commit `874a69f2bd45e07f796f39cb0167afe272d2417c`; the three root algorithm scripts were unchanged between these revisions. GPU files newly available under `运行用的Oct29/` were not used for these CPU results.

## Included

- Frozen protocol, preflight, source snapshots, source hashes and data/split/scaler manifests.
- All four formal runs: configurations, best checkpoints, recovery/final states where saved, predictions with sample IDs, losses, runtime and metrics.
- Overall/per-battery/early-cycle/outlier/stability analysis, plots and reports.
- Prepared source arrays, denoised CSVs and verification evidence; bounded smoke evidence is separate from formal results.
- Chinese meeting presentations and chart data.

**The two presentations named `Denoising_Interim_Analysis_CN*.pptx` are historical meeting snapshots. Their Test 2 pending status was accurate at export time and is now superseded by the completed Markdown/CSV reports. They have not been silently rewritten.**

## Local-only generated files

Four rebuildable matrix `.npy` caches under `prepared/original/` and `prepared/denoised/` remain local (about 3.5 GB total). They are excluded explicitly in `.gitignore`; their exact paths, sizes and SHA-256 hashes are recorded in `LOCAL_CACHE_MANIFEST.json`. No local cache has been deleted. Corresponding matrix JSON metadata and all inputs remain included. Document-rendering drafts under repository `tmp/` are also local-only; final documents are retained here.

The runner's `Data` constructor rebuilds missing matrix caches from the saved arrays using the source matrix functions; construction is preprocessing, not training. Do not invoke `prepare` against this existing bundle: it requires a fresh directory. To reproduce training, use a separate checkout/output location and follow [PROTOCOL.md](PROTOCOL.md); preserve these completed results.

This is an OFFLINE comparison: the denoiser uses future-cycle information. One seed, one frozen MIT split, no hyperparameter tuning. Do not infer online deployment performance or causality for individual parameters.
