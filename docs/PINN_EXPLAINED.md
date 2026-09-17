# PINN4SOH explained in engineering terms

This document explains the model that is actually implemented in this repository at commit `cc3cc5053caf38f04e0665f7f88cb109144d035e`. It separates facts verified in the code and supplied CSV files from definitions taken from the paper's official supplementary information. No model code was changed to produce this explanation.

The model is an **SOH estimator**. For one battery cycle, it receives 16 summary measurements from the end of charging plus a cycle coordinate. A solution network converts those 17 numbers into one estimated state of health. During training, a second network checks whether the learned SOH surface has a degradation rate that is internally consistent with a learned dynamics relation.

## 1. The 17 model inputs

The checked-in CSVs already contain features 1–16. This repository does not extract them from raw current/voltage time series. [`DF.read_one_csv`](../dataloader/dataloader.py#L43) inserts feature 17, `cycle index`, before the final `capacity` column. [`PINN.forward`](../Model/Model.py#L216) then separates columns 0–15 as `x` and column 16 as `t`.

The feature formulas below come from Supplementary Note 2 of the [official Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-48779-z/MediaObjects/41467_2024_48779_MOESM1_ESM.pdf). The selected voltage window is 4.0–4.2 V for XJTU/TJU and 3.4–3.6 V for MIT/HUST. The selected current window is the decay from 0.5 A to 0.1 A during constant-voltage charging.

All units in the table are **before** the repository normalizes the inputs. The CSV files have no formal unit metadata. Time values are represented in seconds in the supplied feature tables, accumulated charge is represented in ampere-hours, and slopes therefore use seconds. If recreating the CSVs from raw data, verify the raw timestamp unit and include the necessary conversion from A·s to Ah.

| Model column index (0-based) | Feature name | Physical meaning | Units before normalization | Calculation from the selected charging data |
| ---: | --- | --- | --- | --- |
| 0 | `voltage mean` | Average terminal voltage near the end of constant-current charging. A shift can reflect changes in resistance, polarization and electrode behavior as the cell ages. | V | For the selected voltage samples, `mean(V) = (1/n) * sum(V_i)`. |
| 1 | `voltage std` | Amount of voltage variation across that late-charge window. | V | Sample standard deviation: `sqrt(sum((V_i-mean(V))^2)/(n-1))`. |
| 2 | `voltage kurtosis` | Shape/tail concentration of the voltage values; it describes the curve-value distribution rather than a new electrical quantity. | Dimensionless | `sum((V_i-mean(V))^4)/((n-1)*std(V)^4)`. This is the supplement's raw kurtosis expression, with no subtraction of 3. |
| 3 | `voltage skewness` | Asymmetry of the voltage-value distribution within the window. | Dimensionless | `sum((V_i-mean(V))^3)/((n-1)*std(V)^3)`. |
| 4 | `CC Q` | Charge passed while voltage traverses the selected near-cutoff window, which the processed table labels as the CC segment. | Ah | Current integration over the voltage-window time interval: `integral from t_start to t_end of I(t) dt`, expressed in Ah. |
| 5 | `CC charge time` | Time required to traverse the selected voltage window. Aging can change how quickly the cell reaches cutoff. | s | `t_end - t_start` for the selected voltage samples. |
| 6 | `voltage slope` | Average voltage rise rate across the selected voltage window. | V/s | `(V_end - V_start)/(t_end - t_start)`. The supplement defines an endpoint slope, not a fitted regression slope. |
| 7 | `voltage entropy` | A dimensionless description of how the normalized voltage-curve values are distributed. This is signal/curve entropy, not thermodynamic entropy. | Dimensionless | `-sum(p_i * log(p_i))` over normalized voltage values `p_i`. The supplement does not define exactly how `p_i` is normalized. |
| 8 | `current mean` | Average current while the current decays from 0.5 A to 0.1 A in the constant-voltage phase. | A | `mean(I) = (1/n) * sum(I_i)` for the selected current samples. |
| 9 | `current std` | Amount of current variation in that constant-voltage current-decay window. | A | Sample standard deviation: `sqrt(sum((I_i-mean(I))^2)/(n-1))`. |
| 10 | `current kurtosis` | Shape/tail concentration of the current-value distribution. | Dimensionless | `sum((I_i-mean(I))^4)/((n-1)*std(I)^4)`. |
| 11 | `current skewness` | Asymmetry of the current-value distribution. | Dimensionless | `sum((I_i-mean(I))^3)/((n-1)*std(I)^3)`. |
| 12 | `CV Q` | Charge passed while current falls through the selected 0.5–0.1 A constant-voltage window. | Ah | `integral from t_start to t_end of I(t) dt`, expressed in Ah. |
| 13 | `CV charge time` | Duration of the selected constant-voltage current-decay window. | s | `t_end - t_start` for the selected current samples. |
| 14 | `current slope` | Average current decay rate in the selected constant-voltage window. It is normally negative. | A/s | Applying the supplement's slope formula to current: `(I_end - I_start)/(t_end - t_start)`. |
| 15 | `current entropy` | A dimensionless description of the distribution of normalized current-curve values. | Dimensionless | `-sum(p_i * log(p_i))` over normalized current values `p_i`; the exact `p_i` normalization is not specified. |
| 16 | `cycle index` | A time-like usage coordinate: how far the battery has progressed through its measured cycling history. It is cycle count, not elapsed seconds. | Cycle number (dimensionless count) | The loader inserts `0, 1, ..., N-1` from CSV row order with `np.arange` before outlier removal. It is not read from the raw experiment file. |

The authors' general formulas for a curve `z_i` are:

```text
mean(z)       = (1/n) sum_i z_i
std(z)        = sqrt[(1/(n-1)) sum_i (z_i - mean(z))^2]
kurtosis(z)   = sum_i (z_i - mean(z))^4 / [(n-1) std(z)^4]
skewness(z)   = sum_i (z_i - mean(z))^3 / [(n-1) std(z)^3]
duration      = t_end - t_start
charge        = integral I(t) dt
slope         = (z_end - z_start) / (t_end - t_start)
entropy       = -sum_i p_i log(p_i)
```

Before entering the network, all 17 columns are scaled independently **within each battery** by [`DF.read_one_csv`](../dataloader/dataloader.py#L43). The main scripts use min–max normalization:

```text
z_normalized = 2 * (z - z_min)/(z_max - z_min) - 1
```

The alternative `z-score` option is `(z-mean(z))/std(z)`. After either transformation, the neural network inputs are dimensionless. Consequently, the automatic derivatives discussed below are derivatives with respect to normalized coordinates, not directly `SOH/cycle`, `SOH/V` or `SOH/A` in physical units.

## 2. Prediction target and SOH calculation

The final CSV column is `capacity`, in ampere-hours. In [`DF.read_one_csv`](../dataloader/dataloader.py#L43), the code replaces it with:

```text
SOH = measured capacity / fixed nominal capacity
```

The exact code is `df['capacity'] = df['capacity']/nominal_capacity`. The divisors are:

| Dataset | Nominal capacity used by the code |
| --- | ---: |
| XJTU | 2.0 Ah |
| HUST | 1.1 Ah |
| MIT | 1.1 Ah |
| TJU NCA / NCM / NCM+NCA | 3.5 / 3.5 / 2.5 Ah |

For example, an XJTU capacity of 1.80 Ah becomes `1.80/2.00 = 0.90`, or 90% SOH. The code uses these fixed nominal capacities, whereas the paper describes SOH more generally as current available capacity divided by initial capacity. That distinction matters when reporting the reproduction: the implementation's target is the **nominal-capacity ratio**.

The target remains a fraction such as 0.90; it is not min–max or z-score normalized. The solution network outputs one unrestricted scalar and has no sigmoid or clipping layer, so neither the code nor architecture guarantees a prediction between 0 and 1.

The loader constructs adjacent retained-cycle pairs `(x1,y1)` and `(x2,y2)` in [`DF.load_one_battery`](../dataloader/dataloader.py#L68). Both predictions are compared with their same-cycle labels. This is an SOH **estimator**, not a direct next-cycle forecast: `u1` estimates `y1` and `u2` estimates `y2`.

## 3. The three loss terms

The exact loss code is in [`PINN.train_one_epoch`](../Model/Model.py#L237). Let a minibatch contain `B` adjacent-cycle pairs. For pair `j`:

- `y1_j`, `y2_j` are measured SOH values at two successive retained cycles.
- `u1_j`, `u2_j` are solution-network SOH estimates at those cycles.
- `f1_j`, `f2_j` are dynamics residuals returned by [`PINN.forward`](../Model/Model.py#L216).
- `MSE(a,b) = (1/B) sum_j (a_j-b_j)^2`, matching PyTorch `nn.MSELoss()` with its default mean reduction.

### 3.1 Data loss

Exact implementation at [`Model/Model.py:248`](../Model/Model.py#L248):

```text
L_data = 0.5 * MSE(u1, y1) + 0.5 * MSE(u2, y2)
```

Engineering meaning: this is the calibration term. It forces the estimated SOH to agree with measured capacity-based SOH at both members of every adjacent-cycle pair. Without it, the two networks could satisfy each other while predicting the wrong battery capacity.

### 3.2 PDE loss

[`PINN.forward`](../Model/Model.py#L216) first defines the residual:

```text
F_j = dynamics_network(concat(xt_j, u_j, u_x_j, u_t_j))
f_j = u_t_j - F_j
```

Then [`Model/Model.py:252`](../Model/Model.py#L252) implements:

```text
L_PDE = 0.5 * MSE(f1, 0) + 0.5 * MSE(f2, 0)
      = 0.5 * mean((u_t1 - F1)^2) + 0.5 * mean((u_t2 - F2)^2)
```

Engineering meaning: this is an internal rate-consistency term. The gradient of predicted SOH along the cycle coordinate, `u_t`, is pushed toward the rate produced by the learned dynamics network. It encourages the SOH surface and degradation-rate model to describe the same behavior.

This code does **not** embed a first-principles electrochemical equation. The right-hand side is another neural network. Its 35 inputs include `u_t` itself, as well as the original 17 inputs, estimated SOH and the 16 feature gradients. Therefore the implemented residual is best understood as a learned PDE-like consistency relation, not an independent physical law with known battery parameters.

### 3.3 Physics loss

Exact implementation at [`Model/Model.py:255`](../Model/Model.py#L255):

```text
L_physics = sum_j ReLU[(u2_j - u1_j) * (y1_j - y2_j)]
```

Engineering meaning: this is a direction-of-change penalty.

- If measured SOH falls (`y2 < y1`), then `y1-y2` is positive. A predicted rise (`u2-u1 > 0`) produces a positive product and is penalized.
- If measured capacity temporarily recovers (`y2 > y1`), then `y1-y2` is negative. A predicted fall is penalized instead.
- If prediction and measurement change in the same direction, the product is zero or negative and `ReLU` returns zero.

The term therefore follows the measured local trend, including capacity regeneration. It does not impose unconditional monotonic degradation, and it uses the labels to determine the permitted direction. It is summed across the batch rather than averaged, so its numerical scale depends on batch size.

### 3.4 Total loss

Exact implementation at [`Model/Model.py:258`](../Model/Model.py#L258):

```text
L_total = L_data + alpha * L_PDE + beta * L_physics
```

`alpha` and `beta` are dataset-specific command-line arguments. The current main scripts set:

| Dataset script | alpha | beta |
| --- | ---: | ---: |
| XJTU | 0.7 | 0.2 |
| HUST | 0.6 | 0.1 |
| MIT | 0.5 | 0.01 |
| TJU | 0.1 | 0.1 |

The solution and dynamics networks each have an Adam optimizer. A single backward pass through `L_total` supplies gradients to both networks; then both optimizers step.

## 4. What the solution network does

[`Solution_u`](../Model/Model.py#L64) represents the function:

```text
u = solution_u(x, t)
```

Here `x` is the 16-feature charging-health vector and `t` is normalized cycle index. Its job is straightforward: convert what was observed near the end of charging, plus where the cell is in its cycling history, into estimated SOH.

The implemented linear-layer widths are:

```text
17 -> 60 -> 60 -> 32 -> 32 -> 1
```

More exactly:

```text
Encoder:
17 -> Linear(60) -> sin
   -> Linear(60) -> sin -> Dropout(0.2)
   -> Linear(32)

Predictor:
Dropout(0.2) -> Linear(32) -> sin -> Linear(1)
```

Linear weights use Xavier-normal initialization and biases are reset to zero. During validation, testing and ordinary inference, [`PINN.predict`](../Model/Model.py#L183) calls only this solution network. The dynamics network is a training constraint and is not needed to turn a new 17-value vector into an SOH estimate.

## 5. What the dynamics network does

[`PINN.__init__`](../Model/Model.py#L127) creates `dynamical_F` as another fully connected sine-activated MLP. With the current default architecture parameters, its widths are:

```text
35 -> 60 -> 60 -> 1
```

Its 35 input values are constructed in [`PINN.forward`](../Model/Model.py#L216):

```text
17 original normalized inputs xt
+ 1 predicted SOH u
+ 16 gradients u_x with respect to charging features
+ 1 gradient u_t with respect to cycle index
= 35 inputs
```

The scalar output `F` is trained to match `u_t`. In engineering terms, it learns a compact degradation-rate relationship tied to the SOH surface. It can respond to the current cycle location, charging indicators, current SOH estimate and the sensitivity of SOH to those indicators.

It does not directly predict the reported SOH and is not compared with a measured degradation-rate label. It is learned jointly through the PDE residual and the other losses.

## 6. How automatic differentiation is used

Automatic differentiation is implemented at [`Model/Model.py:216–230`](../Model/Model.py#L216):

1. `xt.requires_grad = True` tells PyTorch to track how network output changes with the input tensor.
2. `x = xt[:, 0:-1]` selects the 16 charging features; `t = xt[:, -1:]` selects cycle index.
3. The solution network calculates `u = solution_u(x,t)`.
4. `torch.autograd.grad(u.sum(), t, create_graph=True)` calculates `u_t`, one SOH sensitivity to cycle index for each sample.
5. `torch.autograd.grad(u.sum(), x, create_graph=True)` calculates `u_x`, 16 SOH sensitivities to the charging features for each sample.
6. `create_graph=True` retains the derivative calculations as part of the computational graph. When `L_PDE.backward()` runs, PyTorch can adjust solution-network weights based on errors involving `u_t` and `u_x`, rather than treating those derivatives as fixed numbers.

Summing `u` before calling `grad` is a standard way to obtain per-sample input gradients here because the feed-forward network processes each batch row independently. The code calculates first derivatives only; it does not calculate second spatial or temporal derivatives.

These derivatives are model sensitivities. For example, one component of `u_x` answers: “At this normalized point, how would predicted SOH change for a small change in this normalized feature, according to the network?” They should not automatically be interpreted as causal electrochemical coefficients.

## 7. End-to-end flow

```text
Raw CC-CV charge record for one cycle
        |
        +-- voltage window near charge cutoff
        |      -> 8 voltage/CC statistics
        |
        +-- current decay from 0.5 A to 0.1 A
               -> 8 current/CV statistics
                         |
CSV: 16 pre-extracted features
                         |
loader adds cycle index (input 17)
                         |
outlier removal + per-battery input normalization
                         |
                xt = [x1 ... x16, t]
                         |
              +----------+-----------+
              |                      |
              v                      v
      Solution network         automatic differentiation
       u = predicted SOH       u_x (16 values), u_t (1)
              |                      |
              |        +-------------+
              |        |
              |        v
              |   Dynamics network
              |   F(xt, u, u_x, u_t)
              |        |
              |        v
              |   f = u_t - F
              |        |
              |        +-----------------> PDE loss: f -> 0
              |
              +--------------------------> Data loss: u -> measured SOH
              |
adjacent-cycle predictions u1, u2
and labels y1, y2 ------------------------> Physics loss: match change direction

L_total = L_data + alpha L_PDE + beta L_physics
                         |
                         v
          update solution and dynamics networks

Inference after training:
17 normalized inputs -> solution network only -> estimated SOH
```

## Source boundary

The following are verified directly in this repository: column order, inserted cycle index, normalization, nominal-capacity divisors, adjacent-pair construction, both network structures, automatic-differentiation calls and all loss equations.

The raw feature formulas and selection windows come from the paper's Supplementary Note 2 because raw feature extraction is outside this repository. Neither this repository nor the supplement specifies the exact normalization used to turn curve values into entropy probabilities `p_i`. The CSV files also do not carry machine-readable unit metadata. Those limitations should remain visible if the team later rebuilds features from raw battery logs.

Primary references:

- [Published Nature Communications article](https://www.nature.com/articles/s41467-024-48779-z)
- [Official Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-48779-z/MediaObjects/41467_2024_48779_MOESM1_ESM.pdf)
- [`Model/Model.py`](../Model/Model.py)
- [`dataloader/dataloader.py`](../dataloader/dataloader.py)
