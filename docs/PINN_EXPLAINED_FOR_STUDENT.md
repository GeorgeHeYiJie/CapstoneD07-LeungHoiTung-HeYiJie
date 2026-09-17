# PINN4SOH explained for a student

This guide explains the PINN implemented in this repository in simple engineering language. It assumes you know the basic idea of training a neural network with inputs, predictions, labels and a loss function. It does not assume that you already know partial differential equations (PDEs) or detailed battery electrochemistry.

The shortest description of the model is:

> Use measurements from the end of a charging cycle to estimate battery state of health (SOH), while also encouraging the estimated SOH to change with cycle number in a consistent way.

The detailed feature definitions and exact code references are available in [`PINN_EXPLAINED.md`](PINN_EXPLAINED.md).

## 1. What goes into the model

Each row represents one charge-discharge cycle of one battery. The model receives **17 numbers** from that row:

- **Eight voltage-related features** from a voltage window near the end of constant-current charging: mean, standard deviation, kurtosis, skewness, charge passed, charging time, voltage slope and voltage entropy.
- **Eight current-related features** from the part of constant-voltage charging where current falls from 0.5 A to 0.1 A: mean, standard deviation, kurtosis, skewness, charge passed, charging time, current slope and current entropy.
- **One cycle index**, such as cycle 0, cycle 1 or cycle 200.

These measurements are useful because battery aging changes how the cell accepts charge. For example, an aged cell may take a different amount of time to cross a voltage range, and its voltage or current curve may have a different slope or shape.

The repository already contains these 16 charging features in CSV files. It does not calculate them from raw voltage and current signals. The dataloader adds cycle index as the seventeenth input in [`DF.read_one_csv`](../dataloader/dataloader.py#L43).

Before the inputs enter the neural network, every input column is scaled separately for each battery. The main scripts use a range from -1 to +1:

```text
scaled value = 2 * (value - column minimum) / (column maximum - column minimum) - 1
```

Here, `value` is one measurement from the current cycle. The minimum and maximum come from that feature column for the same battery after the dataloader removes statistical outliers. Scaling prevents a value measured in seconds from dominating a much smaller value measured in volts simply because its number is larger.

The 17 model columns are:

```text
0  voltage mean          9  current standard deviation
1  voltage std          10  current kurtosis
2  voltage kurtosis     11  current skewness
3  voltage skewness     12  CV charge passed
4  CC charge passed     13  CV charge time
5  CC charge time       14  current slope
6  voltage slope        15  current entropy
7  voltage entropy      16  cycle index
8  current mean
```

In the model code, columns 0–15 are called `x`, and column 16 is called `t`. The split happens in [`PINN.forward`](../Model/Model.py#L216).

## 2. What the solution network predicts

The **solution network** predicts one number: battery state of health, or SOH.

In this code, the training target is:

```text
SOH = measured discharge capacity / nominal battery capacity
```

For example, if a nominally 1.1 Ah MIT battery delivers 1.0 Ah, its target is:

```text
SOH = 1.0 Ah / 1.1 Ah = 0.909
```

That means about 90.9% SOH. The calculation is performed in [`DF.read_one_csv`](../dataloader/dataloader.py#L43). The target column is still named `capacity` after division, but its values are now SOH ratios rather than capacities in ampere-hours.

The solution network is an ordinary feed-forward neural network with sine activation functions and dropout. Its layer sizes are:

```text
17 inputs -> 60 -> 60 -> 32 -> 32 -> 1 SOH prediction
```

It learns a function that can be written as:

```text
u = solution_network(x, t)
```

Here:

- `x` means the 16 normalized charging features.
- `t` means normalized cycle index.
- `u` means predicted SOH.

The implementation is in [`Solution_u`](../Model/Model.py#L64). When the trained model is used for an SOH prediction, this is the network that produces the reported answer.

## 3. What `u`, `u_t` and `u_x` mean

These names look like PDE notation, but their engineering meanings are manageable.

### `u`: estimated battery health

`u` is simply the solution network's estimated SOH for one cycle.

For example:

```text
u = 0.94
```

means the model estimates 94% SOH.

### `u_t`: sensitivity to cycle index

`u_t` means the derivative of predicted SOH with respect to normalized cycle index:

```text
u_t = change in predicted SOH / small change in normalized cycle index
```

It describes the local slope of the learned SOH curve along the cycling direction. A negative value means the model's predicted SOH is falling as cycle index increases at that point.

Because cycle index is normalized before entering the network, `u_t` is **not directly SOH lost per physical cycle**. It is a derivative with respect to the scaled cycle coordinate.

### `u_x`: sensitivity to the 16 charging features

`u_x` is a group of 16 derivatives, one for every charging feature:

```text
u_x = [change of u with feature 1, ..., change of u with feature 16]
```

For example, one component describes how the model's predicted SOH would change for a small change in normalized CC charge time while its other inputs remain at the current point.

These values describe the neural network's local sensitivity. They do not prove that changing a feature physically causes the battery to gain or lose health.

The code calculates `u_t` and `u_x` in [`PINN.forward`](../Model/Model.py#L216).

## 4. What the dynamics network does

The **dynamics network** tries to describe how the solution network's SOH changes along the cycle direction.

It receives 35 values:

```text
17 original inputs
+ 1 predicted SOH u
+ 16 feature sensitivities u_x
+ 1 cycle sensitivity u_t
= 35 inputs
```

It produces one number called `F`:

```text
F = dynamics_network(inputs, u, u_x, u_t)
```

The training code asks `F` to agree with `u_t`. It forms a residual:

```text
f = u_t - F
```

If `f` is close to zero, the rate implied by the solution network agrees with the rate produced by the dynamics network.

An engineering analogy is to imagine two descriptions of the same moving vehicle. One system calculates speed from how position changes with time. A second system estimates speed from the vehicle's current operating state. Training encourages the two speed descriptions to agree. In PINN4SOH, the quantity being tracked is SOH rather than position.

The dynamics network is created in [`PINN.__init__`](../Model/Model.py#L127), and `F` and `f` are calculated in [`PINN.forward`](../Model/Model.py#L216).

One subtle point matters: `F` is learned by a neural network and even receives `u_t` as an input. It is not a fixed battery degradation equation taken from electrochemistry. It provides a learned consistency rule.

## 5. Why there are three losses

A loss is a number that tells training how wrong the current model is. This model combines three losses because each one checks a different kind of error. Their code is in [`PINN.train_one_epoch`](../Model/Model.py#L237).

The dataloader supplies two adjacent retained cycles at a time:

```text
earlier cycle: inputs x1, measured SOH y1, predicted SOH u1
later cycle:   inputs x2, measured SOH y2, predicted SOH u2
```

### Data loss: are the SOH estimates accurate?

```text
data loss = 0.5*MSE(u1, y1) + 0.5*MSE(u2, y2)
```

`MSE` means mean squared error. It squares the difference between prediction and measurement, then averages across the batch.

This loss tells the solution network: **predict the measured capacity-based SOH at both cycles**.

### PDE loss: are the two rate descriptions consistent?

```text
f1 = u_t1 - F1
f2 = u_t2 - F2

PDE loss = 0.5*MSE(f1, 0) + 0.5*MSE(f2, 0)
```

Here:

- `u_t1` and `u_t2` are the solution network's cycle-direction slopes.
- `F1` and `F2` are the dynamics network's outputs.
- `f1` and `f2` are their differences.

This loss tells both networks: **make the learned SOH slope and learned dynamics rule agree**.

### Physics loss: does the predicted change point in the measured direction?

```text
physics loss = sum ReLU[(u2 - u1) * (y1 - y2)]
```

`ReLU` returns zero for a negative input and leaves a positive input unchanged.

Suppose measured SOH falls, so `y2` is below `y1`. The factor `y1-y2` is then positive. If the model incorrectly predicts a rise, `u2-u1` is also positive. Their product is positive, so the model receives a penalty.

If measured SOH briefly rises because of capacity regeneration, the allowed direction reverses. The loss therefore follows the local direction found in the measurements. It does not force SOH to decrease at every cycle.

### Total loss

```text
total loss = data loss + alpha*PDE loss + beta*physics loss
```

`alpha` controls the importance of rate consistency. `beta` controls the importance of matching the observed direction of change. For the MIT script, `alpha = 0.5` and `beta = 0.01`.

## 6. How the model learns during backpropagation

One training step works as follows:

1. The dataloader provides a batch of adjacent-cycle pairs.
2. The solution network predicts `u1` and `u2`.
3. PyTorch calculates `u_t` and `u_x` automatically from the solution network.
4. The dynamics network uses the inputs, `u`, `u_t` and `u_x` to calculate `F`.
5. The code calculates data loss, PDE loss and physics loss.
6. It adds them to obtain total loss.
7. `loss.backward()` works backward through every calculation and finds how each trainable weight contributed to total loss.
8. Two Adam optimizers update the weights: one for the solution network and one for the dynamics network.

The words **automatic differentiation** mean that PyTorch follows the mathematical operations used to produce `u`. It can then calculate derivatives such as `u_t` and `u_x` without the programmer deriving and coding those formulas by hand.

The code uses `create_graph=True` when calculating these derivatives. This keeps the derivative calculations connected to the training graph. The PDE loss can therefore change the weights of the solution network through `u_t` and `u_x` during backpropagation.

## 7. What makes this different from a normal MLP

A normal supervised MLP for this problem could use the same 17 inputs and learn only from prediction error:

```text
17 inputs -> MLP -> predicted SOH -> data loss
```

PINN4SOH adds two extra ideas during training:

- It calculates how the predicted SOH changes with cycle index and charging features.
- It adds a dynamics network, PDE consistency loss and direction-of-change loss.

Its training path is therefore:

```text
17 inputs -> solution network -> predicted SOH -> data loss
                    |
                    +-> derivatives -> dynamics network -> PDE loss
                    |
adjacent predictions + measured trend -------------> physics loss
```

After training, ordinary prediction is simpler. [`PINN.predict`](../Model/Model.py#L183) calls only the solution network. The additional dynamics calculations mainly shape what the solution network learns during training.

## 8. What makes it “physics-informed”

The model is called physics-informed because training is guided by knowledge about the form of battery degradation, rather than relying only on point-by-point label fitting.

The main physical idea is that SOH is a changing state. Its change along the cycling direction should have a consistent relationship with battery condition and with the charging indicators. This idea appears in the residual `u_t-F`.

The physics loss also uses an engineering expectation about local change: if measured capacity falls between two cycles, the predicted capacity should not rise over that same pair, and vice versa when measured capacity regeneration occurs.

This is a weaker and more data-dependent use of physics than inserting a known electrochemical equation with parameters such as diffusion coefficients, reaction rates or internal resistance. The name “physics-informed” should therefore be understood in the context of this exact implementation.

## 9. What is based on physical knowledge and what is data-driven

| Part | Main source of information | Explanation |
| --- | --- | --- |
| Charging windows | Battery and charging knowledge | The features focus on late constant-current and constant-voltage behavior, where aging information is visible. |
| Sixteen charging features | Physically meaningful measurements plus statistics | Charge passed, time and slopes have direct engineering meanings. Mean, standard deviation, skewness, kurtosis and entropy summarize curve shape. |
| Cycle index | Physical usage history | More cycling is associated with degradation progression. |
| SOH label | Capacity measurement | The target is measured capacity divided by a fixed nominal capacity. |
| `u_t` | Mathematical description of evolution | It treats SOH as a state that changes along cycle index. Its value is calculated from the learned solution network. |
| Adjacent-cycle physics loss | Measured degradation direction | It discourages a predicted change opposite to the observed local capacity change. |
| Solution network | Data-driven | Its weights and the mapping from 17 inputs to SOH are learned from examples. |
| Dynamics network `F` | Data-driven | Its weights and rate relationship are learned. The code does not supply a known electrochemical formula. |
| PDE residual | Mixed | The form `u_t-F=0` expresses a consistency idea, but `F` itself is learned from data and includes `u_t` as an input. |

The model therefore contains physical structure and physically motivated inputs, but most numerical relationships are learned from data. Calling it physics-informed does not mean that it simulates the internal electrochemical reactions of the cell.

## 10. Worked example: one MIT battery sample through the model

Consider the first checked-in row from [`2018-04-12_battery-9.csv`](../data/MIT%20data/2018-04-12/2018-04-12_battery-9.csv). These are real values stored in the repository:

```text
voltage mean       = 3.4463096 V
voltage std        = 0.0422002 V
voltage kurtosis   = 1.7984974
voltage skewness   = 1.5696604
CC charge passed   = 0.1592229 Ah
CC charge time     = 521.1101 s
voltage slope      = 0.0021161 V/s
voltage entropy    = 4.0716247
current mean       = 0.2210291 A
current std        = 0.1067151 A
current kurtosis   = -0.4088009
current skewness   = 0.8463451
CV charge passed   = 0.0111077 Ah
CV charge time     = 255.4106 s
current slope      = -0.0035495 A/s
current entropy    = 3.5574943
measured capacity  = 1.0679353 Ah
```

### Step 1: create the input and target

Because this is the first CSV row, the dataloader inserts cycle index 0. The raw 17-value input is the 16-feature list above followed by `0`.

MIT uses a fixed nominal capacity of 1.1 Ah, so the label becomes:

```text
y = 1.0679353 / 1.1 = 0.9708503
```

The measured target is therefore about 97.09% SOH.

### Step 2: remove outliers and normalize the inputs

The dataloader first applies its three-standard-deviation outlier filter. It then uses the minimum and maximum of every retained feature column from this battery to scale each of the 17 inputs to the range -1 to +1.

The capacity target `y = 0.9708503` is not scaled again. Only the inputs are min–max normalized.

### Step 3: predict SOH

The normalized 17-value vector passes through the solution network:

```text
17 normalized values -> 60 -> 60 -> 32 -> 32 -> u
```

Assume, only for illustrating the calculation, that the current network produces:

```text
u = 0.965
```

This number is illustrative; it is not a saved result or a result from training in this task.

The prediction error for this row is `0.965 - 0.9708503`. Its squared value contributes to data loss.

### Step 4: calculate the learned sensitivities

PyTorch examines the solution-network calculation and obtains:

- One `u_t` value: sensitivity of the 0.965 prediction to normalized cycle index.
- Sixteen `u_x` values: sensitivity of the prediction to each normalized charging feature.

No measured derivative label is required. These derivatives come from the current neural-network function.

### Step 5: check dynamics consistency

The dynamics network receives:

```text
17 normalized inputs + u + 16 u_x values + u_t = 35 values
```

Suppose, again only as an illustration, that `u_t = -0.006` and the dynamics network gives `F = -0.005`.

Then:

```text
f = u_t - F = -0.006 - (-0.005) = -0.001
```

The square of this small residual contributes to PDE loss. Training tries to move it toward zero.

### Step 6: compare with the adjacent cycle

For training, this row is paired with the next retained row. The model predicts both SOH values and compares their direction of change with the two measured SOH labels. If the measurements rise slightly but the predictions fall, or the measurements fall but the predictions rise, physics loss adds a penalty.

### Step 7: update both networks

The code combines the data, PDE and physics contributions into total loss. Backpropagation then updates both neural networks. Repeating this process over many battery cycles teaches the solution network to estimate SOH while satisfying the two additional consistency checks.

No training was run to create this explanation, and none of the model code was modified.
