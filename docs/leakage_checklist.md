# Leakage Prevention Checklist

## Feature Engineering

- [ ] All rolling windows use `.shift(1)` before `.rolling()`
- [ ] VWAP from current bar is NOT used in features
- [ ] ATR uses `close.shift(1)` as previous close
- [ ] Volume ratio uses shifted volume
- [ ] Time features (hour, minute) come from `bar_open_time`, not future

## Label Generation

- [ ] `binary_label[i] = (close[i+1] > close[i])`
- [ ] `ternary_label[i]` threshold uses ATR from bars prior to `i`
- [ ] Last row is always dropped (no label available)
- [ ] `regression_return_label[i] = log(close[i+1]/close[i])`

## Train/Test Splits

- [ ] `TimeSeriesSplit` used, never `train_test_split` with shuffle
- [ ] Test set always comes after training set in time
- [ ] No hyperparameter tuning on the test set

## Options Chain Features

- [ ] Options snapshot timestamp must be `< bar_open_time` of target bar
- [ ] Stale quotes (>15 min old) are marked and excluded
- [ ] IV from options chain never uses same-bar price

## Inference

- [ ] Inference only runs on bars with `is_closed=True`
- [ ] Model loaded at startup, never retrained during inference
- [ ] Feature snapshot ID stored for auditability

## Known Safe Practices

1. Use `df.shift(1)` consistently — never mix shifted and unshifted columns
2. Always use `iloc[:-1]` when dropping the last label row
3. Walk-forward only: fold i tests on data after fold i-1's training end
4. Simulated fills in paper trading use `bar[i+1].open` (next bar open)

## Common Pitfalls Avoided

- No `StandardScaler.fit()` on full dataset before split
- No feature that is a function of the label
- No options price that includes after-close trades
- No survivorship bias: symbols not filtered by post-hoc performance
