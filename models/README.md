# Models

This folder will host model registry, feature pipelines, and ensemble logic.

## Planned contents

- feature engineering pipeline
- model registry metadata
- ensemble weighting logic
- calibration for confidence and probability
- retraining and performance tracking

## Ensemble policy

- Never rely on a single model family.
- Use several model families with rolling performance-based weighting.
- Require out-of-sample validation before promotion.
- Shrink toward equal weighting during uncertain periods.
