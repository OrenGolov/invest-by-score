# Monitoring Dashboard

## 1. Objective

The dashboard should provide visibility into model quality, source quality, risk, and score explanations.

## 2. Required sections

### Score monitor
- ticker
- as_of timestamp
- score out of 10
- confidence
- action state
- veto flag

### Evidence panel
- agent output explanations
- supporting sources
- publication timestamps
- quality flags

### Risk monitor
- current exposure
- concentration
- stress state
- drawdown status
- limits status

### Data monitor
- source freshness
- source reliability
- missing-data rate
- stale-data warnings

### Model monitor
- model performance
- drift detection
- calibration metrics
- recent regret or failure count

### Learning monitor
- false positives
- false negatives
- retraining opportunities
- feature contribution changes

## 3. Dashboard design requirements

- simple navigation
- readable by analysts
- exportable logs
- strong explainability
- clear risk warnings

## 4. Alerting rules

Trigger alerts when:
- stale data exceeds threshold
- confidence drops below acceptable range
- model drift is detected
- risk limits are near breach
- large contradictory evidence appears

## 5. Outcome loop

Dashboard data should feed the learning loop, which continuously recalculates:
- source reliability
- feature importance
- model weights
- confidence calibration
- regime sensitivity

This creates a closed loop of gradual self-improvement without human intervention.
