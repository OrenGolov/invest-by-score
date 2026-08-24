# Paper Trading Engine

## 1. Purpose

Paper trading enables controlled simulation before any real capital is used.

## 2. Rules

- Real capital is prohibited until all validation gates are passed.
- Paper mode must use realistic fills and costs.
- Only approved strategies may trade in paper mode.
- Risk management remains active even in paper mode.

## 3. Minimum paper-trading requirements

The project should not move to live execution until it demonstrates:
- 6+ months of paper trading
- 500+ simulated trades
- stable profitability
- positive Sharpe ratio
- positive profit factor
- acceptable drawdown
- no material unexplained model failures

## 4. Paper execution flow

1. Request score and action.
2. Run ensemble and risk check.
3. If approved, create a simulated order.
4. Apply slippage and fill assumptions.
5. Update portfolio state.
6. Store full trade record and audit trail.

## 5. Risk rules for paper mode

- maximum allocation per position: 1%
- maximum portfolio exposure: 20%
- maximum daily loss: 1%
- maximum weekly loss: 3%
- maximum monthly drawdown: 8%

If limits are breached, revert to analysis-only mode immediately.

## 6. Trade lifecycle record

Every simulated trade must store:
- timestamp
- symbol
- signal source
- action
- size
- entry price
- exit price
- slippage
- realized PnL
- risk flags
- current portfolio state

## 7. Governance

The paper engine must never bypass:
- regime checks
- source quality checks
- risk management
- auditor objections
- abnormal-market rules
