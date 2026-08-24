# API

This folder will hold the external request/response layer for ticker scoring.

## Planned capabilities

- Query a score for a ticker at a given timestamp
- Return score, confidence, reasoning, evidence, and veto state
- Support analysis-only mode and paper-trading mode
- Provide auditable score snapshot access

## Example request

```json
{
  "ticker": "AAPL",
  "as_of": "2026-08-23T14:30:00Z"
}
```

## Example response

```json
{
  "ticker": "AAPL",
  "as_of": "2026-08-23T14:30:00Z",
  "score": 7.4,
  "confidence": 0.78,
  "action": "ANALYSIS_ONLY",
  "reasons": ["Regime supports trend continuation", "Quality data available"]
}
```
