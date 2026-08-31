# Investment Project

Continuously learn the relationship between information, market context, price/volume behavior, and future stock returns across multiple horizons — while preserving exactly what was knowable at the prediction time.

Point-in-time stock scoring and long-term portfolio analysis platform.

The platform is designed as a governed multi-agent system. It produces an
explainable ticker score for a requested timestamp, learns from forecast
outcomes, and remains in analysis or paper-trading mode until strict evidence
and risk criteria are met. A score is not a promise of return and is not
financial advice.

## Current Status

The repository currently contains the first data foundation in
[`fetch_data.py`](fetch_data.py): Yahoo Finance OHLCV retrieval with local
Parquet caching. The system design is documented before implementation so that
data lineage, point-in-time correctness, agent contracts, and veto rules are
testable from the start.

## Design Documents

- [`docs/architecture.md`](docs/architecture.md): system boundaries, data flow, and timestamp policy
- [`docs/data-model.md`](docs/data-model.md): database schema, lineage, quality, and source reliability
- [`docs/agents.md`](docs/agents.md): ten agent responsibilities, contracts, ensemble, and veto logic
- [`docs/features-and-models.md`](docs/features-and-models.md): feature engineering, model families, and calibration
- [`docs/validation.md`](docs/validation.md): backtesting, paper trading, monitoring, and release gates
- [`docs/roadmap.md`](docs/roadmap.md): incremental implementation plan and definition of done
- [`docs/sprint-2.md`](docs/sprint-2.md): Sprint 2 deliverables and approval checklist
- [`db/migrations/001_initial_schema.sql`](db/migrations/001_initial_schema.sql): schema-only TimescaleDB migration

## Non-Negotiable Rules

1. Point-in-time data only: a prediction may use information available by its
	requested timestamp, including publication and exchange timestamps.
2. No single model or agent may create a trade decision.
3. Risk Management and Performance Auditor can veto any proposed trade.
4. High uncertainty, contradictory evidence, stale data, or abnormal market
	conditions produces `NO_TRADE` or `ANALYSIS_ONLY`.
5. Real-capital execution is disabled until the validation gates in
	[`docs/validation.md`](docs/validation.md) are passed and explicitly
	approved.

## Existing Data Collector

```powershell
python fetch_data.py
```

The current collector is a prototype input adapter, not a production-grade
real-time or execution service. Provider licensing, rate limits, timestamps,
corporate actions, and source outages must be handled before deployment.
## github
git add .
git commit -m ""
git push origin spring-vs
## Open on Another PC
web_app.py
PS C:\Users\Oren.Golovchik\Desktop\investment-project> python .\web_app.py


Clone the repository and recreate the virtual environment locally. The local
`venv/`, `__pycache__/`, and generated `data/` cache are intentionally not
stored in GitHub.

```powershell
git clone https://github.com/OrenGolov/investment-project.git
cd investment-project
py -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python fetch_data.py
```

The last command downloads fresh market data and recreates the local `data/`
cache. If PowerShell blocks activation, run the project with
`venv\Scripts\python.exe` directly or adjust the local execution policy.
