# LA-9 Working System v2.0

A mobile-friendly web app with a persistent SQLite backend, historical CSV import, manual fixtures, API-Sports live fixture/result sync, 1X2 predictions, and a central prediction ledger.

## Run locally
1. Install Python 3.11+
2. In this folder run: `python -m venv .venv`
3. Activate it, then `pip install -r requirements.txt`
4. Optional: set `APISPORTS_KEY` in the server environment (never in browser code).
5. Run: `uvicorn app:app --host 0.0.0.0 --port 8000`
6. Open `http://localhost:8000`.

## Use
- Import a CSV with `date,competition,season,home_team,away_team,home_goals,away_goals`.
- Add fixtures manually or configure API-Sports and use Sync live fixtures.
- Generate predictions; predictions are stored in SQLite and visible across devices when hosted.
- Settle predictions as HOME, DRAW or AWAY after full time.

## Live data
Create an API-Football/API-Sports account and obtain a key. Set `APISPORTS_KEY` on the server. The connector currently queries league IDs configured in `app.py`; confirm the league IDs, seasons and competition coverage in your provider dashboard before relying on it. API quotas apply. The Kenya league identifier and season coverage must be verified with the provider.

## Historical data
No historical results are fabricated or bundled. Import verified historical results. Ten seasons across nine competitions may require a paid data plan and substantial API quota. The API-Sports connector currently syncs upcoming fixtures and recent results; historical backfill is intentionally a CSV import until provider plan/season coverage is verified.

## Cloud deployment
Deploy with Docker using `render.yaml` or `Dockerfile`. Configure persistent disk at `/data`, `LA9_DB_PATH=/data/la9.sqlite3`, and `APISPORTS_KEY` in host environment settings. Do not use ephemeral storage for the database. A hosted deployment requires the user's hosting account and provider API key.

## Model limitations
The prediction logic is a transparent statistical baseline using the last ten completed results and home/away form. Several named engines are neutral placeholders when verified pre-match data is unavailable; it does not infer injuries, motivation, odds or travel. It is not a validated 90% model and must be backtested/calibrated on unseen seasons before betting or commercial use. The API connector and app are software components, not a guarantee of provider data coverage or uptime.
