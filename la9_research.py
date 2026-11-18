#!/usr/bin/env python3
"""
LA-9 v1.0 — independent Home/Draw/Away research model.

This is a research implementation, not a betting guarantee.
It learns only from pre-match information available in the supplied CSV.
Bookmaker odds are excluded from model fitting and used only for a post-model
market comparison.

Install:
    pip install pandas numpy scikit-learn

Run:
    python la9.py --matches matches.csv --out la9_results

Required CSV columns:
    date, competition, season, home_team, away_team, home_goals, away_goals

Optional pre-match columns:
    home_xg, away_xg, home_absences, away_absences,
    home_key_absences, away_key_absences,
    home_rest_days, away_rest_days,
    home_travel_km, away_travel_km,
    home_odds, draw_odds, away_odds

Important:
- Every row is a completed match; scores are final scores.
- Optional injury/rest/travel/odds fields must be values known BEFORE kick-off.
- Do not put post-match or closing information into pre-match fields unless
  it was genuinely available at the prediction timestamp.
- Use decimal odds. Leave unknown values blank, not zero.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve

REQUIRED = [
    "date", "competition", "season", "home_team", "away_team",
    "home_goals", "away_goals"
]
OPTIONAL = [
    "home_xg", "away_xg",
    "home_absences", "away_absences",
    "home_key_absences", "away_key_absences",
    "home_rest_days", "away_rest_days",
    "home_travel_km", "away_travel_km",
    "home_odds", "draw_odds", "away_odds"
]

# Proposed starting weights for the nine interpretable LA-9 engines.
# These are design priors, NOT empirically proven optimal weights.
ENGINE_WEIGHTS = {
    "hidden_strength": 0.18,
    "venue_dna": 0.16,
    "fortune_decomposition": 0.12,
    "opponent_weakness": 0.12,
    "motivation": 0.05,
    "fatigue": 0.08,
    "market_blind_spot": 0.09,
    "chaos": 0.10,
    "signal_agreement": 0.10,
}

def safe_mean(values, default=np.nan):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else default

class TeamState:
    """Stores only information from matches already played."""
    def __init__(self, keep=10):
        self.keep = keep
        self.matches = []

    def add(self, gf, ga, xgf=np.nan, xga=np.nan, venue=None, date=None):
        self.matches.append({
            "gf": float(gf), "ga": float(ga),
            "xgf": float(xgf) if pd.notna(xgf) else np.nan,
            "xga": float(xga) if pd.notna(xga) else np.nan,
            "venue": venue, "date": date
        })
        self.matches = self.matches[-self.keep:]

    def stats(self, venue=None):
        rows = self.matches if venue is None else [
            r for r in self.matches if r["venue"] == venue
        ]
        if not rows:
            return {
                "gf": np.nan, "ga": np.nan, "xgf": np.nan, "xga": np.nan,
                "points": np.nan, "n": 0, "fortune": np.nan
            }
        points = []
        for r in rows:
            points.append(3 if r["gf"] > r["ga"] else 1 if r["gf"] == r["ga"] else 0)
        xgf = safe_mean([r["xgf"] for r in rows])
        xga = safe_mean([r["xga"] for r in rows])
        gf = safe_mean([r["gf"] for r in rows])
        ga = safe_mean([r["ga"] for r in rows])
        fortune = (gf - xgf) - (xga - ga) if np.isfinite(xgf) and np.isfinite(xga) else np.nan
        return {
            "gf": gf, "ga": ga, "xgf": xgf, "xga": xga,
            "points": safe_mean(points), "n": len(rows), "fortune": fortune
        }

def make_features(df):
    """Build chronological, strictly lagged features for every match."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if df["date"].isna().any():
        raise ValueError("Some dates could not be parsed. Use YYYY-MM-DD.")
    df = df.sort_values(["date", "competition"]).reset_index(drop=True)

    states = {}
    last_played = {}
    feature_rows = []

    for _, row in df.iterrows():
        comp = str(row["competition"])
        ht, at = str(row["home_team"]), str(row["away_team"])
        hkey, akey = (comp, ht), (comp, at)
        hs = states.setdefault(hkey, TeamState())
        aws = states.setdefault(akey, TeamState())

        h_all, a_all = hs.stats(), aws.stats()
        h_home, a_away = hs.stats("home"), aws.stats("away")
        h_away, a_home = hs.stats("away"), aws.stats("home")

        def rest(team_key):
            prev = last_played.get(team_key)
            if prev is None:
                return np.nan
            return max(0.0, (row["date"] - prev).total_seconds() / 86400.0)

        # Engine 1: hidden strength — recent scoring/defending and points.
        hidden = (
            (h_all["gf"] - a_all["ga"]) if np.isfinite(h_all["gf"]) and np.isfinite(a_all["ga"]) else np.nan
        )
        hidden_xg = (
            (h_all["xgf"] - a_all["xga"]) if np.isfinite(h_all["xgf"]) and np.isfinite(a_all["xga"]) else np.nan
        )

        # Engine 2: venue DNA — venue-specific points and goal difference.
        venue = (
            (h_home["points"] - a_away["points"])
            if np.isfinite(h_home["points"]) and np.isfinite(a_away["points"]) else np.nan
        )
        venue_gd = (
            ((h_home["gf"] - h_home["ga"]) - (a_away["gf"] - a_away["ga"]))
            if all(np.isfinite(v) for v in [h_home["gf"], h_home["ga"], a_away["gf"], a_away["ga"]])
            else np.nan
        )

        # Engine 3: fortune decomposition; positive means recent results
        # exceeded xG-based expectation. Model learns whether it predicts regression.
        fortune_diff = (
            h_all["fortune"] - a_all["fortune"]
            if np.isfinite(h_all["fortune"]) and np.isfinite(a_all["fortune"]) else np.nan
        )

        # Engine 4: opponent weakness — recent opponent goals conceded,
        # expressed as home attack vs away defence and vice versa.
        opp_weak = (
            (h_all["gf"] + a_all["ga"]) - (a_all["gf"] + h_all["ga"])
            if all(np.isfinite(v) for v in [h_all["gf"], a_all["ga"], a_all["gf"], h_all["ga"]])
            else np.nan
        )

        # Engine 5: motivation proxy is deliberately conservative.
        # League-table stakes cannot be inferred reliably from scores alone.
        # Use season phase only as a weak feature; no invented motivation labels.
        season_phase = np.nan
        season_rows = df[(df["competition"] == row["competition"]) &
                         (df["season"] == row["season"]) & (df["date"] <= row["date"])]
        if len(season_rows) > 0:
            season_phase = min(1.0, len(season_rows) / 380.0)

        # Engine 6: fatigue — rest-day difference and travel if supplied.
        h_rest = row.get("home_rest_days", rest(hkey))
        a_rest = row.get("away_rest_days", rest(akey))
        try:
            rest_diff = float(h_rest) - float(a_rest)
        except (TypeError, ValueError):
            rest_diff = np.nan
        try:
            travel_diff = float(row.get("away_travel_km", np.nan)) - float(row.get("home_travel_km", np.nan))
        except (TypeError, ValueError):
            travel_diff = np.nan

        # Engine 7: market blind spot is calculated AFTER prediction, not used
        # to train the independent model. Placeholder feature is excluded.
        # Engine 8: chaos proxies from pre-match absences and uncertainty.
        try:
            absence_diff = float(row.get("away_absences", np.nan)) - float(row.get("home_absences", np.nan))
        except (TypeError, ValueError):
            absence_diff = np.nan
        try:
            key_absence_diff = float(row.get("away_key_absences", np.nan)) - float(row.get("home_key_absences", np.nan))
        except (TypeError, ValueError):
            key_absence_diff = np.nan

        # Engine 9: agreement — agreement between goals, points and venue signals.
        signals = [hidden, hidden_xg, venue, venue_gd, opp_weak]
        finite_signs = [np.sign(v) for v in signals if np.isfinite(v) and v != 0]
        agreement = abs(np.mean(finite_signs)) if finite_signs else np.nan

        feature_rows.append({
            "hidden_strength": hidden,
            "hidden_xg_strength": hidden_xg,
            "venue_dna": venue,
            "venue_goal_difference": venue_gd,
            "fortune_decomposition": fortune_diff,
            "opponent_weakness": opp_weak,
            "season_phase_proxy": season_phase,
            "fatigue_rest_difference": rest_diff,
            "fatigue_travel_difference": travel_diff,
            "chaos_absence_difference": absence_diff,
            "chaos_key_absence_difference": key_absence_diff,
            "signal_agreement": agreement,
            "home_sample_size": h_all["n"],
            "away_sample_size": a_all["n"],
            "home_venue_sample_size": h_home["n"],
            "away_venue_sample_size": a_away["n"],
        })

        # Update states ONLY after features for this match are calculated.
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        hxg, axg = row.get("home_xg", np.nan), row.get("away_xg", np.nan)
        hs.add(hg, ag, hxg, axg, "home", row["date"])
        aws.add(ag, hg, axg, hxg, "away", row["date"])
        last_played[hkey] = row["date"]
        last_played[akey] = row["date"]

    X = pd.DataFrame(feature_rows)
    # Engine 7 (market) remains excluded from X to preserve independence.
    return df, X

def market_probabilities(df):
    odds = df[["home_odds", "draw_odds", "away_odds"]].apply(pd.to_numeric, errors="coerce")
    valid = (odds > 1.0).all(axis=1)
    raw = 1.0 / odds
    norm = raw.div(raw.sum(axis=1), axis=0)
    norm.loc[~valid, :] = np.nan
    norm.columns = ["market_home", "market_draw", "market_away"]
    return norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True, help="Historical CSV")
    ap.add_argument("--out", default="la9_results")
    ap.add_argument("--min-history", type=int, default=5)
    args = ap.parse_args()

    path = Path(args.matches)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["target"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]],
        ["H", "D"], default="A"
    )

    df, X = make_features(df)
    # Require a minimum of previous matches for both teams.
    eligible = (X["home_sample_size"] >= args.min_history) & (X["away_sample_size"] >= args.min_history)
    df, X = df.loc[eligible].reset_index(drop=True), X.loc[eligible].reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("At least 100 eligible historical matches are required; 1,000+ is strongly preferred.")

    # Chronological split: 60% fit, 20% calibration/validation, 20% final test.
    n = len(df)
    i1, i2 = int(n * .60), int(n * .80)
    X_train, y_train = X.iloc[:i1], df["target"].iloc[:i1]
    X_val, y_val = X.iloc[i1:i2], df["target"].iloc[i1:i2]
    X_test, y_test = X.iloc[i2:], df["target"].iloc[i2:]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000, C=0.5, class_weight=None,
            multi_class="auto"
        ))
    ])
    model.fit(X_train, y_train)

    # Selective temperature scaling on validation data only.
    # This is a simple calibration adjustment; final test remains untouched.
    val_probs = model.predict_proba(X_val)
    classes = list(model.named_steps["classifier"].classes_)
    class_index = {c: i for i, c in enumerate(classes)}
    y_val_idx = np.array([class_index[y] for y in y_val])
    best_temp, best_ll = 1.0, float("inf")
    for temp in np.arange(0.70, 1.61, 0.05):
        z = np.log(np.clip(val_probs, 1e-9, 1.0)) / temp
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        ll = log_loss(y_val, p, labels=classes)
        if ll < best_ll:
            best_temp, best_ll = float(temp), ll

    test_raw = model.predict_proba(X_test)
    z = np.log(np.clip(test_raw, 1e-9, 1.0)) / best_temp
    test_probs = np.exp(z - z.max(axis=1, keepdims=True))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    # Explicitly map classes in case a split lacks a class.
    pmap = {c: test_probs[:, i] for i, c in enumerate(classes)}
    result = df.iloc[i2:].copy()
    result["p_home"] = pmap.get("H", np.zeros(len(result)))
    result["p_draw"] = pmap.get("D", np.zeros(len(result)))
    result["p_away"] = pmap.get("A", np.zeros(len(result)))
    result["predicted_1x2"] = np.array(classes)[test_probs.argmax(axis=1)]
    result["correct_1x2"] = (result["predicted_1x2"] == y_test.to_numpy())

    # Opportunity Hunter: 1X2 only. Thresholds are deliberately conservative
    # starting gates, not claims of 90% accuracy.
    mkt = market_probabilities(result)
    result = pd.concat([result.reset_index(drop=True), mkt.reset_index(drop=True)], axis=1)
    result["home_edge"] = result["p_home"] - result["market_home"]
    result["away_edge"] = result["p_away"] - result["market_away"]
    result["la9_pick"] = "NO BET"
    home_gate = (result["p_home"] >= 0.70) & (result["home_edge"] >= 0.04)
    away_gate = (result["p_away"] >= 0.70) & (result["away_edge"] >= 0.04)
    result.loc[home_gate & (result["p_home"] >= result["p_away"]), "la9_pick"] = "HOME"
    result.loc[away_gate & (result["p_away"] > result["p_home"]), "la9_pick"] = "AWAY"

    selected = result[result["la9_pick"] != "NO BET"].copy()
    selected["selected_correct"] = np.where(
        selected["la9_pick"] == "HOME", selected["target"] == "H",
        selected["target"] == "A"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "la9_test_predictions.csv", index=False)
    selected.to_csv(out / "la9_selected_opportunities.csv", index=False)

    metrics = {
        "eligible_matches": int(n),
        "train_matches": int(len(X_train)),
        "validation_matches": int(len(X_val)),
        "untouched_test_matches": int(len(X_test)),
        "test_1x2_accuracy": float(accuracy_score(y_test, result["predicted_1x2"])),
        "test_1x2_log_loss": float(log_loss(y_test, test_probs, labels=classes)),
        "opportunities_selected": int(len(selected)),
        "selected_home_away_accuracy": float(selected["selected_correct"].mean()) if len(selected) else None,
        "home_selections": int((selected["la9_pick"] == "HOME").sum()),
        "away_selections": int((selected["la9_pick"] == "AWAY").sum()),
        "calibration_temperature": best_temp,
        "warning": "Research output only. Accuracy and ROI are unknown until real data is supplied and tested."
    }
    (out / "la9_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results in: {out.resolve()}")

if __name__ == "__main__":
    main()
