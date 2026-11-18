# LA-9 data dictionary

## Required columns
| Column | Type | Meaning |
|---|---|---|
| date | YYYY-MM-DD | Match kick-off date; use ISO timestamp if available |
| competition | text | Competition name, e.g. EPL, La Liga, Kenyan Premier League |
| season | text | Season identifier, e.g. 2025/2026 |
| home_team | text | Canonical home team name |
| away_team | text | Canonical away team name |
| home_goals | integer | Final home goals |
| away_goals | integer | Final away goals |

## Optional columns
| Column | Type | Meaning |
|---|---|---|
| home_xg / away_xg | decimal | Pre-match history's match xG, recorded after each match for future matches |
| home_absences / away_absences | integer | Confirmed unavailable players known before kick-off |
| home_key_absences / away_key_absences | integer | Count of unavailable players pre-labelled as key players |
| home_rest_days / away_rest_days | decimal | Days since previous competitive match |
| home_travel_km / away_travel_km | decimal | Travel distance before the match |
| home_odds / draw_odds / away_odds | decimal | Pre-match decimal odds; excluded from model training |

## Ten-season coverage
Use one row per completed fixture across the requested competitions:
EPL, La Liga, Bundesliga, Serie A, Ligue 1, English Championship,
Dutch Eredivisie, Portuguese Primeira Liga, Kenyan Premier League, and
additional explicitly named competitions under `Other leagues`.

Do not put several unrelated competitions under one generic `Other leagues`
label if team names overlap. Prefer a unique competition name for each league.

## Data integrity rules
1. Use consistent team names within each competition and season.
2. Do not duplicate fixtures.
3. Exclude abandoned matches unless a consistent final result is recorded.
4. Store postponed matches on their actual played date.
5. Never include post-match facts in pre-match injury, travel, odds or lineup fields.
6. Keep source, retrieval time and prediction timestamp in a separate audit log.
7. xG may be missing for older seasons; leave it blank rather than fabricate it.
