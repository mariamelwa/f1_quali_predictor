## 🏎️ F1 Qualifying Pace Predictor (Q1–Q3 best lap)

F1 Qualifying Pace Predictor using FastF1 and scikit-learn. Fetches multi-season qualifying data, trains a Ridge Regression model on drivers, teams, and circuits, and predicts best lap times for future GPs (e.g., São Paulo 2025). Data-driven insights into qualifying performance.



Predicts each driver’s best qualifying lap (minimum of Q1/Q2/Q3) using multi-season data from FastF1 and a Ridge Regression model (scikit-learn). Example included for São Paulo (Interlagos) 2025.

## Features

• Auto-fetch official qualifying data via FastF1

• Build multi-year dataset (default: 2022–2024)

• One-hot encode Driver, Team, Circuit (+ numeric Year trend)

• Train Ridge model

• Predict + rank best quali lap for a provided lineup

## Requirements

• Python 3.9+

### Packages:

```bash
pip install fastf1 scikit-learn pandas numpy matplotlib
```

## Quick start

### 1. Clone the repo and enter the folder:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create a venv:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```


### 3. Install dependencies:
```bash
pip install fastf1 scikit-learn pandas numpy matplotlib
```


### 4.Run the script:
```bash
python f1_quali_predictor.py
```


On first run, FastF1 will cache data under cache/ and may take a few minutes.

## Output

• Logs for dataset building & validation metrics

• A ranked table of predicted best qualifying laps for the sample São Paulo 2025 lineup


## How it works (brief)

#### • build_training_set(years=(2022, 2023, 2024))
Iterates the event schedule, fetches qualifying results per round, keeps “Grand Prix” events only.

#### • fetch_quali_results(year, gp_id)
Pulls Q1/Q2/Q3 from session.results, converts to seconds, and computes best_sec = min(Q1, Q2, Q3).

 #### •fit_model(df)
          • ColumnTransformer:
          
          • One-hot: Team, Driver, CircuitShortName
          
          • Pass-through: Year
          Then fits a Ridge model and prints hold-out metrics.

#### predict_interlagos(model, lineup, year=2025)
Builds an inference frame for Sao Paulo Grand Prix and returns ranked predictions.

## Customize
### Train on different seasons

Edit the call in __main__:
```bash
data = build_training_set(years=(2021, 2022, 2023, 2024))
```

### Predict for a different circuit

The example targets Sao Paulo Grand Prix. To predict another race:

Find its ### EventName as it appears in FastF1 schedules (e.g., “Bahrain Grand Prix”, “Japanese Grand Prix”).

### Change the lineup

• Edit the ### sao_paulo_2025 dict in __main__ with:
```bash
 "Driver Name": "Team Name" pairs.
```

• Make sure names match FastF1 conventions (e.g., “Red Bull Racing”, “Ferrari”, “McLaren”, …).
