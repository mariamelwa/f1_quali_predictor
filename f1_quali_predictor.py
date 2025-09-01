# pip install fastf1 scikit-learn pandas numpy matplotlib
import warnings
warnings.filterwarnings("ignore")

import fastf1
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# FastF1 setup
fastf1.Cache.enable_cache("cache")  


# Data fetching and assembly
def fetch_quali_results(year: int, gp_id) -> pd.DataFrame:
    """
    Fetch qualifying results for a GP.
    gp_id can be a round number (int) or the event name ('Sao Paulo',...).
    Returns a tidy frame with best quali time in seconds and useful meta.
    """
    session = fastf1.get_session(year, gp_id, "Q")
    session.load()  # pulls timing from official F1 timing feed

    # Select & rename
    df = session.results[["DriverNumber", "FullName", "TeamName", "Q1", "Q2", "Q3"]].copy()
    df = df.rename(columns={"FullName": "Driver", "TeamName": "Team"})

    # Convert times to seconds
    for col in ["Q1", "Q2", "Q3"]:
        df[col] = df[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else np.nan)

    # Take each driver's best quali lap across segments
    df["best_sec"] = df[["Q1", "Q2", "Q3"]].min(axis=1)

    # Meta fields
    df["Year"] = year
    df["Round"] = int(session.event["RoundNumber"])
    df["EventName"] = session.event["EventName"]    
    df["CircuitShortName"] = session.event["EventName"] 

    # Keep only rows that have any valid quali time
    df = df.dropna(subset=["best_sec"])
    return df[["DriverNumber", "Driver", "Team", "best_sec", "Year", "Round", "EventName", "CircuitShortName"]]


def build_training_set(years=(2022, 2023, 2024)) -> pd.DataFrame:
    """
    Build a multi-year dataset from *all* events in the chosen seasons.
    Usession event schedules to iterate every round that has Qualifying.
    """
    all_parts = []
    for yr in years:
        schedule = fastf1.get_event_schedule(yr)
        # Keep F1 championship events 
        schedule = schedule[schedule["EventName"].str.contains("Grand Prix", na=False)]
        for _, row in schedule.iterrows():
            rnd = int(row["RoundNumber"])
            try:
                part = fetch_quali_results(yr, rnd)
                all_parts.append(part)
            except Exception as e:
                print(f"[warn] {yr} round {rnd}: {e}")

    if not all_parts:
        raise RuntimeError("No qualifying data fetched. Check cache/network and seasons.")
    return pd.concat(all_parts, ignore_index=True)


# Modeling

def fit_model(df: pd.DataFrame):
    """
    Learn additive effects for Team, Driver, and Circuit using a regularized linear model.
    Target: best_sec (lower is faster).
    Features: Team, Driver, CircuitShortName (+ a light numerical Year trend).
    """
    work = df.dropna(subset=["best_sec"]).copy()

    # Features / target
    X = work[["Team", "Driver", "CircuitShortName", "Year"]]
    y = work["best_sec"].astype(float)

    # One-hot for categorical columns; pass-through Year
    cat_cols = ["Team", "Driver", "CircuitShortName"]
    num_cols = ["Year"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Ridge helps stabilize per-driver/team coefficients
    model = Ridge(alpha=3.0, random_state=42)

    pipe = Pipeline(steps=[("prep", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipe.fit(X_train, y_train)

    # quick eval
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n=== Validation (hold-out 20%) ===")
    print(f"MAE: {mae:.3f} s   |   R²: {r2:.3f}")

    return pipe



# Inference for Interlagos (São Paulo)

def predict_interlagos(pipe, lineup: dict, year: int = 2025) -> pd.DataFrame:
    """
    Predict Q pace for a given Interlagos lineup.
    lineup: dict like {"Driver Name": "Team Name", ...}
    """
    # São Paulo’s official event name matches schedule’s EventName we trained on
    interlagos_event = "Sao Paulo Grand Prix"

    # Build inference frame with same columns the pipeline expects
    inf = (
        pd.DataFrame(
            [(drv, team, interlagos_event, year) for drv, team in lineup.items()],
            columns=["Driver", "Team", "CircuitShortName", "Year"],
        )
        .sort_values(["Team", "Driver"])
        .reset_index(drop=True)
    )

    # Predict best quali seconds
    inf["Predicted_best_sec"] = pipe.predict(inf[["Team", "Driver", "CircuitShortName", "Year"]])
    # Rank by predicted best lap
    inf = inf.sort_values("Predicted_best_sec").reset_index(drop=True)
    inf.index = inf.index + 1
    return inf[["Driver", "Team", "Predicted_best_sec"]]




# Example main: train on 2022–2024 data, then prediction for São Paulo 2025

if __name__ == "__main__":
    print("Building training dataset (this may take a few minutes the first time)...")
    data = build_training_set(years=(2022, 2023, 2024))
    print(f"Rows: {len(data)}  |  Drivers: {data['Driver'].nunique()}  |  Teams: {data['Team'].nunique()}  |  Tracks: {data['CircuitShortName'].nunique()}")

    model = fit_model(data)

    sao_paulo_2025 = {
        "Max Verstappen": "Red Bull Racing",
        "Yuki Tsunoda": "Red Bull Racing",
        "Charles Leclerc": "Ferrari",
        "Lewis Hamilton": "Ferrari",
        "Lando Norris": "McLaren",
        "Oscar Piastri": "McLaren",
        "George Russell": "Mercedes",
        "Kimi Antonelli": "Mercedes",
        "Fernando Alonso": "Aston Martin",
        "Lance Stroll": "Aston Martin",
        "Carlos Sainz": "Williams",
        "Alexander Albon": "Williams",
        "Nico Hulkenberg": "Kick Sauber",
        "Gabriel Bortoleto": "Kick Sauber",
        "Liam Lawson": "Racing Bulls",
        "Isack Hadjar": "Racing Bulls",
        "Pierre Gasly": "Alpine",
        "Franco Colapinto": "Alpine",
        "Esteban Ocon": "Haas",
        "Oliver Bearman": "Haas",
    }

    pred = predict_interlagos(model, sao_paulo_2025, year=2025)
    print("\n=== São Paulo 2025 – Predicted Qualifying (best lap) ===")
    print(pred.to_string())
