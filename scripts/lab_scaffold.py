"""
================================================================================
COMPUTATIONAL OPTIMIZATION — LAB SESSION
Italian Electricity Market: Price Forecasting and Spread Classification

TEAM NAME: XXXXXXXXXXXXXXXXXXXXXX

TEMA MEMBERS:
    NAME 1, SURNAME 1
    NAME 2, SURNAME 2

================================================================================

INSTRUCTIONS
------------
This file is your submission template. You must implement exactly two functions:

    fit_mia1_model(slot_data)        ->  list[RegressionPredictor]
    fit_spread_classifier(slot_data) ->  list[ClassificationPredictor]

Rules:
  - Do NOT modify any function outside the two marked sections.
  - Do NOT rename, remove, or change the signature of any existing function.
  - You may add private helper functions anywhere in this file.
  - Allowed imports: numpy, pandas, gurobipy, Python standard library only.
  - Both models MUST be solved with Gurobi. Other solvers are not accepted.
  - Run the smoke test (python lab_scaffold.py) before submitting.

INPUT FILE
----------
You are provided with a CSV file:  prices_nord_sard.csv

It is produced from the official real price files and has the following structure:

    datetime     — timestamp, 15-min frequency, format YYYY-MM-DD HH:MM:SS
                   Periodo 1 -> 00:00:00, Periodo 96 -> 23:45:00
    mgp_NORD     — MGP  price, NORD zone  [EUR/MWh]
    mia1_NORD    — MI-A1 price, NORD zone [EUR/MWh]
    mia2_NORD    — MI-A2 price, NORD zone [EUR/MWh]
    mgp_SARD     — MGP  price, SARD zone  [EUR/MWh]
    mia1_SARD    — MI-A1 price, SARD zone [EUR/MWh]
    mia2_SARD    — MI-A2 price, SARD zone [EUR/MWh]

The file covers approximately six months (~180 days x 96 slots = ~17 280 rows).
Use load_slot_data() below to convert it into the per-slot format expected by
the two exercise functions.

INPUT DATA FORMAT (per-slot DataFrames)
----------------------------------------
load_slot_data(csv_path, zone, slots) returns a list of DataFrames, one per
requested slot index. Each DataFrame has one row per day and columns:

    date  (datetime.date)  — calendar date, sorted chronologically
    mgp   (float)          — MGP price for this slot on this date  [EUR/MWh]
    mia1  (float)          — MI-A1 price                           [EUR/MWh]
    mia2  (float)          — MI-A2 price                           [EUR/MWh]

Slot index = Periodo - 1  (slot 0 = 00:00, slot 47 = 11:45, slot 95 = 23:45).
You may work on NORD, SARD, or both zones — your choice.

WHAT YOU RETURN
---------------
Two lists of three predictor objects (one per slot), each satisfying the
protocols below. The evaluation infrastructure will call:

    Exercise 1:  predictor.predict(window)  -> float
    Exercise 2:  predictor.predict(window)  -> float  (+1.0 or -1.0)

Both predictors receive the same 22-row window DataFrame in the exact
format of prices_nord_sard.csv (columns: datetime, mgp_NORD, mia1_NORD,
mia2_NORD, mgp_SARD, mia1_SARD, mia2_SARD):

    Rows 0-20  : the 21 most recent completed days (all columns filled).
    Row 21     : the target day — mgp_NORD and mgp_SARD are filled
                 (MGP has cleared); mia1_* and mia2_* are NaN because
                 neither MI-A1 nor MI-A2 has cleared yet.

For Exercise 1, the predictor returns its MI-A1 price forecast.
For Exercise 2, the predictor returns the predicted sign of the spread
mia2 - mia1 (+1.0 or -1.0), which must be predicted before MI-A1 clears.
Both exercises use the same window structure; the predictor extracts
whichever features it needs from the 21 history rows and mgp of row 21.

INFORMATION AVAILABILITY AT PREDICTION TIME
--------------------------------------------
  MGP clears the day before delivery      -> always available (row 21, mgp_*)
  MI-A1 clears on the morning of delivery -> NOT available (row 21, mia1_* = NaN)
  MI-A2 clears later in the delivery day  -> NOT available (row 21, mia2_* = NaN)

EVALUATION
----------
Your predictors are called on several hidden future days (same three slots,
same zone). For each hidden day d the evaluator builds a fresh 22-row window
from the preceding 21 days and the MGP of day d, and calls predict(window).

  Exercise 1 — MI-A1 price prediction
      Metric 1 : Mean Absolute Error  (MAE)
      Metric 2 : Mean Absolute Percentage Error  (MAPE)
      Baselines : (a) median of last 7 days' MI-A1 prices visible in the window
                  (b) mean   of last 7 days' MI-A1 prices visible in the window

    * For MAE, read here: https://en.wikipedia.org/wiki/Mean_absolute_error
    * For MAPE, read here: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    
  Exercise 2 — Spread sign classification  (sign of mia2 - mia1)
      Predicted from the same window as Exercise 1 (mia1/mia2 of day d are NaN).
      Metric 1 : F1 score  (macro-averaged over {-1, +1})
      Metric 2 : Accuracy
      Metric 3 : AUC  (area under ROC curve, using the raw decision score)
      Baseline : sign of the spread on the previous day (persistence)

    * For F1-score: https://en.wikipedia.org/wiki/F-score
    * For Accuracy: https://en.wikipedia.org/wiki/Accuracy_and_precision
    * For the ROC AUC metric read this page: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from typing import Protocol


# ---------------------------------------------------------------------------
# Data loader  —  DO NOT MODIFY
# ---------------------------------------------------------------------------

def load_slot_data(
    csv_path: str,
    zone: str,
    slots: list[int],
) -> list[pd.DataFrame]:
    """
    Load prices_nord_sard.csv and return one per-slot DataFrame per slot index.

    Parameters
    ----------
    csv_path : path to prices_nord_sard.csv
    zone     : "NORD" or "SARD"
    slots    : list of slot indices to extract  (slot = Periodo - 1, range 0-95)

    Returns
    -------
    List of DataFrames, one per slot, each with columns:
        date  (datetime.date)
        mgp   (float)  [EUR/MWh]
        mia1  (float)  [EUR/MWh]
        mia2  (float)  [EUR/MWh]
    Rows are one per trading day, sorted chronologically.

    Example
    -------
    >>> slot_data = load_slot_data("prices_nord_sard.csv", "NORD", [19, 47, 71])
    >>> slot_data[0]   # slot 19 = 04:45
    """
    if zone not in ("NORD", "SARD"):
        raise ValueError(f"zone must be 'NORD' or 'SARD', got {zone!r}")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # derive date and slot index (slot = Periodo - 1)
    df["date"] = df["datetime"].dt.normalize().dt.date
    df["slot"] = (
        df["datetime"].dt.hour * 4 + df["datetime"].dt.minute // 15
    )

    result = []
    for slot in slots:
        if not (0 <= slot <= 95):
            raise ValueError(f"slot must be in 0..95, got {slot}")
        sub = (
            df[df["slot"] == slot]
            [[  "date",
                f"mgp_{zone}",
                f"mia1_{zone}",
                f"mia2_{zone}",
            ]]
            .rename(columns={
                f"mgp_{zone}":  "mgp",
                f"mia1_{zone}": "mia1",
                f"mia2_{zone}": "mia2",
            })
            .reset_index(drop=True)
        )
        result.append(sub)

    return result


# ---------------------------------------------------------------------------
# Predictor protocols  —  DO NOT MODIFY
# ---------------------------------------------------------------------------

class RegressionPredictor(Protocol):
    def predict(self, window: pd.DataFrame) -> float:
        """
        Predict the MI-A1 price for a new day.

        Parameters
        ----------
        window : pd.DataFrame with exactly 22 rows and the same columns as
                 prices_nord_sard.csv:
                     datetime, mgp_NORD, mia1_NORD, mia2_NORD,
                               mgp_SARD, mia1_SARD, mia2_SARD

                 Rows 0-20  : the 21 most recent completed days (all columns filled).
                 Row 21     : the target day — mgp_NORD and mgp_SARD are filled
                              (MGP has cleared); mia1_* and mia2_* are NaN.

                 Rows are sorted chronologically; row 21 is the day to predict.

        Returns
        -------
        Predicted MI-A1 price [EUR/MWh] for the slot and zone you trained on.
        """
        ...


class ClassificationPredictor(Protocol):
    def predict(self, window: pd.DataFrame) -> float:
        """
        Predict the sign of the spread  mia2 - mia1  for a new day.

        Parameters
        ----------
        window : pd.DataFrame with exactly 22 rows and the same columns as
                 prices_nord_sard.csv:
                     datetime, mgp_NORD, mia1_NORD, mia2_NORD,
                               mgp_SARD, mia1_SARD, mia2_SARD

                 Rows 0-20  : the 21 most recent completed days (all columns filled).
                 Row 21     : the target day — mgp_* are filled; mia1_* and
                              mia2_* are NaN (neither MI-A1 nor MI-A2 has cleared).

                 Rows are sorted chronologically; row 21 is the day to predict.

        Returns
        -------
        +1.0  if predicted spread  mia2 - mia1  > 0
        -1.0  if predicted spread  mia2 - mia1  < 0
        """
        ...

    def decision_value(self, window: pd.DataFrame) -> float:
        """
        Return the raw signed score (before thresholding at zero).
        Used to compute AUC. Larger values indicate stronger belief in +1.
        Receives the same window as predict().
        """
        ...


# ---------------------------------------------------------------------------
# Baseline predictors  —  DO NOT MODIFY
# ---------------------------------------------------------------------------

class MedianBaseline:
    """
    Predicts the median MI-A1 price over the last 7 days visible in the window.
    Reads column mia1_{zone} from rows 0..20 (ignores the NaN target row).
    """

    def __init__(self, zone: str):
        self._col = f"mia1_{zone}"

    def predict(self, window: pd.DataFrame) -> float:
        history = window[self._col].dropna().values
        return float(np.median(history[-7:]))


class MeanBaseline:
    """
    Predicts the mean MI-A1 price over the last 7 days visible in the window.
    Reads column mia1_{zone} from rows 0..20 (ignores the NaN target row).
    """

    def __init__(self, zone: str):
        self._col = f"mia1_{zone}"

    def predict(self, window: pd.DataFrame) -> float:
        history = window[self._col].dropna().values
        return float(np.mean(history[-7:]))


class PreviousDaySignBaseline:
    """
    Persistence baseline: reads the spread sign from row 20 of the window
    (the most recent completed day) and returns it as the prediction.
    """

    def _sign_from_window(self, window: pd.DataFrame) -> float:
        mia1 = window["mia1_NORD"].iloc[20]
        mia2 = window["mia2_NORD"].iloc[20]
        s = float(np.sign(mia2 - mia1))
        return s if s != 0.0 else 1.0

    def predict(self, window: pd.DataFrame) -> float:
        return self._sign_from_window(window)

    def decision_value(self, window: pd.DataFrame) -> float:
        return self._sign_from_window(window)


# ---------------------------------------------------------------------------
# Evaluation utilities  —  DO NOT MODIFY
# ---------------------------------------------------------------------------

def _build_window(
    full_df: pd.DataFrame,
    test_idx: int,
    slot: int,
    zone: str,
    n_history: int = 21,
) -> pd.DataFrame:
    """
    Build the 22-row window DataFrame passed to both predictor.predict() calls.

    Rows 0..n_history-1 : last n_history completed days before test_idx
                          (all 7 columns filled).
    Row n_history        : the target day — mgp_* filled, mia1_* and mia2_* NaN.

    full_df must be the raw prices_nord_sard.csv DataFrame (all 7 columns),
    already filtered to the relevant slot and sorted chronologically.
    """
    history = full_df.iloc[test_idx - n_history : test_idx].copy()
    target  = full_df.iloc[[test_idx]].copy()
    for col in full_df.columns:
        if col.startswith("mia1_") or col.startswith("mia2_"):
            target[col] = float("nan")
    return pd.concat([history, target], ignore_index=True)


def evaluate_regression(
    predictor,
    full_df:   pd.DataFrame,
    slot:      int,
    zone:      str,
    test_indices: list[int],
    true_mia1: np.ndarray,
) -> dict:
    """
    Evaluate a regression predictor on a sequence of test days.

    Parameters
    ----------
    predictor    : object with .predict(window: pd.DataFrame) -> float
    full_df      : raw prices_nord_sard.csv DataFrame (all columns), slot-filtered
    slot         : slot index (used only for labelling)
    zone         : "NORD" or "SARD"
    test_indices : row indices in full_df of the test days
    true_mia1    : true MI-A1 prices for the test days, shape (n_test,)

    Returns
    -------
    dict with keys: mae, mape
    """
    preds = np.array([
        predictor.predict(_build_window(full_df, idx, slot, zone))
        for idx in test_indices
    ])
    mae  = float(np.mean(np.abs(preds - true_mia1)))
    mape = float(np.mean(np.abs((preds - true_mia1) / true_mia1))) * 100
    return {"mae": mae, "mape": mape}


def evaluate_classifier(
    predictor,
    full_df:      pd.DataFrame,
    slot:         int,
    zone:         str,
    test_indices: list[int],
    true_labels:  np.ndarray,
) -> dict:
    """
    Evaluate a classification predictor on a sequence of test days.

    Parameters
    ----------
    predictor    : object with .predict(window) and .decision_value(window)
    full_df      : raw prices_nord_sard.csv DataFrame (all columns), slot-filtered
    slot         : slot index
    zone         : "NORD" or "SARD"
    test_indices : row indices in full_df of the test days
    true_labels  : true spread signs in {-1, +1}, shape (n_test,)

    Returns
    -------
    dict with keys: f1_macro, accuracy, auc
    """
    windows = [_build_window(full_df, idx, slot, zone) for idx in test_indices]
    preds   = np.array([predictor.predict(w)        for w in windows])
    scores  = np.array([predictor.decision_value(w) for w in windows])

    accuracy = float(np.mean(preds == true_labels))

    f1s = []
    for label in (-1.0, 1.0):
        tp = float(np.sum((preds == label) & (true_labels == label)))
        fp = float(np.sum((preds == label) & (true_labels != label)))
        fn = float(np.sum((preds != label) & (true_labels == label)))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    f1_macro = float(np.mean(f1s))

    pos_scores = scores[true_labels ==  1]
    neg_scores = scores[true_labels == -1]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        auc = float("nan")
    else:
        auc = float(np.mean(pos_scores[:, None] > neg_scores[None, :])
                    + 0.5 * float(np.mean(pos_scores[:, None] == neg_scores[None, :])))

    return {"f1_macro": f1_macro, "accuracy": accuracy, "auc": auc}


def compare_regression(
    student_predictor,
    full_df:       pd.DataFrame,
    slot:          int,
    zone:          str,
    test_indices:  list[int],
    true_mia1:     np.ndarray,
) -> None:
    """Print a comparison table against the two regression baselines."""
    student  = evaluate_regression(student_predictor,    full_df, slot, zone, test_indices, true_mia1)
    median_b = evaluate_regression(MedianBaseline(zone), full_df, slot, zone, test_indices, true_mia1)
    mean_b   = evaluate_regression(MeanBaseline(zone),   full_df, slot, zone, test_indices, true_mia1)

    print(f"  {'Model':<22} {'MAE':>8} {'MAPE':>8}")
    print(f"  {'-'*40}")
    print(f"  {'Your model':<22} {student['mae']:>8.3f} {student['mape']:>7.2f}%")
    print(f"  {'Baseline: median(7d)':<22} {median_b['mae']:>8.3f}  {median_b['mape']:>6.2f}%")
    print(f"  {'Baseline: mean(7d)':<22}  {mean_b['mae']:>8.3f}  {mean_b['mape']:>6.2f}%")


def compare_classifier(
    student_predictor,
    full_df:      pd.DataFrame,
    slot:         int,
    zone:         str,
    test_indices: list[int],
    true_labels:  np.ndarray,
) -> None:
    """Print a comparison table against the persistence baseline."""
    student  = evaluate_classifier(student_predictor,        full_df, slot, zone, test_indices, true_labels)
    baseline = evaluate_classifier(PreviousDaySignBaseline(), full_df, slot, zone, test_indices, true_labels)

    print(f"  {'Model':<28} {'F1':>7} {'Acc':>7} {'AUC':>7}")
    print(f"  {'-'*52}")
    print(f"  {'Your model':<28} "
          f"{student['f1_macro']:>7.3f} {student['accuracy']:>7.3f} {student['auc']:>7.3f}")
    print(f"  {'Baseline: persistence':<28} "
          f"{baseline['f1_macro']:>7.3f} {baseline['accuracy']:>7.3f} {baseline['auc']:>7.3f}")


# ---------------------------------------------------------------------------
# Exercise 1  —  EXAMPLE BASELINE SOLUTION
# ---------------------------------------------------------------------------
#
# The predictor receives a 22-row window DataFrame (same columns as
# prices_nord_sard.csv). Rows 0-20 are the 21 most recent completed days;
# row 21 is the target day with mia1_* and mia2_* set to NaN.
#
# This baseline solution ignores the Gurobi training step entirely and simply
# returns the mean of the last 7 mia1 values visible in the window — matching
# the MeanBaseline. It is the simplest valid implementation of the interface.
#
# Note: fit_mia1_model still receives slot_data and may run a Gurobi model
# to learn parameters offline. Those parameters can be stored inside the
# predictor object and used at predict() time alongside the window.
# ---------------------------------------------------------------------------

class MeanWindowPredictor:
    """
    Returns the mean of the last 7 mia1 values visible in the window.
    Reads column mia1_{zone} from the history rows (NaN target row excluded).
    """

    def __init__(self, zone: str):
        # TODO: You store the data you have "learnt" for your prediction
        #       And eventually the name of the columns you have to use for the input data
        self._col = f"mia1_{zone}"

    def predict(self, window: pd.DataFrame) -> float:
        # TODO: Here you really use your model for the prediction
        history = window[self._col].dropna().values
        return float(np.mean(history[-7:]))


def fit_mia1_model(
    slot_data: list[pd.DataFrame],
    zone: str = "NORD",
) -> list:
    """
    EXAMPLE BASELINE SOLUTION — mean of last 7 days (no Gurobi training).

    fit_mia1_model receives slot_data (list of per-slot DataFrames) which
    you may use to train a Gurobi model and store learned parameters inside
    your predictor. This baseline ignores the training data and returns a
    predictor that reads directly from the window at prediction time.

    Parameters
    ----------
    slot_data : list of 3 DataFrames (one per slot), columns:
                    date, mgp, mia1, mia2  —  sorted chronologically.
    zone      : "NORD" or "SARD" — which zone column to read in predict().

    Returns
    -------
    List of 3 predictor objects, one per slot. Each must implement:
        .predict(window: pd.DataFrame) -> float
    where window has 22 rows and the same 7 columns as prices_nord_sard.csv.
    Row 21 (the target day) has mia1_* and mia2_* set to NaN.
    """
    predictors = []

    for slot_df in slot_data:
        
        # TODO: here you write your model that fit the training data

        # TODO: later you build the predictor using the solution of your MILP model
        #       if you run a linear prediction, you have to pass the coefficient found
        #       by the optimization model

        # TODO: add your predictior
        predictors.append(MeanWindowPredictor(zone))      

    return predictors


# ---------------------------------------------------------------------------
# Exercise 2  —  EXAMPLE BASELINE SOLUTION
# ---------------------------------------------------------------------------
#
# The predictor receives the same 22-row window as Exercise 1. Rows 0-20
# are fully populated history; row 21 is the target day with mgp_* filled
# and mia1_*/mia2_* as NaN (neither market has cleared yet).
#
# This baseline uses persistence: it reads the spread sign from row 20
# (yesterday) and returns it unchanged. This is the floor to beat.
# ---------------------------------------------------------------------------

class PersistencePredictor:
    """
    Reads the spread sign from the last history row (row 20) of the window
    and returns it as the prediction. Ignores mgp of the target day.
    """

    def _sign_from_window(self, window: pd.DataFrame) -> float:
        # TODO: You store the data you have "learnt" for your prediction
        #       And eventually the name of the columns you have to use for the input data

        mia1 = window["mia1_NORD"].iloc[20]
        mia2 = window["mia2_NORD"].iloc[20]
        s = float(np.sign(mia2 - mia1))
        return s if s != 0.0 else 1.0

    def predict(self, window: pd.DataFrame) -> float:
        # TODO: Here you really use your model for the prediction

        return self._sign_from_window(window)

    def decision_value(self, window: pd.DataFrame) -> float:
        # TODO: Here you really use your model for the prediction

        # Returns the sign itself as the score (+1 or -1).
        # A richer model would return a real-valued confidence here.
        return self._sign_from_window(window)


def fit_spread_classifier(
    slot_data: list[pd.DataFrame],
    zone: str = "NORD",
) -> list:
    """
    EXAMPLE BASELINE SOLUTION — persistence predictor.

    Ignores slot_data entirely (no training step). At prediction time,
    reads the spread sign from the most recent history row of the window.

    Parameters
    ----------
    slot_data : list of 3 DataFrames (one per slot), columns:
                    date, mgp, mia1, mia2  —  sorted chronologically.
                May be used to train a Gurobi model and store parameters
                inside the predictor.
    zone      : "NORD" or "SARD" — which zone columns to use in predict().

    Returns
    -------
    List of 3 predictor objects, one per slot. Each must implement:
        .predict(window: pd.DataFrame)        -> +1.0 or -1.0
        .decision_value(window: pd.DataFrame) -> float

    The window has 22 rows and the same 7 columns as prices_nord_sard.csv.
    Rows 0-20 are fully populated history. Row 21 is the target day:
    mgp_* are filled; mia1_* and mia2_* are NaN.
    """

    predictors = []

    for slot_df in slot_data:

        # TODO: here you write your model that fit the training data

        # TODO: later you build the predictor using the solution of your MILP model
        #       you have to pass the coefficient found by the optimization model

        predictors.append(PersistencePredictor())

    return predictors


# ---------------------------------------------------------------------------
# Smoke test  —  run before submitting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    CSV_PATH   = "prices_nord_sard.csv"
    ZONE       = "NORD"          # change to "SARD" if preferred
    SLOTS      = [19, 47, 71]    # slot 19 = 04:45, slot 47 = 11:45, slot 71 = 17:45
    N_TEST     = 14              # last 14 days held out as test
    N_HISTORY  = 21              # window size passed to predict()

    ALL_COLS = [
        "datetime",
        "mgp_NORD", "mia1_NORD", "mia2_NORD",
        "mgp_SARD", "mia1_SARD", "mia2_SARD",
    ]

    # --- load real data ---
    print(f"Loading {CSV_PATH}  (zone={ZONE}, slots={SLOTS}) ...\n")
    raw_full  = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
    raw_full  = raw_full.sort_values("datetime").reset_index(drop=True)
    raw_full["_slot"] = (
        raw_full["datetime"].dt.hour * 4
        + raw_full["datetime"].dt.minute // 15
    )
    slot_data_full = load_slot_data(CSV_PATH, ZONE, SLOTS)
    # per-slot raw frames (7 columns) for window construction
    raw_slot_frames = [
        raw_full[raw_full["_slot"] == s][ALL_COLS].reset_index(drop=True)
        for s in SLOTS
    ]

    # temporal split
    n_total = len(slot_data_full[0])
    n_train = n_total - N_TEST
    slot_data_train = [df.iloc[:n_train].reset_index(drop=True)
                       for df in slot_data_full]

    print(f"Training days : {n_train}")
    print(f"Test days     : {N_TEST}\n")

    # test indices: rows n_train .. n_total-1 in the raw slot frame
    # (each requires N_HISTORY rows of history before it)
    test_indices = list(range(n_train, n_total))

    # --- Exercise 1 ---
    print("=" * 56)
    print("Exercise 1 — MI-A1 Price Regression (baseline solution)")
    print("=" * 56)
    reg_predictors = fit_mia1_model(slot_data_train, zone=ZONE)
    for i, slot in enumerate(SLOTS):
        true_mia1 = raw_slot_frames[i][f"mia1_{ZONE}"].values[n_train:]
        print(f"\nSlot {slot} (={slot * 15 // 60:02d}:{slot * 15 % 60:02d}):")
        compare_regression(
            reg_predictors[i],
            raw_slot_frames[i], slot, ZONE, test_indices, true_mia1,
        )

    # --- Exercise 2 ---
    print()
    print("=" * 56)
    print("Exercise 2 — Spread Sign Classification (baseline solution)")
    print("=" * 56)
    cls_predictors = fit_spread_classifier(slot_data_train, zone=ZONE)
    for i, slot in enumerate(SLOTS):
        raw_mia1 = raw_slot_frames[i][f"mia1_{ZONE}"].values
        raw_mia2 = raw_slot_frames[i][f"mia2_{ZONE}"].values
        true_labels = np.sign(raw_mia2[n_train:] - raw_mia1[n_train:]).astype(float)
        true_labels[true_labels == 0] = 1.0
        print(f"\nSlot {slot} (={slot * 15 // 60:02d}:{slot * 15 % 60:02d}):")
        compare_classifier(
            cls_predictors[i],
            raw_slot_frames[i], slot, ZONE, test_indices, true_labels,
        )
