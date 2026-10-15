"""Evaluate a weekday-aware demand forecasting model.

This experiment tests whether recent channel × weekday demand patterns

improve forecasts over a simple recent channel-level mean.

"""

import argparse

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(

        description="Backtest a weekday-aware room-night demand model."

    )

    parser.add_argument(

        "--input",

        required=True,

        help="Path to the private Limehome Excel workbook.",

    )

    return parser.parse_args()

def mae(actual, predicted):

    """Return mean absolute error."""

    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):

    """Return root mean squared error."""

    return np.sqrt(np.mean((actual - predicted) ** 2))

def run_weekday_backtest(

    demand,

    origin,

    history_months=3,

    forecast_months=2,

):

    """Backtest channel x weekday historical means."""

    origin = pd.Timestamp(origin)

    history_start = origin - pd.DateOffset(

        months=history_months

    )

    history_end = origin - pd.Timedelta(days=1)

    evaluation_start = origin

    evaluation_end = (

        origin

        + pd.DateOffset(months=forecast_months)

        - pd.Timedelta(days=1)

    )

    history = demand.loc[

        (demand["stay_date"] >= history_start)

        & (demand["stay_date"] <= history_end)

    ].copy()

    evaluation = demand.loc[

        (demand["stay_date"] >= evaluation_start)

        & (demand["stay_date"] <= evaluation_end)

    ].copy()

    # ---------------------------------------------------------

    # Validate historical coverage

    # ---------------------------------------------------------

    expected_history_days = (

        history_end - history_start

    ).days + 1

    history_coverage = (

        history

        .groupby("code")["stay_date"]

        .nunique()

    )

    incomplete_history = history_coverage[

        history_coverage != expected_history_days

    ]

    if not incomplete_history.empty:

        raise ValueError(

            f"Incomplete history before "

            f"{origin.date()}: expected "

            f"{expected_history_days} days per channel, "

            f"found {incomplete_history.to_dict()}"

        )

    # ---------------------------------------------------------

    # Validate evaluation coverage

    # ---------------------------------------------------------

    expected_evaluation_days = (

        evaluation_end - evaluation_start

    ).days + 1

    evaluation_coverage = (

        evaluation

        .groupby("code")["stay_date"]

        .nunique()

    )

    incomplete_evaluation = evaluation_coverage[

        evaluation_coverage

        != expected_evaluation_days

    ]

    if not incomplete_evaluation.empty:

        raise ValueError(

            f"Incomplete evaluation after "

            f"{origin.date()}: expected "

            f"{expected_evaluation_days} days per channel, "

            f"found {incomplete_evaluation.to_dict()}"

        )

    # ---------------------------------------------------------

    # Learn channel x weekday means from history only

    # ---------------------------------------------------------

    history["weekday"] = (

        history["stay_date"].dt.dayofweek

    )

    evaluation["weekday"] = (

        evaluation["stay_date"].dt.dayofweek

    )

    weekday_means = (

        history

        .groupby(["code", "weekday"])["room_nights"]

        .mean()

        .rename("prediction")

        .reset_index()

    )

    # ---------------------------------------------------------

    # Apply learned weekday means to future dates

    # ---------------------------------------------------------

    scored = evaluation.merge(

        weekday_means,

        on=["code", "weekday"],

        how="left",

        validate="many_to_one",

    )

    if scored["prediction"].isna().any():

        raise ValueError(

            "Some evaluation rows did not receive "

            "a weekday prediction."

        )

    scored["absolute_error"] = np.abs(

        scored["room_nights"]

        - scored["prediction"]

    )

    scored["squared_error"] = (

        scored["room_nights"]

        - scored["prediction"]

    ) ** 2

    results = (

        scored

        .groupby("code")

        .agg(

            mean_actual=("room_nights", "mean"),

            mae=("absolute_error", "mean"),

            mse=("squared_error", "mean"),

        )

        .reset_index()

    )

    results["rmse"] = np.sqrt(results["mse"])

    results = results.drop(columns="mse")

    results["forecast_origin"] = origin

    return results

def run_recent_mean_backtest(

    demand,

    origin,

    history_months=3,

    forecast_months=2,

):

    """Backtest a simple channel-level recent-mean model."""

    origin = pd.Timestamp(origin)

    history_start = origin - pd.DateOffset(

        months=history_months

    )

    history_end = origin - pd.Timedelta(days=1)

    evaluation_start = origin

    evaluation_end = (

        origin

        + pd.DateOffset(months=forecast_months)

        - pd.Timedelta(days=1)

    )

    history = demand.loc[

        (demand["stay_date"] >= history_start)

        & (demand["stay_date"] <= history_end)

    ].copy()

    evaluation = demand.loc[

        (demand["stay_date"] >= evaluation_start)

        & (demand["stay_date"] <= evaluation_end)

    ].copy()

    predictions = (

        history

        .groupby("code")["room_nights"]

        .mean()

        .rename("prediction")

        .reset_index()

    )

    scored = evaluation.merge(

        predictions,

        on="code",

        how="left",

        validate="many_to_one",

    )

    if scored["prediction"].isna().any():

        raise ValueError(

            "Some evaluation rows did not receive "

            "a recent-mean prediction."

        )

    scored["absolute_error"] = np.abs(

        scored["room_nights"]

        - scored["prediction"]

    )

    results = (

        scored

        .groupby("code")

        .agg(

            mae=("absolute_error", "mean"),

        )

        .reset_index()

    )

    results["forecast_origin"] = origin

    return results

def main():

    args = parse_args()

    input_path = Path(args.input).expanduser()

    print(f"Input file: {input_path}")

    # ---------------------------------------------------------

    # 1. Load reservation data

    # ---------------------------------------------------------

    reservations = pd.read_excel(

        input_path,

        sheet_name="reservations",

    )

    reservations["arrival_at"] = pd.to_datetime(

        reservations["arrival_at"]

    )

    reservations["departure_at"] = pd.to_datetime(

        reservations["departure_at"]

    )

    reservations["length_of_stay"] = (

        reservations["departure_at"]

        - reservations["arrival_at"]

    ).dt.days

    # ---------------------------------------------------------

    # 2. Keep realized stays

    # ---------------------------------------------------------

    completed_stays = reservations.loc[

        (reservations["status"] == "CheckedOut")

        & (reservations["length_of_stay"] > 0)

    ].copy()

    # ---------------------------------------------------------

    # 3. Expand reservations into room nights

    # ---------------------------------------------------------

    completed_stays["stay_date"] = completed_stays.apply(

        lambda row: pd.date_range(

            start=row["arrival_at"],

            end=row["departure_at"] - pd.Timedelta(days=1),

            freq="D",

        ),

        axis=1,

    )

    room_nights = completed_stays.explode(

        "stay_date",

        ignore_index=True,

    )

    expected_room_nights = completed_stays[

        "length_of_stay"

    ].sum()

    actual_room_nights = len(room_nights)

    if expected_room_nights != actual_room_nights:

        raise ValueError(

            "Room-night expansion validation failed: "

            f"expected {expected_room_nights:,}, "

            f"found {actual_room_nights:,}."

        )

    # ---------------------------------------------------------

    # 4. Aggregate to daily channel demand

    # ---------------------------------------------------------

    daily_demand = (

        room_nights

        .groupby(["stay_date", "code"])

        .size()

        .rename("room_nights")

        .reset_index()

    )

    # ---------------------------------------------------------

    # 5. Build complete date x channel grid

    # ---------------------------------------------------------

    modelling_start = pd.Timestamp("2022-10-01")

    modelling_end = daily_demand["stay_date"].max()

    channels = sorted(

        daily_demand.loc[

            daily_demand["stay_date"] >= modelling_start,

            "code",

        ].unique()

    )

    full_dates = pd.date_range(

        modelling_start,

        modelling_end,

        freq="D",

    )

    complete_index = pd.MultiIndex.from_product(

        [full_dates, channels],

        names=["stay_date", "code"],

    )

    daily_demand = (

        daily_demand

        .set_index(["stay_date", "code"])

        .reindex(complete_index, fill_value=0)

        .reset_index()

    )

    print("\nPrepared modelling data:")

    print(

        f"Dates: {daily_demand['stay_date'].min().date()} "

        f"to {daily_demand['stay_date'].max().date()}"

    )

    print(f"Channels: {channels}")

    print(f"Rows: {len(daily_demand):,}")

    print(

        f"Room-night expansion valid: "

        f"{expected_room_nights:,} == "

        f"{actual_room_nights:,}"

    )

    # ---------------------------------------------------------

    # 6. Test weekday model on one forecast origin

    # ---------------------------------------------------------

    test_origin = pd.Timestamp("2024-04-01")

    weekday_result = run_weekday_backtest(

        demand=daily_demand,

        origin=test_origin,

        history_months=3,

        forecast_months=2,

    )

    print(

        "\nWeekday-aware model "

        "(3-month history, 2024-04-01 origin):"

    )

    print(

        weekday_result[

            [

                "code",

                "mean_actual",

                "mae",

                "rmse",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 7. Rolling backtest across all forecast origins

    # ---------------------------------------------------------

    forecast_origins = pd.to_datetime(

        [

            "2023-07-01",

            "2023-10-01",

            "2024-01-01",

            "2024-04-01",

        ]

    )

    weekday_backtests = []

    for origin in forecast_origins:

        result = run_weekday_backtest(

            demand=daily_demand,

            origin=origin,

            history_months=3,

            forecast_months=2,

        )

        weekday_backtests.append(result)

    weekday_backtests = pd.concat(

        weekday_backtests,

        ignore_index=True,

    )

    print(

        "\nWeekday-aware rolling backtest results:"

    )

    print(

        weekday_backtests[

            [

                "forecast_origin",

                "code",

                "mean_actual",

                "mae",

                "rmse",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 8. Compare weekday model against recent-mean baseline

    # ---------------------------------------------------------

    baseline_backtests = []

    for origin in forecast_origins:

        result = run_recent_mean_backtest(

            demand=daily_demand,

            origin=origin,

            history_months=3,

            forecast_months=2,

        )

        baseline_backtests.append(result)

    baseline_backtests = pd.concat(

        baseline_backtests,

        ignore_index=True,

    )

    comparison = weekday_backtests.merge(

        baseline_backtests,

        on=["forecast_origin", "code"],

        suffixes=("_weekday", "_baseline"),

        validate="one_to_one",

    )

    comparison["mae_change"] = (

        comparison["mae_weekday"]

        - comparison["mae_baseline"]

    )

    comparison["weekday_wins"] = (

        comparison["mae_change"] < 0

    )

    print("\nWeekday vs recent-mean baseline:")

    print(

        comparison[

            [

                "forecast_origin",

                "code",

                "mae_baseline",

                "mae_weekday",

                "mae_change",

                "weekday_wins",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    comparison_summary = (

        comparison

        .groupby("code")

        .agg(

            baseline_mean_mae=(

                "mae_baseline",

                "mean",

            ),

            weekday_mean_mae=(

                "mae_weekday",

                "mean",

            ),

            mean_mae_change=(

                "mae_change",

                "mean",

            ),

            weekday_wins=(

                "weekday_wins",

                "sum",

            ),

        )

        .reset_index()

    )

    comparison_summary["relative_change_pct"] = (

        comparison_summary["mean_mae_change"]

        / comparison_summary["baseline_mean_mae"]

        * 100

    )

    print("\nAverage performance across forecast origins:")

    print(

        comparison_summary

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 9. Visualize baseline vs weekday model

    # ---------------------------------------------------------

    plot_data = (

        comparison_summary[

            [

                "code",

                "baseline_mean_mae",

                "weekday_mean_mae",

            ]

        ]

        .set_index("code")

        .rename(

            columns={

                "baseline_mean_mae": "Recent mean",

                "weekday_mean_mae": "Weekday-aware",

            }

        )

    )

    ax = plot_data.plot(

        kind="bar",

        figsize=(10, 6),

    )

    ax.set_title(

        "Does Weekday Seasonality Improve Forecasting?"

    )

    ax.set_xlabel("Booking Channel")

    ax.set_ylabel("Mean Backtest MAE (room nights)")

    plt.xticks(rotation=0)

    plt.tight_layout()

    output_path = (

        Path(__file__).resolve().parents[2]

        / "images"

        / "forecasting"

        / "06_weekday_model_comparison.png"

    )

    plt.savefig(

        output_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close()

    print(

        f"\nSaved weekday comparison chart to: "

        f"{output_path}"

    )

if __name__ == "__main__":

    main()
