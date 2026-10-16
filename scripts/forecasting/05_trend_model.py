"""Evaluate whether linear trend improves room-night demand forecasts.

This experiment compares a channel-level linear trend against the

recent-mean baseline using rolling-origin backtesting.

"""

import argparse

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(

        description="Backtest a linear-trend room-night demand model."

    )

    parser.add_argument(

        "--input",

        required=True,

        help="Path to the private Limehome Excel workbook.",

    )

    return parser.parse_args()

def run_trend_backtest(

    demand,

    origin,

    history_months=3,

    forecast_months=2,

):

    """Fit and backtest a separate linear trend for each channel."""

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

    # Validate coverage

    # ---------------------------------------------------------

    expected_history_days = (

        history_end - history_start

    ).days + 1

    expected_evaluation_days = (

        evaluation_end - evaluation_start

    ).days + 1

    history_coverage = (

        history.groupby("code")["stay_date"].nunique()

    )

    evaluation_coverage = (

        evaluation.groupby("code")["stay_date"].nunique()

    )

    incomplete_history = history_coverage[

        history_coverage != expected_history_days

    ]

    incomplete_evaluation = evaluation_coverage[

        evaluation_coverage != expected_evaluation_days

    ]

    if not incomplete_history.empty:

        raise ValueError(

            f"Incomplete history before {origin.date()}: "

            f"{incomplete_history.to_dict()}"

        )

    if not incomplete_evaluation.empty:

        raise ValueError(

            f"Incomplete evaluation after {origin.date()}: "

            f"{incomplete_evaluation.to_dict()}"

        )

    # ---------------------------------------------------------

    # Fit one linear trend per channel

    # ---------------------------------------------------------

    scored_parts = []

    model_details = []

    for channel in sorted(history["code"].unique()):

        channel_history = (

            history.loc[history["code"] == channel]

            .sort_values("stay_date")

            .copy()

        )

        channel_evaluation = (

            evaluation.loc[evaluation["code"] == channel]

            .sort_values("stay_date")

            .copy()

        )

        # Historical time index:

        # first history day = 0, second = 1, ...

        x_train = np.arange(

            len(channel_history),

            dtype=float,

        )

        y_train = channel_history[

            "room_nights"

        ].to_numpy(dtype=float)

        # Fit:

        # demand = intercept + slope * time

        slope, intercept = np.polyfit(

            x_train,

            y_train,

            deg=1,

        )

        # Future time continues from the end of history.

        x_future = np.arange(

            len(channel_history),

            len(channel_history)

            + len(channel_evaluation),

            dtype=float,

        )

        predictions = (

            intercept

            + slope * x_future

        )

        channel_evaluation["prediction"] = predictions

        channel_evaluation["absolute_error"] = np.abs(

            channel_evaluation["room_nights"]

            - channel_evaluation["prediction"]

        )

        channel_evaluation["squared_error"] = (

            channel_evaluation["room_nights"]

            - channel_evaluation["prediction"]

        ) ** 2

        scored_parts.append(channel_evaluation)

        model_details.append(

            {

                "code": channel,

                "slope_per_day": slope,

                "slope_per_30_days": slope * 30,

                "first_prediction": predictions[0],

                "last_prediction": predictions[-1],

            }

        )

    scored = pd.concat(

        scored_parts,

        ignore_index=True,

    )

    model_details = pd.DataFrame(model_details)

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

    return results, model_details

def run_recent_mean_backtest(

    demand,

    origin,

    history_months=3,

    forecast_months=2,

):

    """Backtest the recent channel-level mean baseline."""

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

    channel_means = (

        history

        .groupby("code")["room_nights"]

        .mean()

        .rename("prediction")

        .reset_index()

    )

    scored = evaluation.merge(

        channel_means,

        on="code",

        how="left",

        validate="many_to_one",

    )

    scored["absolute_error"] = np.abs(

        scored["room_nights"] - scored["prediction"]

    )

    results = (

        scored

        .groupby("code")

        .agg(

            mean_actual=("room_nights", "mean"),

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

    # 6. Test trend model at one forecast origin

    # ---------------------------------------------------------

    test_origin = pd.Timestamp("2024-04-01")

    trend_result, trend_details = run_trend_backtest(

        demand=daily_demand,

        origin=test_origin,

        history_months=3,

        forecast_months=2,

    )

    print(

        "\nLinear trend learned from "

        "2024-01-01 to 2024-03-31:"

    )

    print(

        trend_details

        .round(2)

        .to_string(index=False)

    )

    print(

        "\nTrend model validation performance:"

    )

    print(

        trend_result[

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

    # 7. Rolling backtest the trend model

    # ---------------------------------------------------------

    forecast_origins = pd.to_datetime(

        [

            "2023-07-01",

            "2023-10-01",

            "2024-01-01",

            "2024-04-01",

        ]

    )

    trend_backtests = []

    trend_detail_backtests = []

    for origin in forecast_origins:

        result, details = run_trend_backtest(

            demand=daily_demand,

            origin=origin,

            history_months=3,

            forecast_months=2,

        )

        details["forecast_origin"] = origin

        trend_backtests.append(result)

        trend_detail_backtests.append(details)

    trend_backtests = pd.concat(

        trend_backtests,

        ignore_index=True,

    )

    trend_detail_backtests = pd.concat(

        trend_detail_backtests,

        ignore_index=True,

    )

    print("\nTrend rolling backtest results:")

    print(

        trend_backtests[

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

    print("\nTrend slopes across forecast origins:")

    print(

        trend_detail_backtests[

            [

                "forecast_origin",

                "code",

                "slope_per_30_days",

                "first_prediction",

                "last_prediction",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 8. Compare trend against recent-mean baseline

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

    comparison = baseline_backtests[

        ["forecast_origin", "code", "mae"]

    ].merge(

        trend_backtests[

            ["forecast_origin", "code", "mae"]

        ],

        on=["forecast_origin", "code"],

        suffixes=("_baseline", "_trend"),

    )

    comparison["mae_change"] = (

        comparison["mae_trend"]

        - comparison["mae_baseline"]

    )

    comparison["trend_wins"] = (

        comparison["mae_trend"]

        < comparison["mae_baseline"]

    )

    print("\nTrend vs recent-mean baseline:")

    print(

        comparison

        .round(2)

        .to_string(index=False)

    )

    summary = (

        comparison

        .groupby("code")

        .agg(

            baseline_mean_mae=("mae_baseline", "mean"),

            trend_mean_mae=("mae_trend", "mean"),

            mean_mae_change=("mae_change", "mean"),

            trend_wins=("trend_wins", "sum"),

        )

        .reset_index()

    )

    summary["relative_change_pct"] = (

        100

        * summary["mean_mae_change"]

        / summary["baseline_mean_mae"]

    )

    print("\nTrend comparison summary:")

    print(

        summary

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 9. Visualize baseline vs trend

    # ---------------------------------------------------------

    output_path = Path(

        "images/forecasting/07_trend_model_comparison.png"

    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    x = np.arange(len(summary))

    width = 0.35

    fig, ax = plt.subplots(

        figsize=(10, 6)

    )

    ax.bar(

        x - width / 2,

        summary["baseline_mean_mae"],

        width,

        label="Recent mean",

    )

    ax.bar(

        x + width / 2,

        summary["trend_mean_mae"],

        width,

        label="Linear trend",

    )

    ax.set_xticks(x)

    ax.set_xticklabels(summary["code"])

    ax.set_ylabel(

        "Mean Backtest MAE (room nights)"

    )

    ax.set_title(

        "Does Linear Trend Improve Forecasting?"

    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(

        output_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close(fig)

    print(

        f"\nSaved comparison chart: {output_path}"

    )

if __name__ == "__main__":

    main()
