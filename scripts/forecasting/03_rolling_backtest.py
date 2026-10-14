"""Rolling backtesting for demand forecasting."""

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

import argparse

def parse_args():

    parser = argparse.ArgumentParser(

        description="Run rolling forecasting backtests."

    )

    parser.add_argument(

        "--input",

        type=Path,

        required=True,

        help="Path to the private Excel dataset.",

    )

    return parser.parse_args()

def run_recent_mean_backtest(

    demand,

    origin,

    history_months=3,

    forecast_months=2,

):

    """Run one fixed-origin recent-mean backtest."""

    history_start = (

        origin

        - pd.DateOffset(months=history_months)

    )

    history_end = (

        origin

        - pd.Timedelta(days=1)

    )

    evaluation_start = origin

    evaluation_end = (

        origin

        + pd.DateOffset(months=forecast_months)

        - pd.Timedelta(days=1)

    )

    # Historical information available at forecast time.

    history = demand.loc[

        (

            demand["stay_date"] >= history_start

        )

        & (

            demand["stay_date"] <= history_end

        )

    ].copy()

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

        expected_dates = pd.date_range(

            start=history_start,

            end=history_end,

            freq="D",

        )

        for channel in incomplete_history.index:

            channel_dates = set(

                history.loc[

                    history["code"] == channel,

                    "stay_date",

                ]

            )

            missing_dates = [

                date

                for date in expected_dates

                if date not in channel_dates

            ]

            print(

                f"\nMissing history dates for "

                f"{channel} before {origin.date()}:"

            )

            for date in missing_dates:

                print(date.date())

        raise ValueError(

            f"Incomplete history before {origin.date()}: "

            f"expected {expected_history_days} days per channel, "

            f"found {incomplete_history.to_dict()}"

        )

    predictions = (

        history

        .groupby("code")["room_nights"]

        .mean()

        .reset_index(name="prediction")

    )

    # Future period being simulated.

    evaluation = demand.loc[

        (

            demand["stay_date"] >= evaluation_start

        )

        & (

            demand["stay_date"] <= evaluation_end

        )

    ].copy()

    # Validate that every channel has complete evaluation coverage.

    expected_evaluation_days = (

        evaluation_end - evaluation_start

    ).days + 1

    evaluation_coverage = (

        evaluation

        .groupby("code")["stay_date"]

        .nunique()

    )

    incomplete_evaluation = evaluation_coverage[

        evaluation_coverage != expected_evaluation_days

    ]

    if not incomplete_evaluation.empty:

        raise ValueError(

            f"Incomplete evaluation after {origin.date()}: "

            f"expected {expected_evaluation_days} days per channel, "

            f"found {incomplete_evaluation.to_dict()}"

        )

    evaluation = evaluation.merge(

        predictions,

        on="code",

        how="left",

    )

    evaluation["error"] = (

        evaluation["room_nights"]

        - evaluation["prediction"]

    )

    evaluation["absolute_error"] = (

        evaluation["error"].abs()

    )

    evaluation["squared_error"] = (

        evaluation["error"] ** 2

    )

    metrics = (

        evaluation

        .groupby("code")

        .agg(

            mae=("absolute_error", "mean"),

            mse=("squared_error", "mean"),

            mean_actual=("room_nights", "mean"),

        )

        .reset_index()

    )

    metrics["rmse"] = (

        metrics["mse"] ** 0.5

    )

    metrics["forecast_origin"] = origin

    return metrics

def main():

    args = parse_args()

    # ---------------------------------------------------------

    # Load and transform realized room-night demand

    # ---------------------------------------------------------

    reservations = pd.read_excel(

        args.input,

        sheet_name="reservations",

    )

    for column in [

        "booking_date",

        "arrival_at",

        "departure_at",

        "cancelled_at",

    ]:

        reservations[column] = pd.to_datetime(

            reservations[column],

            errors="coerce",

        )

    reservations["length_of_stay"] = (

        reservations["departure_at"]

        - reservations["arrival_at"]

    ).dt.days

    completed_stays = reservations.loc[

        (reservations["status"] == "CheckedOut")

        & (reservations["length_of_stay"] > 0)

    ].copy()

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

    daily_channel_demand = (

        room_nights

        .groupby(["stay_date", "code"])

        .size()

        .rename("room_nights")

        .reset_index()

    )

        # ---------------------------------------------------------

    # Build a complete date x channel demand grid

    # ---------------------------------------------------------

    modelling_start = pd.Timestamp("2022-10-01")

    modelling_end = daily_channel_demand[

        "stay_date"

    ].max()

    channels = sorted(

        daily_channel_demand.loc[

            daily_channel_demand["stay_date"]

            >= modelling_start,

            "code",

        ].unique()

    )

    full_dates = pd.date_range(

        start=modelling_start,

        end=modelling_end,

        freq="D",

    )

    complete_index = pd.MultiIndex.from_product(

        [full_dates, channels],

        names=["stay_date", "code"],

    )

    daily_channel_demand = (

        daily_channel_demand

        .set_index(["stay_date", "code"])

        .reindex(

            complete_index,

            fill_value=0,

        )

        .reset_index()

    )

    print("\nComplete modelling grid:")

    print(f"Start: {modelling_start.date()}")

    print(f"End:   {modelling_end.date()}")

    print(f"Channels: {channels}")

    print(f"Rows: {len(daily_channel_demand):,}")


    # ---------------------------------------------------------

    # 1. Define historical forecast origins

    # ---------------------------------------------------------

    forecast_origins = pd.to_datetime(

        [

            "2023-07-01",

            "2023-10-01",

            "2024-01-01",

            "2024-04-01",

        ]

    )

    # ---------------------------------------------------------

    # 2. Construct two-month evaluation windows

    # ---------------------------------------------------------

    print("\nRolling backtest windows:\n")

    for origin in forecast_origins:

        evaluation_start = origin

        evaluation_end = (

            origin

            + pd.DateOffset(months=2)

            - pd.Timedelta(days=1)

        )

        evaluation_days = (

            evaluation_end

            - evaluation_start

        ).days + 1

        print(

            f"Forecast origin:   "

            f"{origin.date()}"

        )

        print(

            f"Evaluation start: "

            f"{evaluation_start.date()}"

        )

        print(

            f"Evaluation end:   "

            f"{evaluation_end.date()}"

        )

        print(

            f"Evaluation days:  "

            f"{evaluation_days}"

        )

        print("-" * 40)


    # ---------------------------------------------------------

    # 3. Inspect the first backtest manually

    # ---------------------------------------------------------

    first_origin = pd.Timestamp("2023-07-01")

    history_start = (

        first_origin

        - pd.DateOffset(months=3)

    )

    history_end = (

        first_origin

        - pd.Timedelta(days=1)

    )

    first_history = daily_channel_demand.loc[

        (

            daily_channel_demand["stay_date"]

            >= history_start

        )

        & (

            daily_channel_demand["stay_date"]

            <= history_end

        )

    ].copy()

    print("\nFirst backtest history:")

    print("History start:", history_start.date())

    print("History end:", history_end.date())

    print("\nObserved dates in history:")

    print(

        first_history["stay_date"].min().date(),

        "to",

        first_history["stay_date"].max().date(),

    )

    # ---------------------------------------------------------

    # Check whether every channel has every calendar date

    # ---------------------------------------------------------

    expected_history_days = (

        history_end - history_start

    ).days + 1

    observations_per_channel = (

        first_history

        .groupby("code")["stay_date"]

        .nunique()

        .reset_index(name="observed_days")

    )

    observations_per_channel["expected_days"] = (

        expected_history_days

    )

    observations_per_channel["missing_days"] = (

        observations_per_channel["expected_days"]

        - observations_per_channel["observed_days"]

    )

    print("\nHistory coverage by channel:")

    print(

        observations_per_channel

        .to_string(index=False)

    )

    first_channel_means = (

        first_history

        .groupby("code")["room_nights"]

        .mean()

        .reset_index(name="prediction")

    )

    print("\nFirst backtest channel means:")

    print(

        first_channel_means

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 4. Evaluate the first historical backtest

    # ---------------------------------------------------------

    evaluation_start = first_origin

    evaluation_end = (

        first_origin

        + pd.DateOffset(months=2)

        - pd.Timedelta(days=1)

    )

    first_evaluation = daily_channel_demand.loc[

        (

            daily_channel_demand["stay_date"]

            >= evaluation_start

        )

        & (

            daily_channel_demand["stay_date"]

            <= evaluation_end

        )

    ].copy()

    # Check evaluation coverage before scoring.

    expected_evaluation_days = (

        evaluation_end - evaluation_start

    ).days + 1

    evaluation_coverage = (

        first_evaluation

        .groupby("code")["stay_date"]

        .nunique()

        .reset_index(name="observed_days")

    )

    evaluation_coverage["expected_days"] = (

        expected_evaluation_days

    )

    evaluation_coverage["missing_days"] = (

        evaluation_coverage["expected_days"]

        - evaluation_coverage["observed_days"]

    )

    print("\nFirst backtest evaluation coverage:")

    print(

        evaluation_coverage

        .to_string(index=False)

    )

    # Attach the prediction for each channel.

    first_evaluation = first_evaluation.merge(

        first_channel_means,

        on="code",

        how="left",

    )

    # Calculate forecast errors.

    first_evaluation["error"] = (

        first_evaluation["room_nights"]

        - first_evaluation["prediction"]

    )

    first_evaluation["absolute_error"] = (

        first_evaluation["error"].abs()

    )

    first_evaluation["squared_error"] = (

        first_evaluation["error"] ** 2

    )

    first_backtest_metrics = (

        first_evaluation

        .groupby("code")

        .agg(

            mae=("absolute_error", "mean"),

            mse=("squared_error", "mean"),

            mean_actual=("room_nights", "mean"),

        )

        .reset_index()

    )

    first_backtest_metrics["rmse"] = (

        first_backtest_metrics["mse"] ** 0.5

    )

    print("\nFirst backtest results:")

    print(

        first_backtest_metrics[

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

    # 5. Run rolling backtests across all forecast origins

    # ---------------------------------------------------------

    all_backtest_results = []

    for origin in forecast_origins:

        result = run_recent_mean_backtest(

            demand=daily_channel_demand,

            origin=origin,

            history_months=3,

            forecast_months=2,

        )

        all_backtest_results.append(result)

    backtest_results = pd.concat(

        all_backtest_results,

        ignore_index=True,

    )

    print("\nRolling backtest results:")

    print(

        backtest_results[

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

    # 6. Compare different historical window lengths

    # ---------------------------------------------------------

    history_windows = [1, 3, 6]

    window_results = []

    for history_months in history_windows:

        for origin in forecast_origins:

            result = run_recent_mean_backtest(

                demand=daily_channel_demand,

                origin=origin,

                history_months=history_months,

                forecast_months=2,

            )

            result["history_months"] = history_months

            window_results.append(result)

    window_backtests = pd.concat(

        window_results,

        ignore_index=True,

    )

    window_summary = (

        window_backtests

        .groupby(["history_months", "code"])

        .agg(

            mean_mae=("mae", "mean"),

            median_mae=("mae", "median"),

            worst_mae=("mae", "max"),

        )

        .reset_index()

    )

    print("\nHistory-window comparison:")

    print(

        window_summary

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 7. Visualize history-window backtest performance

    # ---------------------------------------------------------

    plot_data = (

        window_summary

        .pivot(

            index="code",

            columns="history_months",

            values="mean_mae",

        )

    )

    ax = plot_data.plot(

        kind="bar",

        figsize=(10, 6),

    )

    ax.set_title(

        "Mean Backtest MAE by Historical Window"

    )

    ax.set_xlabel("Booking Channel")

    ax.set_ylabel("Mean MAE (room nights)")

    ax.legend(

        title="History window (months)"

    )

    plt.xticks(rotation=0)

    plt.tight_layout()

    output_path = (

        Path(__file__).resolve().parents[2]

        / "images"

        / "forecasting"

        / "05_history_window_backtest.png"

    )

    plt.savefig(

        output_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close()

    print(

        f"\nSaved history-window chart to: "

        f"{output_path}"

    )

if __name__ == "__main__":

    main()
