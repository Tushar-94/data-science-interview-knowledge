"""Demonstrate temporal train/validation splitting for forecasting."""

from pathlib import Path

import argparse

import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(

        description="Create a temporal forecasting validation split."

    )

    parser.add_argument(

        "--input",

        type=Path,

        required=True,

        help="Path to the private Excel dataset.",

    )

    return parser.parse_args()

def main():

    args = parse_args()

    # ---------------------------------------------------------

    # 1. Load reservations

    # ---------------------------------------------------------

    reservations = pd.read_excel(

        args.input,

        sheet_name="reservations",

    )

    date_columns = [

        "booking_date",

        "arrival_at",

        "departure_at",

        "cancelled_at",

    ]

    for column in date_columns:

        reservations[column] = pd.to_datetime(

            reservations[column],

            errors="coerce",

        )

    # ---------------------------------------------------------

    # 2. Reconstruct realized room-night demand

    # ---------------------------------------------------------

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

    # 3. Build complete modelling grid

    # ---------------------------------------------------------

    modelling_start = pd.Timestamp("2022-10-01")

    validation_start = pd.Timestamp("2024-04-01")

    validation_end = pd.Timestamp("2024-05-31")

    channels = sorted(

        daily_channel_demand["code"].unique()

    )

    full_dates = pd.date_range(

        start=modelling_start,

        end=validation_end,

        freq="D",

    )

    complete_index = pd.MultiIndex.from_product(

        [full_dates, channels],

        names=["stay_date", "code"],

    )

    modelling_demand = (

        daily_channel_demand

        .set_index(["stay_date", "code"])

        .reindex(

            complete_index,

            fill_value=0,

        )

        .reset_index()

    )

    # ---------------------------------------------------------

    # 4. Temporal train/validation split

    # ---------------------------------------------------------

    train = modelling_demand.loc[

        modelling_demand["stay_date"]

        < validation_start

    ].copy()

    validation = modelling_demand.loc[

        (

            modelling_demand["stay_date"]

            >= validation_start

        )

        & (

            modelling_demand["stay_date"]

            <= validation_end

        )

    ].copy()

    print("\nTraining period:")

    print(

        train["stay_date"].min(),

        "to",

        train["stay_date"].max(),

    )

    print("\nValidation period:")

    print(

        validation["stay_date"].min(),

        "to",

        validation["stay_date"].max(),

    )

    print("\nTraining days:")

    print(

        train["stay_date"].nunique()

    )

    print("\nValidation days:")

    print(

        validation["stay_date"].nunique()

    )

    print("\nTraining rows:")

    print(len(train))

    print("\nValidation rows:")

    print(len(validation))

    print("\nChannels:")

    print(channels)

    # ---------------------------------------------------------

    # 5. Seven-day seasonal-naive baseline

    # ---------------------------------------------------------

    last_training_week_start = (

        validation_start

        - pd.Timedelta(days=7)

    )

    last_training_week = train.loc[

        train["stay_date"] >= last_training_week_start

    ].copy()

    last_training_week["weekday"] = (

        last_training_week["stay_date"]

        .dt.dayofweek

    )

    validation["weekday"] = (

        validation["stay_date"]

        .dt.dayofweek

    )

    weekly_pattern = (

        last_training_week[

            [

                "code",

                "weekday",

                "room_nights",

            ]

        ]

        .rename(

            columns={

                "room_nights": "prediction"

            }

        )

    )

    validation_7d = validation.merge(

        weekly_pattern,

        on=["code", "weekday"],

        how="left",

    )

    print("\nLast training week:")

    print(

        last_training_week[

            [

                "stay_date",

                "code",

                "room_nights",

            ]

        ]

        .to_string(index=False)

    )

    print("\nFirst 12 validation predictions:")

    print(

        validation_7d[

            [

                "stay_date",

                "code",

                "room_nights",

                "prediction",

            ]

        ]

        .head(12)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 6. Evaluate the seven-day seasonal-naive baseline

    # ---------------------------------------------------------

    validation_7d["error"] = (

        validation_7d["room_nights"]

        - validation_7d["prediction"]

    )

    validation_7d["absolute_error"] = (

        validation_7d["error"].abs()

    )

    validation_7d["squared_error"] = (

        validation_7d["error"] ** 2

    )

    baseline_metrics = (

        validation_7d

        .groupby("code")

        .agg(

            mae=(

                "absolute_error",

                "mean",

            ),

            mse=(

                "squared_error",

                "mean",

            ),

        )

        .reset_index()

    )

    baseline_metrics["rmse"] = (

        baseline_metrics["mse"] ** 0.5

    )

    baseline_metrics = (

        baseline_metrics[

            [

                "code",

                "mae",

                "rmse",

            ]

        ]

    )

    print("\n7-day seasonal-naive baseline:")

    print(

        baseline_metrics

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 7. Put forecast error into business context

    # ---------------------------------------------------------

    validation_demand_summary = (

        validation_7d

        .groupby("code")["room_nights"]

        .agg(

            mean_actual="mean",

            median_actual="median",

        )

        .reset_index()

    )

    baseline_metrics = baseline_metrics.merge(

        validation_demand_summary,

        on="code",

        how="left",

    )

    baseline_metrics["mae_pct_of_mean"] = (

        baseline_metrics["mae"]

        / baseline_metrics["mean_actual"]

        * 100

    )

    print("\nBaseline performance with demand context:")

    print(

        baseline_metrics[

            [

                "code",

                "mean_actual",

                "median_actual",

                "mae",

                "rmse",

                "mae_pct_of_mean",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 8. Three-month recent-mean baseline

    # ---------------------------------------------------------

    recent_start = (

        validation_start

        - pd.DateOffset(months=3)

    )

    recent_history = train.loc[

        train["stay_date"] >= recent_start

    ].copy()

    recent_channel_means = (

        recent_history

        .groupby("code")["room_nights"]

        .mean()

        .rename("prediction")

        .reset_index()

    )

    validation_recent_mean = validation.merge(

        recent_channel_means,

        on="code",

        how="left",

    )

    validation_recent_mean["error"] = (

        validation_recent_mean["room_nights"]

        - validation_recent_mean["prediction"]

    )

    validation_recent_mean["absolute_error"] = (

        validation_recent_mean["error"].abs()

    )

    validation_recent_mean["squared_error"] = (

        validation_recent_mean["error"] ** 2

    )

    recent_mean_metrics = (

        validation_recent_mean

        .groupby("code")

        .agg(

            mae=("absolute_error", "mean"),

            mse=("squared_error", "mean"),

        )

        .reset_index()

    )

    recent_mean_metrics["rmse"] = (

        recent_mean_metrics["mse"] ** 0.5

    )

    print("\nThree-month historical means:")

    print(

        recent_channel_means

        .round(2)

        .to_string(index=False)

    )

    print("\nThree-month recent-mean baseline:")

    print(

        recent_mean_metrics[

            ["code", "mae", "rmse"]

        ]

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 9. Compare baseline MAE

    # ---------------------------------------------------------

    import matplotlib.pyplot as plt

    comparison = baseline_metrics[

        ["code", "mae"]

    ].rename(

        columns={"mae": "repeated_last_week"}

    )

    comparison = comparison.merge(

        recent_mean_metrics[

            ["code", "mae"]

        ].rename(

            columns={"mae": "three_month_mean"}

        ),

        on="code",

    )

    print("\nBaseline MAE comparison:")

    print(

        comparison

        .round(2)

        .to_string(index=False)

    )

    output_dir = Path("images/forecasting")

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    figure_path = (

        output_dir

        / "04_baseline_mae_comparison.png"

    )

    comparison_plot = comparison.set_index("code")

    comparison_plot.plot(

        kind="bar",

        figsize=(10, 6),

    )

    plt.title("Forecast Baseline Comparison")

    plt.xlabel("Booking channel")

    plt.ylabel("MAE (room nights)")

    plt.xticks(rotation=25, ha="right")

    plt.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    plt.savefig(

        figure_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close()

    print("\nBaseline comparison chart saved to:")

    print(figure_path)

if __name__ == "__main__":

    main()
