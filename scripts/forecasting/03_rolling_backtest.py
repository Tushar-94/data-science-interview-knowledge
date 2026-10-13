"""Rolling backtesting for demand forecasting."""

import pandas as pd

from pathlib import Path

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

if __name__ == "__main__":

    main()
