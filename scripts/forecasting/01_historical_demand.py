"""Explore historical realized room-night demand.

This script is for private learning/reproduction of the forecasting case study.

The source assessment dataset is intentionally kept outside this repository.

"""
import matplotlib.pyplot as plt

from pathlib import Path

import argparse

import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(

        description="Explore historical hospitality demand."

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

    # 1. Load source data

    # ---------------------------------------------------------

    reservations = pd.read_excel(

        args.input,

        sheet_name="reservations",

    )

    occupancy = pd.read_excel(

        args.input,

        sheet_name="occupancy_rate",

    )

    print("Reservations shape:", reservations.shape)

    print("Occupancy shape:", occupancy.shape)

    # ---------------------------------------------------------

    # 2. Parse dates

    # ---------------------------------------------------------

    reservation_date_columns = [

        "booking_date",

        "arrival_at",

        "departure_at",

        "cancelled_at",

    ]

    for column in reservation_date_columns:

        reservations[column] = pd.to_datetime(

            reservations[column],

            errors="coerce",

        )

    occupancy["date"] = pd.to_datetime(

        occupancy["date"],

        errors="coerce",

    )

    # ---------------------------------------------------------

    # 3. Basic data-quality checks

    # ---------------------------------------------------------

    reservations["length_of_stay"] = (

        reservations["departure_at"]

        - reservations["arrival_at"]

    ).dt.days

    print("\nReservation status counts:")

    print(

        reservations["status"]

        .value_counts(dropna=False)

    )

    print("\nInvalid/non-positive stays:")

    print(

        (reservations["length_of_stay"] <= 0).sum()

    )

    print("\nBooking channels:")

    print(

        reservations["code"]

        .value_counts(dropna=False)

    )

    # ---------------------------------------------------------

    # 4. Define valid realized stays

    # ---------------------------------------------------------

    completed_stays = reservations.loc[

        (reservations["status"] == "CheckedOut")

        & (reservations["length_of_stay"] > 0)

    ].copy()

    print("\nValid completed reservation lines:")

    print(len(completed_stays))

    # ---------------------------------------------------------

    # 5. Expand each stay into individual room nights

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

    # ---------------------------------------------------------

    # 6. Validate the transformation

    # ---------------------------------------------------------

    expected_room_nights = (

        completed_stays["length_of_stay"].sum()

    )

    actual_room_nights = len(room_nights)

    print("\nRoom-night transformation validation:")

    print("Expected:", expected_room_nights)

    print("Actual:", actual_room_nights)

    print(

        "Valid:",

        expected_room_nights == actual_room_nights,

    )

    # ---------------------------------------------------------

    # 7. Aggregate into daily channel demand

    # ---------------------------------------------------------

    daily_channel_demand = (

        room_nights

        .groupby(

            ["stay_date", "code"]

        )

        .size()

        .rename("room_nights")

        .reset_index()

    )

    print("\nDaily demand coverage:")

    print(

        daily_channel_demand["stay_date"].min(),

        "to",

        daily_channel_demand["stay_date"].max(),

    )

    # ---------------------------------------------------------

    # 8. Define usable historical data window

    # ---------------------------------------------------------

    modelling_start = pd.Timestamp("2022-10-01")

    data_end = min(

        daily_channel_demand["stay_date"].max(),

        occupancy["date"].max(),

    )

    print("\nUsable historical data window:")

    print("Start:", modelling_start)

    print("End:", data_end)

    print(

        "Number of calendar days:",

        (data_end - modelling_start).days + 1,

    )

    # ---------------------------------------------------------

    # 9. Build a complete date × channel time-series grid

    # ---------------------------------------------------------

    channels = sorted(

        daily_channel_demand["code"].unique()

    )

    full_dates = pd.date_range(

        start=modelling_start,

        end=data_end,

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

    print("\nChannels used for modelling:")

    print(channels)

    print("\nComplete modelling dataset shape:")

    print(modelling_demand.shape)

    print("\nExpected number of rows:")

    print(len(full_dates) * len(channels))

    print("\nMissing values:")

    print(modelling_demand.isna().sum())

    print("\nFirst 12 rows:")

    print(

        modelling_demand

        .head(12)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 10. Visualize total historical realized demand

    # ---------------------------------------------------------

    total_daily_demand = (

        modelling_demand

        .groupby("stay_date")["room_nights"]

        .sum()

        .reset_index()

    )

    total_daily_demand["rolling_7d"] = (

        total_daily_demand["room_nights"]

        .rolling(

            window=7,

            min_periods=1,

        )

        .mean()

    )

    output_dir = Path("images/forecasting")

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    figure_path = (

        output_dir

        / "01_historical_daily_demand.png"

    )

    plt.figure(

        figsize=(14, 6)

    )

    plt.plot(

        total_daily_demand["stay_date"],

        total_daily_demand["room_nights"],

        alpha=0.35,

        label="Daily room nights",

    )

    plt.plot(

        total_daily_demand["stay_date"],

        total_daily_demand["rolling_7d"],

        linewidth=2,

        label="7-day rolling average",

    )

    plt.title(

        "Historical Realized Room-Night Demand"

    )

    plt.xlabel("Stay date")

    plt.ylabel("Realized room nights")

    plt.legend()

    plt.grid(

        alpha=0.2

    )

    plt.tight_layout()

    plt.savefig(

        figure_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close()

    print("\nHistorical-demand chart saved to:")

    print(figure_path)

    # ---------------------------------------------------------

    # 11. Explore day-of-week seasonality

    # ---------------------------------------------------------

    total_daily_demand["day_of_week"] = (

        total_daily_demand["stay_date"]

        .dt.day_name()

    )

    day_order = [

        "Monday",

        "Tuesday",

        "Wednesday",

        "Thursday",

        "Friday",

        "Saturday",

        "Sunday",

    ]

    weekday_summary = (

        total_daily_demand

        .groupby("day_of_week")["room_nights"]

        .agg(["mean", "median", "std", "count"])

        .reindex(day_order)

    )

    print("\nDemand by day of week:")

    print(

        weekday_summary.round(2)

    )

    # ---------------------------------------------------------

    # 12. Visualize day-of-week seasonality

    # ---------------------------------------------------------

    weekday_plot = (

        weekday_summary

        .reset_index()

    )

    weekday_figure_path = (

        output_dir

        / "02_demand_by_weekday.png"

    )

    plt.figure(

        figsize=(10, 6)

    )

    plt.bar(

        weekday_plot["day_of_week"],

        weekday_plot["mean"],

    )

    plt.title(

        "Average Realized Room-Night Demand by Day of Week"

    )

    plt.xlabel("Day of week")

    plt.ylabel("Average realized room nights")

    plt.xticks(

        rotation=30,

        ha="right",

    )

    plt.grid(

        axis="y",

        alpha=0.2,

    )

    plt.tight_layout()

    plt.savefig(

        weekday_figure_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close()

    print("\nWeekday-demand chart saved to:")

    print(weekday_figure_path)

    # ---------------------------------------------------------

    # 13. Explore demand by calendar month

    # ---------------------------------------------------------

    total_daily_demand["year_month"] = (

        total_daily_demand["stay_date"]

        .dt.to_period("M")

    )

    monthly_summary = (

        total_daily_demand

        .groupby("year_month")["room_nights"]

        .agg(["mean", "median", "std", "min", "max", "count"])

        .reset_index()

    )

    print("\nDemand by calendar month:")

    print(

        monthly_summary

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 14. Visualize monthly demand evolution

    # ---------------------------------------------------------

    monthly_plot = monthly_summary.copy()

    monthly_plot["year_month"] = (

        monthly_plot["year_month"]

        .astype(str)

    )

    monthly_figure_path = (

        output_dir

        / "03_monthly_demand_evolution.png"

    )

    plt.figure(

        figsize=(12, 6)

    )

    plt.plot(

        monthly_plot["year_month"],

        monthly_plot["mean"],

        marker="o",

    )

    plt.title(

        "Average Daily Realized Demand by Month"

    )

    plt.xlabel("Month")

    plt.ylabel("Average realized room nights")

    plt.xticks(

        rotation=45,

        ha="right",

    )

    plt.grid(

        alpha=0.2,

    )

    plt.tight_layout()

    plt.savefig(

        monthly_figure_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close()

    print("\nMonthly-demand chart saved to:")

    print(monthly_figure_path)

if __name__ == "__main__":

    main()