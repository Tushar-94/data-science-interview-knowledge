"""Generate the final 365-day channel demand forecast.

The forecast uses channel-specific recent-mean models selected through

rolling-origin backtesting.

Forecast origin:

    2024-06-01

Forecast horizon:

    2024-06-01 to 2025-05-31

Important:

No observations on or after the forecast origin may be used to fit the

forecast.

"""

import argparse

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

FORECAST_ORIGIN = pd.Timestamp("2024-06-01")

FORECAST_END = (

    FORECAST_ORIGIN

    + pd.DateOffset(years=1)

    - pd.Timedelta(days=1)

)

CHANNEL_HISTORY_MONTHS = {

    "BookingCom": 6,

    "ChannelManager": 3,

    "Direct": 1,

    "Ibe": 3,

}

BACKTEST_MAE = {

    "BookingCom": 5.69,

    "ChannelManager": 2.61,

    "Direct": 3.60,

    "Ibe": 4.67,

}

SCENARIO_STRESS_PCT = 0.08

def parse_args():

    parser = argparse.ArgumentParser(

        description="Generate the final 365-day demand forecast."

    )

    parser.add_argument(

        "--input",

        required=True,

        help="Path to the private Limehome Excel workbook.",

    )

    return parser.parse_args()

def calculate_historical_total_errors(

    demand,

    origins,

    history_windows,

    forecast_months=2,

):

    """Measure total-demand error in historical forecast simulations."""

    results = []

    for origin in origins:

        origin = pd.Timestamp(origin)

        evaluation_end = (

            origin

            + pd.DateOffset(months=forecast_months)

            - pd.Timedelta(days=1)

        )

        predicted_total = 0.0

        actual_total = 0.0

        for channel, history_months in history_windows.items():

            history_start = (

                origin

                - pd.DateOffset(months=history_months)

            )

            history_end = (

                origin

                - pd.Timedelta(days=1)

            )

            channel_history = demand.loc[

                (demand["code"] == channel)

                & (demand["stay_date"] >= history_start)

                & (demand["stay_date"] <= history_end)

            ].copy()

            evaluation = demand.loc[

                (demand["code"] == channel)

                & (demand["stay_date"] >= origin)

                & (demand["stay_date"] <= evaluation_end)

            ].copy()

            history_dates = pd.date_range(

                history_start,

                history_end,

                freq="D",

            )

            evaluation_dates = pd.date_range(

                origin,

                evaluation_end,

                freq="D",

            )

            channel_history = (

                channel_history

                .set_index("stay_date")

                .reindex(history_dates, fill_value=0)

                .rename_axis("stay_date")

                .reset_index()

            )

            evaluation = (

                evaluation

                .set_index("stay_date")

                .reindex(evaluation_dates, fill_value=0)

                .rename_axis("stay_date")

                .reset_index()

            )

            daily_prediction = (

                channel_history["room_nights"].mean()

            )

            predicted_total += (

                daily_prediction * len(evaluation_dates)

            )

            actual_total += evaluation[

                "room_nights"

            ].sum()

        error = predicted_total - actual_total

        error_pct = (

            100 * error / actual_total

            if actual_total > 0

            else np.nan

        )

        results.append(

            {

                "forecast_origin": origin,

                "predicted_total": predicted_total,

                "actual_total": actual_total,

                "error": error,

                "absolute_error": abs(error),

                "error_pct": error_pct,

                "absolute_error_pct": abs(error_pct),

            }

        )

    return pd.DataFrame(results)

def main():

    args = parse_args()

    input_path = Path(args.input).expanduser()

    forecast_dates = pd.date_range(

        start=FORECAST_ORIGIN,

        end=FORECAST_END,

        freq="D",

    )

    print(f"Input file: {input_path}")

    print("\nForecast configuration:")

    print(f"Origin: {FORECAST_ORIGIN.date()}")

    print(f"End:    {FORECAST_END.date()}")

    print(f"Days:   {len(forecast_dates)}")

    print("\nChannel history windows:")

    for channel, months in CHANNEL_HISTORY_MONTHS.items():

        print(

            f"{channel:<16} {months} month(s)"

        )

    # ---------------------------------------------------------

    # 1. Load and prepare reservation data

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

    completed_stays = reservations.loc[

        (reservations["status"] == "CheckedOut")

        & (reservations["length_of_stay"] > 0)

    ].copy()

    # ---------------------------------------------------------

    # 2. Expand completed stays into daily room nights

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

    # 3. Aggregate to daily channel demand

    # ---------------------------------------------------------

    daily_demand = (

        room_nights

        .groupby(["stay_date", "code"])

        .size()

        .rename("room_nights")

        .reset_index()

    )

    # ---------------------------------------------------------

    # 4. Apply the forecast-time information boundary

    # ---------------------------------------------------------

    historical_demand = daily_demand.loc[

        daily_demand["stay_date"] < FORECAST_ORIGIN

    ].copy()

    if historical_demand["stay_date"].max() >= FORECAST_ORIGIN:

        raise ValueError(

            "Leakage detected: historical demand contains "

            "observations on or after the forecast origin."

        )

    print("\nHistorical information boundary:")

    print(

        "Latest raw demand date: "

        f"{daily_demand['stay_date'].max().date()}"

    )

    print(

        "Latest allowed historical date: "

        f"{historical_demand['stay_date'].max().date()}"

    )

    print(

        "Forecast starts: "

        f"{FORECAST_ORIGIN.date()}"

    )

    print(

        "\nRoom-night expansion valid: "

        f"{expected_room_nights:,} == "

        f"{actual_room_nights:,}"

    )

    # ---------------------------------------------------------

    # 5. Calculate final channel-level historical means

    # ---------------------------------------------------------

    channel_models = []

    history_end = (

        FORECAST_ORIGIN

        - pd.Timedelta(days=1)

    )

    for channel, history_months in CHANNEL_HISTORY_MONTHS.items():

        history_start = (

            FORECAST_ORIGIN

            - pd.DateOffset(months=history_months)

        )

        channel_history = historical_demand.loc[

            (historical_demand["code"] == channel)

            & (historical_demand["stay_date"] >= history_start)

            & (historical_demand["stay_date"] <= history_end)

        ].copy()

        # Build a complete daily series so dates with zero realized

        # demand are represented explicitly.

        expected_dates = pd.date_range(

            start=history_start,

            end=history_end,

            freq="D",

        )

        channel_history = (

            channel_history

            .set_index("stay_date")

            .reindex(expected_dates, fill_value=0)

            .rename_axis("stay_date")

            .reset_index()

        )

        expected_days = len(expected_dates)

        actual_days = len(channel_history)

        if actual_days != expected_days:

            raise ValueError(

                f"History coverage failed for {channel}: "

                f"expected {expected_days} days, "

                f"found {actual_days}."

            )

        mean_daily_demand = (

            channel_history["room_nights"].mean()

        )

        channel_models.append(

            {

                "code": channel,

                "history_months": history_months,

                "history_start": history_start,

                "history_end": history_end,

                "history_days": expected_days,

                "mean_daily_demand": mean_daily_demand,

            }

        )

    channel_models = pd.DataFrame(channel_models)

    print("\nFinal channel model inputs:")

    print(

        channel_models

        .round({"mean_daily_demand": 2})

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 6. Generate 365-day base forecast

    # ---------------------------------------------------------

    forecast_rows = []

    for _, model in channel_models.iterrows():

        channel = model["code"]

        daily_prediction = model["mean_daily_demand"]

        for forecast_date in forecast_dates:

            forecast_rows.append(

                {

                    "stay_date": forecast_date,

                    "code": channel,

                    "predicted_room_nights": daily_prediction,

                }

            )

    forecast = pd.DataFrame(forecast_rows)

    expected_rows = (

        len(forecast_dates)

        * len(CHANNEL_HISTORY_MONTHS)

    )

    if len(forecast) != expected_rows:

        raise ValueError(

            "Forecast shape validation failed: "

            f"expected {expected_rows:,} rows, "

            f"found {len(forecast):,}."

        )

    if forecast["predicted_room_nights"].isna().any():

        raise ValueError(

            "Forecast contains missing predictions."

        )

    if (forecast["predicted_room_nights"] < 0).any():

        raise ValueError(

            "Forecast contains negative room-night predictions."

        )

    print("\nBase forecast validation:")

    print(

        f"Forecast dates: "

        f"{forecast['stay_date'].min().date()} "

        f"to {forecast['stay_date'].max().date()}"

    )

    print(

        f"Days: {forecast['stay_date'].nunique()}"

    )

    print(

        f"Channels: {forecast['code'].nunique()}"

    )

    print(

        f"Rows: {len(forecast):,} "

        f"(expected {expected_rows:,})"

    )

    print(

        "Missing predictions: "

        f"{forecast['predicted_room_nights'].isna().sum()}"

    )

    print(

        "Negative predictions: "

        f"{(forecast['predicted_room_nights'] < 0).sum()}"

    )

    # ---------------------------------------------------------

    # 7. Summarize annual forecast by channel

    # ---------------------------------------------------------

    annual_summary = (

        forecast

        .groupby("code")

        .agg(

            avg_daily_room_nights=(

                "predicted_room_nights",

                "mean",

            ),

            annual_room_nights=(

                "predicted_room_nights",

                "sum",

            ),

        )

        .reset_index()

    )

    total_room_nights = annual_summary[

        "annual_room_nights"

    ].sum()

    annual_summary["share_pct"] = (

        100

        * annual_summary["annual_room_nights"]

        / total_room_nights

    )

    print("\n365-day base forecast by channel:")

    print(

        annual_summary

        .round(

            {

                "avg_daily_room_nights": 2,

                "annual_room_nights": 0,

                "share_pct": 1,

            }

        )

        .to_string(index=False)

    )

    print(

        "\nTotal expected room nights: "

        f"{total_room_nights:,.0f}"

    )

    print(

        "Average total room nights per day: "

        f"{total_room_nights / len(forecast_dates):.2f}"

    )

    # ---------------------------------------------------------

    # 8. Add empirical backtest error context

    # ---------------------------------------------------------

    annual_summary["backtest_mae"] = (

        annual_summary["code"].map(BACKTEST_MAE)

    )

    annual_summary["mae_pct_of_daily_forecast"] = (

        100

        * annual_summary["backtest_mae"]

        / annual_summary["avg_daily_room_nights"]

    )

    print("\nBacktest error relative to forecast level:")

    print(

        annual_summary[

            [

                "code",

                "avg_daily_room_nights",

                "backtest_mae",

                "mae_pct_of_daily_forecast",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 9. Measure historical aggregate forecast error

    # ---------------------------------------------------------

    backtest_origins = pd.to_datetime(

        [

            "2023-07-01",

            "2023-10-01",

            "2024-01-01",

            "2024-04-01",

        ]

    )

    historical_errors = calculate_historical_total_errors(

        demand=historical_demand,

        origins=backtest_origins,

        history_windows=CHANNEL_HISTORY_MONTHS,

        forecast_months=2,

    )

    print("\nHistorical total-demand forecast errors:")

    print(

        historical_errors[

            [

                "forecast_origin",

                "predicted_total",

                "actual_total",

                "error",

                "error_pct",

                "absolute_error_pct",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    print(

        "\nHistorical aggregate error summary:"

    )

    print(

        "Mean absolute percentage error of totals: "

        f"{historical_errors['absolute_error_pct'].mean():.2f}%"

    )

    print(

        "Median absolute percentage error of totals: "

        f"{historical_errors['absolute_error_pct'].median():.2f}%"

    )

    print(

        "Worst absolute percentage error of totals: "

        f"{historical_errors['absolute_error_pct'].max():.2f}%"

    )

    # ---------------------------------------------------------

    # 10. Build planning scenarios

    # ---------------------------------------------------------

    base_total = total_room_nights

    scenario_summary = pd.DataFrame(

        {

            "scenario": [

                "Downside",

                "Base",

                "Upside",

            ],

            "multiplier": [

                1 - SCENARIO_STRESS_PCT,

                1.0,

                1 + SCENARIO_STRESS_PCT,

            ],

        }

    )

    scenario_summary["annual_room_nights"] = (

        base_total

        * scenario_summary["multiplier"]

    )

    scenario_summary["avg_daily_room_nights"] = (

        scenario_summary["annual_room_nights"]

        / len(forecast_dates)

    )

    print("\n365-day planning scenarios:")

    print(

        scenario_summary

        .round(

            {

                "multiplier": 2,

                "annual_room_nights": 0,

                "avg_daily_room_nights": 2,

            }

        )

        .to_string(index=False)

    )

    print(

        "\nScenario interpretation:"

    )

    print(

        "The +/-8% stress is informed by the worst "

        "aggregate absolute error observed in the historical "

        "backtests (~7.87%), rounded to 8%."

    )

    print(

        "These are planning scenarios, NOT statistical "

        "prediction intervals."

    )

    print(

        "Because the backtests used approximately two-month "

        "horizons while the final forecast spans 365 days, "

        "long-horizon uncertainty may be larger."

    )

    # ---------------------------------------------------------

    # 11. Visualize annual demand forecast by channel

    # ---------------------------------------------------------

    output_path = Path(

        "images/forecasting/09_final_annual_demand_forecast.png"

    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    plot_data = annual_summary.sort_values(

        "annual_room_nights",

        ascending=False,

    )

    fig, ax = plt.subplots(

        figsize=(10, 6)

    )

    bars = ax.bar(

        plot_data["code"],

        plot_data["annual_room_nights"],

    )

    ax.set_title(

        "365-Day Base Demand Forecast by Booking Channel"

    )

    ax.set_ylabel(

        "Expected Room Nights"

    )

    ax.set_xlabel(

        "Booking Channel"

    )

    for bar, value in zip(

        bars,

        plot_data["annual_room_nights"],

    ):

        ax.text(

            bar.get_x() + bar.get_width() / 2,

            bar.get_height(),

            f"{value:,.0f}",

            ha="center",

            va="bottom",

        )

    fig.tight_layout()

    fig.savefig(

        output_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close(fig)

    print(

        f"\nSaved annual forecast chart: {output_path}"

    )

    # ---------------------------------------------------------

    # 12. Visualize planning scenarios

    # ---------------------------------------------------------

    scenario_output_path = Path(

        "images/forecasting/10_demand_scenarios.png"

    )

    fig, ax = plt.subplots(

        figsize=(8, 6)

    )

    scenario_bars = ax.bar(

        scenario_summary["scenario"],

        scenario_summary["annual_room_nights"],

    )

    ax.set_title(

        "365-Day Demand Planning Scenarios"

    )

    ax.set_ylabel(

        "Expected Room Nights"

    )

    ax.set_xlabel(

        "Scenario"

    )

    for bar, value in zip(

        scenario_bars,

        scenario_summary["annual_room_nights"],

    ):

        ax.text(

            bar.get_x() + bar.get_width() / 2,

            bar.get_height(),

            f"{value:,.0f}",

            ha="center",

            va="bottom",

        )

    fig.tight_layout()

    fig.savefig(

        scenario_output_path,

        dpi=150,

        bbox_inches="tight",

    )

    plt.close(fig)

    print(

        f"Saved scenario chart: {scenario_output_path}"

    )

if __name__ == "__main__":

    main()
