"""Evaluate channel-specific history-window selection.

Different booking channels may require different amounts of historical

data. This experiment evaluates a simple recent-mean forecast using a

channel-specific history window selected from previous rolling

backtests.

Important:

The same historical backtests contributed to window selection, so the

result is model-selection evidence rather than an unbiased estimate of

future production performance.

"""

import argparse

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(

        description=(

            "Evaluate channel-specific forecasting windows."

        )

    )

    parser.add_argument(

        "--input",

        required=True,

        help="Path to the private Limehome Excel workbook.",

    )

    return parser.parse_args()

CHANNEL_HISTORY_MONTHS = {

    "BookingCom": 6,

    "ChannelManager": 3,

    "Direct": 1,

    "Ibe": 3,

}

def run_channel_specific_backtest(

    demand,

    origin,

    history_windows,

    forecast_months=2,

):

    """Backtest recent-mean forecasts with channel-specific history windows."""

    origin = pd.Timestamp(origin)

    evaluation_start = origin

    evaluation_end = (

        origin

        + pd.DateOffset(months=forecast_months)

        - pd.Timedelta(days=1)

    )

    results = []

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

        channel_evaluation = demand.loc[

            (demand["code"] == channel)

            & (demand["stay_date"] >= evaluation_start)

            & (demand["stay_date"] <= evaluation_end)

        ].copy()

        # -----------------------------------------------------

        # Validate temporal coverage

        # -----------------------------------------------------

        expected_history_days = (

            history_end - history_start

        ).days + 1

        expected_evaluation_days = (

            evaluation_end - evaluation_start

        ).days + 1

        actual_history_days = (

            channel_history["stay_date"].nunique()

        )

        actual_evaluation_days = (

            channel_evaluation["stay_date"].nunique()

        )

        if actual_history_days != expected_history_days:

            raise ValueError(

                f"Incomplete history for {channel} "

                f"at origin {origin.date()}: "

                f"expected {expected_history_days} days, "

                f"found {actual_history_days}."

            )

        if actual_evaluation_days != expected_evaluation_days:

            raise ValueError(

                f"Incomplete evaluation for {channel} "

                f"at origin {origin.date()}: "

                f"expected {expected_evaluation_days} days, "

                f"found {actual_evaluation_days}."

            )

        # -----------------------------------------------------

        # Fit recent-mean model

        # -----------------------------------------------------

        prediction = (

            channel_history["room_nights"].mean()

        )

        channel_evaluation["prediction"] = prediction

        channel_evaluation["absolute_error"] = np.abs(

            channel_evaluation["room_nights"]

            - channel_evaluation["prediction"]

        )

        channel_evaluation["squared_error"] = (

            channel_evaluation["room_nights"]

            - channel_evaluation["prediction"]

        ) ** 2

        mae = channel_evaluation[

            "absolute_error"

        ].mean()

        rmse = np.sqrt(

            channel_evaluation[

                "squared_error"

            ].mean()

        )

        results.append(

            {

                "forecast_origin": origin,

                "code": channel,

                "history_months": history_months,

                "prediction": prediction,

                "mean_actual": channel_evaluation[

                    "room_nights"

                ].mean(),

                "mae": mae,

                "rmse": rmse,

            }

        )

    return pd.DataFrame(results)

def run_common_window_backtest(

    demand,

    origin,

    history_months=3,

    forecast_months=2,

):

    """Backtest the same recent-mean history window for every channel."""

    channels = sorted(demand["code"].unique())

    history_windows = {

        channel: history_months

        for channel in channels

    }

    return run_channel_specific_backtest(

        demand=demand,

        origin=origin,

        history_windows=history_windows,

        forecast_months=forecast_months,

    )

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

        "Room-night expansion valid: "

        f"{expected_room_nights:,} == "

        f"{actual_room_nights:,}"

    )

    # ---------------------------------------------------------

    # 6. Test channel-specific configuration at one origin

    # ---------------------------------------------------------

    test_origin = pd.Timestamp("2024-04-01")

    test_result = run_channel_specific_backtest(

        demand=daily_demand,

        origin=test_origin,

        history_windows=CHANNEL_HISTORY_MONTHS,

        forecast_months=2,

    )

    print(

        "\nChannel-specific model at "

        "2024-04-01 forecast origin:"

    )

    print(

        test_result[

            [

                "code",

                "history_months",

                "prediction",

                "mean_actual",

                "mae",

                "rmse",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 7. Rolling backtest channel-specific configuration

    # ---------------------------------------------------------

    forecast_origins = pd.to_datetime(

        [

            "2023-07-01",

            "2023-10-01",

            "2024-01-01",

            "2024-04-01",

        ]

    )

    channel_specific_backtests = []

    for origin in forecast_origins:

        result = run_channel_specific_backtest(

            demand=daily_demand,

            origin=origin,

            history_windows=CHANNEL_HISTORY_MONTHS,

            forecast_months=2,

        )

        channel_specific_backtests.append(result)

    channel_specific_backtests = pd.concat(

        channel_specific_backtests,

        ignore_index=True,

    )

    print(

        "\nChannel-specific rolling backtest results:"

    )

    print(

        channel_specific_backtests[

            [

                "forecast_origin",

                "code",

                "history_months",

                "prediction",

                "mean_actual",

                "mae",

                "rmse",

            ]

        ]

        .round(2)

        .to_string(index=False)

    )

    summary = (

        channel_specific_backtests

        .groupby(["code", "history_months"])

        .agg(

            mean_mae=("mae", "mean"),

            median_mae=("mae", "median"),

            worst_mae=("mae", "max"),

        )

        .reset_index()

    )

    print(

        "\nChannel-specific performance summary:"

    )

    print(

        summary

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 8. Compare common vs channel-specific configuration

    # ---------------------------------------------------------

    common_backtests = []

    for origin in forecast_origins:

        result = run_common_window_backtest(

            demand=daily_demand,

            origin=origin,

            history_months=3,

            forecast_months=2,

        )

        common_backtests.append(result)

    common_backtests = pd.concat(

        common_backtests,

        ignore_index=True,

    )

    common_summary = (

        common_backtests

        .groupby("code")

        .agg(

            common_mean_mae=("mae", "mean"),

        )

        .reset_index()

    )

    specific_summary = (

        channel_specific_backtests

        .groupby("code")

        .agg(

            specific_mean_mae=("mae", "mean"),

        )

        .reset_index()

    )

    comparison = common_summary.merge(

        specific_summary,

        on="code",

        validate="one_to_one",

    )

    comparison["mae_change"] = (

        comparison["specific_mean_mae"]

        - comparison["common_mean_mae"]

    )

    comparison["relative_change_pct"] = (

        100

        * comparison["mae_change"]

        / comparison["common_mean_mae"]

    )

    print(

        "\nCommon 3-month vs channel-specific configuration:"

    )

    print(

        comparison

        .round(2)

        .to_string(index=False)

    )

    # ---------------------------------------------------------

    # 9. Visualize common vs channel-specific configuration

    # ---------------------------------------------------------

    output_path = Path(

        "images/forecasting/08_channel_specific_comparison.png"

    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    x = np.arange(len(comparison))

    width = 0.35

    fig, ax = plt.subplots(

        figsize=(10, 6)

    )

    ax.bar(

        x - width / 2,

        comparison["common_mean_mae"],

        width,

        label="Common 3-month window",

    )

    ax.bar(

        x + width / 2,

        comparison["specific_mean_mae"],

        width,

        label="Channel-specific window",

    )

    ax.set_xticks(x)

    ax.set_xticklabels(comparison["code"])

    ax.set_ylabel(

        "Mean Backtest MAE (room nights)"

    )

    ax.set_title(

        "Common vs Channel-Specific History Windows"

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
