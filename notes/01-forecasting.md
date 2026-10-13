# Forecasting

This note captures what I learned about solving a real-world demand forecasting problem from raw reservation data.

The objective is not to memorize one forecasting solution. It is to understand how to go from an ambiguous business question to a defensible forecasting target, dataset, validation strategy, model, and business recommendation.

---

# 1. Start With the Business Question

A forecasting problem should not begin with:

> Which machine-learning model should I use?

It should begin with:

> What exactly am I trying to predict, and what business decision will use that prediction?

In the case study, the high-level requirement was to forecast demand for the next 365 days separately by booking channel.

But "demand" is ambiguous.

It could mean:

- number of bookings,
- number of reservation lines,
- number of arrivals,
- requested room nights,
- realized room nights,
- occupancy,
- or unconstrained customer demand.

These are not equivalent.

Therefore, defining the forecasting target is one of the most important parts of the problem.

---

# 2. What Is a Room Night?

A **room night** represents one room being occupied for one night.

Suppose a reservation has:

```text
Arrival:   1 July
Departure: 4 July
```

The guest occupies the room on:

```text
1 July
2 July
3 July
```

Therefore:

```text
Length of stay = 3 nights
Room nights    = 3
```

The departure date is not an occupied night.

Mathematically:

```text
Length of Stay = Departure Date - Arrival Date
```

---

# 3. Why Not Simply Forecast Number of Bookings?

Bookings do not represent the amount of inventory consumed.

Consider:

```text
Booking A → 1 room × 1 night = 1 room night
Booking B → 1 room × 10 nights = 10 room nights
```

If I counted only bookings:

```text
Booking A = 1
Booking B = 1
```

They appear equally important.

But Booking B consumes ten times as much accommodation inventory.

For a hospitality demand problem, room nights therefore provide a more useful measure of inventory consumption.

---

# 4. Requested Demand vs Realized Demand

Another important distinction is between:

## Requested room nights

Room nights originally requested through reservations.

This represents booking intent before cancellations.

## Realized room nights

Room nights associated with stays that were actually completed.

This represents accommodation demand that actually materialized.

A cancelled reservation contains useful information about customer intent, but it does not necessarily result in an occupied room.

For the primary forecasting target, I therefore used:

> **Daily realized room nights by booking channel, allocated to the actual calendar dates on which the room inventory was consumed.**

Cancelled reservations were retained for separate cancellation analysis rather than included in the primary realized-demand target.

---

# 5. Transforming Reservations Into Daily Room Nights

Raw reservation data is usually stored at booking or reservation-line level.

That is not directly suitable for daily demand forecasting.

Suppose the raw data contains:

```text
booking_id | arrival    | departure  | channel
-----------|------------|------------|----------
ABC123     | 2024-07-01 | 2024-07-04 | BookingCom
```

For forecasting daily inventory consumption, it should become:

```text
booking_id | stay_date  | channel
-----------|------------|-----------
ABC123     | 2024-07-01 | BookingCom
ABC123     | 2024-07-02 | BookingCom
ABC123     | 2024-07-03 | BookingCom
```

Now every row represents one realized room night.

---

# 6. Python: Expanding Reservations Into Room Nights

The core transformation is:

```python
completed_stays = reservations.loc[
    (reservations["status"] == "CheckedOut")
    & (reservations["length_of_stay"] > 0)
].copy()

completed_stays["stay_date"] = completed_stays.apply(
    lambda row: pd.date_range(
        start=row["arrival_at"],
        end=row["departure_at"] - pd.Timedelta(days=1),
        freq="D"
    ),
    axis=1
)

room_nights = completed_stays.explode(
    "stay_date",
    ignore_index=True
)
```

### What is happening?

`pd.date_range()` creates all occupied dates between arrival and the night before departure.

`explode()` then converts the list of dates into individual rows.

Conceptually:

```text
ONE RESERVATION

Arrival ───────────────────────────── Departure
  │                                      │
  ▼                                      ▼
July 1       July 2       July 3       July 4
  │            │            │            │
  ▼            ▼            ▼            ▼
occupied     occupied     occupied     leave

BECOMES

Reservation
    │
    ├──── July 1 → 1 room night
    ├──── July 2 → 1 room night
    └──── July 3 → 1 room night
```

---

# 7. Always Validate the Transformation

A transformation can run successfully and still be wrong.

Therefore I checked:

```python
expected_room_nights = completed_stays[
    "length_of_stay"
].sum()

actual_room_nights = len(room_nights)

print("Expected:", expected_room_nights)
print("Actual:", actual_room_nights)
print(
    "Transformation valid:",
    expected_room_nights == actual_room_nights
)
```

The logic is:

```text
Sum of original lengths of stay
                =
Number of rows after room-night expansion
```

In the case study:

```text
Expected room nights = 51,696
Expanded rows        = 51,696
Transformation valid = True
```

This is an important data-engineering lesson:

> Never assume a transformation preserved the data correctly just because the code executed without an error.

Always create an invariant or reconciliation check.

---

# 8. Aggregate to Daily Channel Demand

After creating one row per occupied room night, the data can be aggregated:

```python
daily_channel_demand = (
    room_nights
    .groupby(["stay_date", "code"])
    .size()
    .rename("room_nights")
    .reset_index()
)
```

The resulting structure is approximately:

```text
stay_date   | channel        | room_nights
------------|----------------|------------
2024-01-01  | BookingCom     | 48
2024-01-01  | Direct         | 3
2024-01-01  | Ibe            | 21
2024-01-01  | ChannelManager | 10
2024-01-02  | BookingCom     | 51
...

```
This is now a time-series forecasting dataset.

---

# 9. Why Missing Date × Channel Combinations Matter

Suppose no Direct room nights occurred on a particular day.

The raw aggregation may simply contain no row:

```text
2024-01-05 | Direct | <missing>
```

But for demand forecasting, that means:

```text
2024-01-05 | Direct | 0
```

Missing and zero are not the same thing.

Therefore I constructed the complete combination of:

```text
every date × every channel
```

and filled missing combinations with zero.

```python
full_dates = pd.date_range(
    start=room_nights["stay_date"].min(),
    end=room_nights["stay_date"].max(),
    freq="D"
)

full_index = pd.MultiIndex.from_product(
    [full_dates, channels],
    names=["stay_date", "code"]
)

daily_channel_demand = (
    daily_channel_demand
    .set_index(["stay_date", "code"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)
```
This prevents missing observations from accidentally being interpreted as unknown data rather than zero observed demand.

---

# 10. Realized Demand Is Not Necessarily True Market Demand

This was an important conceptual limitation.

Suppose a property has:

```text
100 rooms available
100 rooms sold
20 additional customers wanted rooms
```

Observed realized demand is:

```text
100 room nights
```

But actual market demand may have been:

```text
120 room nights
```

The extra 20 customers cannot appear in completed reservation data because the property had no inventory left to sell.

This is called **censored demand**.

Therefore:

```text
Observed realized demand
        ≠
Unconstrained market demand
```

To estimate unconstrained demand properly, additional information could be

useful, such as:

- searches,
- availability requests,
- rejected bookings,
- sold-out periods,
- competitor demand,
- inventory availability,
- booking pace.

This means the forecast should be described precisely as a forecast of **realized occupied room nights**, not necessarily total latent market demand.

---

# 11. Interview Explanation

### Question

**Why did you forecast room nights instead of bookings?**

### Short answer

Room nights better represent accommodation inventory consumption. A one-night booking and a ten-night booking both count as one booking, but they have very different operational and commercial impact. I therefore expanded completed reservations across their occupied stay dates and forecast daily realized room nights by channel.

### Deeper answer

I first clarified what "demand" should mean for the business problem. Booking counts, arrivals and room nights represent different things. Because the forecast was intended to support accommodation and revenue decisions, I chose realized room nights because they align directly with when inventory is consumed.

I expanded every completed reservation from arrival until the night before departure, validated that the expansion preserved the original total length of stay, and then aggregated the result by stay date and booking channel.

I would also explicitly distinguish this from unconstrained market demand, because realized bookings can be censored when the property is sold out.

---

# 12. Key Lesson

The first important forecasting lesson was:

> **Do not start by choosing a model. Start by defining exactly what the target represents.**

A sophisticated model predicting the wrong business quantity is less useful than a simple model predicting the right one.

---

# 13. Determine the Grain Before Deduplicating

A column called `booking_id` should not automatically be assumed to uniquely identify each row.

During exploration of the reservation dataset:

- 21,112 reservation rows were present.
- There were 20,067 unique `booking_id` values.
- 651 booking IDs appeared more than once.
- Repeated booking IDs were not generally exact duplicate rows.
- Only 32 rows were exact duplicates.
- Repeated booking IDs could differ in booking timestamp, room type, amount, number of persons, status, arrival date, departure date, and other attributes.

One particularly useful example was a Direct booking ID that appeared many times with the same stay dates but different booking timestamps and room types.

This means that blindly applying:

```python
reservations.drop_duplicates("booking_id")
```
would be unsafe.

The important lesson is:

> An ID-looking column is not necessarily the primary key of the dataset.

Before deduplicating, determine the grain of the data:

> What does one row actually represent?

Without a formal data dictionary or a unique room/unit identifier, the exact business meaning of repeated booking IDs cannot be proven from the available fields alone. Therefore, repeated IDs should not be removed simply because the identifier repeats.

---

# 14. Validate at Multiple Levels
A transformation should not be considered correct merely because the code runs.

For the room-night transformation:
```bash
expected_room_nights = completed_stays["length_of_stay"].sum()

actual_room_nights = len(room_nights)
```
The result was:
```txt
Expected room nights: 51,696

Expanded rows:        51,696
```
This is an important mathematical validation, but it does not prove that the source data itself is correct.

A stronger validation process has several layers:
```text
Structural validation
        |
        v
Are columns, dates, statuses and values sensible?


Mathematical validation
        |
        v
Did the transformation preserve the expected totals?


Business validation
        |
        v
Do the resulting patterns make sense in the business context?
```

The general lesson is:

> Correct totals do not automatically imply a correct dataset.

---

# 15. Do Not Confuse Missing History With Zero Demand

The early history showed incomplete temporal coverage.

Reservation-derived daily demand coverage:
```txt
2022-07   11 observed demand dates
2022-08   13 observed demand dates
2022-09   20 observed demand dates
2022-10   31 observed demand dates
2022-11   30 observed demand dates
2022-12   31 observed demand dates
```
Occupancy data showed a similar pattern:
```txt
2022-07    1 observation
2022-08   25 observations
2022-09   25 observations
2022-10   31 observations
2022-11   30 observations
2022-12   31 observations
```

October 1, 2022 was therefore chosen as the start of the usable historical period.

This does not prove that the source system became perfectly complete on that exact date. It is a modelling assumption supported by the fact that both datasets have continuous daily coverage from the first complete calendar month onward.

The important distinction is:
```txt
Missing observation != zero demand
```
A missing date in an incomplete source period should not automatically be filled with zero.

---

# 16. Build a Complete Date x Channel Grid

Once the trusted historical period was selected, every combination of date and active realized-demand channel was constructed.

The four realized-demand channels were:

* BookingCom
* ChannelManager
* Direct
* Ibe

The usable common historical window contained 619 calendar days.

Therefore:
```txt
619 days x 4 channels = 2,476 rows
```

The complete modelling table contained exactly 2,476 rows and no missing values.

Within this accepted observation window, a missing date-channel combination in the derived room-night table is interpreted as zero realized room nights.

Example:
```txt
2022-10-03   BookingCom       33
2022-10-03   ChannelManager    0
2022-10-03   Direct            1
2022-10-03   Ibe              12
```

This is different from filling missing dates in an incomplete historical period with zero.

---

# 17. Explore the Time Series Before Choosing a Forecasting Model

The first visualization should answer a simple question:
>How has realized demand changed over time?

![Historical daily realized room-night demand](../images/forecasting/01_historical_daily_demand.png)

The daily series shows substantial short-term variation, while the 7-day rolling average makes the underlying demand level easier to see.

The rolling average is:
[
MA_7(t) =
\frac{x_t + x_{t-1} + \dots + x_{t-6}}{7}
]

It is used here for visualization, not as a replacement for the original target.

The historical series suggests that the demand-generating process is not perfectly stationary. Demand generally moved to a higher level after the early part of the usable history.

This matters because:

More historical data is not automatically better forecasting data.

Older observations can become less representative when the business or demand process changes.

---

# 18. Weekly Seasonality

Average realized room-night demand by weekday was:

|Day   | Mean Room Nights        |
|------------|----------------|
|Monday|77.39|
|Tuesday|81.49|
|Wednesday|81.41|
|Thursday|79.31|
|Friday|80.69|
|Saturday|82.35|
|Sunday|77.85|

![Average demand by weekday](../images/forecasting/02_demand_by_weekday.png)

Saturday had the highest average demand, while Monday and Sunday were lower.

However, the difference between the highest and lowest weekday mean was only about five room nights, while the daily standard deviation was generally larger.

Therefore:
> Weekly seasonality exists, but weekday alone does not explain most of the variation in demand.

This means weekday features may be useful, but their value should be tested against simpler forecasting baselines.

---

# 19. Monthly Demand Evolution

Monthly averages reveal both a changing demand level and possible longer-term seasonal effects.

Examples:
```txt
2022-10   65.23
2023-01   69.42
2023-05   84.74
2023-09   87.70
2024-02   86.45
2024-04   87.23
2024-05   85.42
```

![Monthly demand evolution](../images/forecasting/03_monthly_demand_evolution.png)

Demand increased substantially between late 2022 and 2023 and later remained mostly in the 80s.

There are also possible seasonal movements, such as lower demand during summer 2023 followed by stronger September demand.

However, the dataset contains too few complete annual cycles to confidently claim stable yearly seasonality.

The safer interpretation is:
> The series contains a changing demand level, modest weekly seasonality, and possible longer-term seasonal effects.

---

# 20. Partial Periods Must Be Treated Carefully

June 2024 contains only partial observations in the available data.

A partial month should not be compared directly with complete historical months as though both represented the same amount of observation time.

More importantly, if the forecast is intended to begin on June 1, 2024, using observations from June 1 onward to construct that forecast would introduce look-ahead leakage.

The concepts must therefore remain separate:
```txt
Data availability window
        !=
Forecast training window
        !=
Forecast horizon
```

The forecast origin must be defined explicitly before model evaluation.