from datetime import datetime
import streamlit as st
import altair as alt
import vega_datasets


full_df = vega_datasets.data("seattle_weather")

st.set_page_config(
    # Title and icon for the browser's tab bar:
    page_title="Seattle Weather",
    page_icon="🌦️",
    # Make the content take up the width of the page:
    layout="wide",
)


"""
# :blue[:material/partly_cloudy_day:] Seattle Weather

Let's explore the [classic Seattle Weather
dataset](https://altair-viz.github.io/case_studies/exploring-weather.html)!
"""

""  # Add a little vertical space. Same as st.write("").

"""
### 2015 Summary
"""

df_2015 = full_df[full_df["date"].dt.year == 2015]
df_2014 = full_df[full_df["date"].dt.year == 2014]

max_temp_2015 = df_2015["temp_max"].max()
max_temp_2014 = df_2014["temp_max"].max()

min_temp_2015 = df_2015["temp_min"].min()
min_temp_2014 = df_2014["temp_min"].min()

max_wind_2015 = df_2015["wind"].max()
max_wind_2014 = df_2014["wind"].max()

min_wind_2015 = df_2015["wind"].min()
min_wind_2014 = df_2014["wind"].min()

max_prec_2015 = df_2015["precipitation"].max()
max_prec_2014 = df_2014["precipitation"].max()

min_prec_2015 = df_2015["precipitation"].min()
min_prec_2014 = df_2014["precipitation"].min()

cols = st.columns(2)

with cols[0].container(border=True):
    inner_cols = st.columns(2)

    with inner_cols[0]:
        st.metric(
            "Max tempearture",
            f"{max_temp_2015:0.1f}C",
            delta=f"{max_temp_2015 - max_temp_2014:0.1f}C",
        )

    with inner_cols[1]:
        st.metric(
            "Min tempearture",
            f"{min_temp_2015:0.1f}C",
            delta=f"{min_temp_2015 - min_temp_2014:0.1f}C",
        )

with cols[1].container(border=True):
    inner_cols = st.columns(2)

    with inner_cols[0]:
        st.metric(
            "Max precipitation",
            f"{max_prec_2015:0.1f}C",
            delta=f"{max_prec_2015 - max_prec_2014:0.1f}C",
        )

    with inner_cols[1]:
        st.metric(
            "Min precipitation",
            f"{min_prec_2015:0.1f}C",
            delta=f"{min_prec_2015 - min_prec_2014:0.1f}C",
        )

cols = st.columns(2)

with cols[0].container(border=True):
    inner_cols = st.columns(2)

    with inner_cols[0]:
        st.metric(
            "Max wind",
            f"{max_wind_2015:0.1f}m/s",
            delta=f"{max_wind_2015 - max_wind_2014:0.1f}m/s",
        )

    with inner_cols[1]:
        st.metric(
            "Min wind",
            f"{min_wind_2015:0.1f}m/s",
            delta=f"{min_wind_2015 - min_wind_2014:0.1f}m/s",
        )

with cols[1].container(border=True, height="stretch"):
    inner_cols = st.columns(2)

    weather_icons = {
        "sun": "☀️",
        "snow": "☃️",
        "rain": "💧",
        "fog": "😶‍🌫️",
        "drizzle": "🌧️",
    }

    with inner_cols[0]:
        st.metric(
            "Most common weather",
            weather_icons[
                full_df["weather"].value_counts().head(1).reset_index()["weather"][0]
            ],
        )

    with inner_cols[1]:
        st.metric(
            "Least common weather",
            weather_icons[
                full_df["weather"].value_counts().tail(1).reset_index()["weather"][0]
            ],
        )

""

"""
### Details
"""

YEARS = full_df["date"].dt.year.unique()
selected_years = st.multiselect("Years to show", YEARS, default=YEARS)

if not selected_years:
    st.warning("You must select at least 1 year.", icon=":material/warning:")

df = full_df[full_df["date"].dt.year.isin(selected_years)]

cols = st.columns([3, 1])

with cols[0].container(border=True, height="stretch"):
    "##### Temperature"

    st.altair_chart(
        alt.Chart(df)
        .mark_bar(width=1)
        .encode(
            alt.X("date", timeUnit="monthdate").title("date"),
            alt.Y("temp_max").title("temperature range (C)"),
            alt.Y2("temp_min"),
            alt.Color("date:N", timeUnit="year").title("year"),
            alt.XOffset("date:N", timeUnit="year"),
        )
        .configure_legend(orient="bottom")
    )

with cols[1].container(border=True, height="stretch"):
    "##### Weather distribution"

    st.altair_chart(
        alt.Chart(df)
        .mark_arc()
        .encode(
            alt.Theta("count()"),
            alt.Color("weather:N"),
        )
        .configure_legend(orient="bottom")
    )


cols = st.columns(2)

with cols[0].container(border=True, height="stretch"):
    "##### Wind"

    st.altair_chart(
        alt.Chart(df)
        .transform_window(
            avg_wind="mean(wind)",
            std_wind="stdev(wind)",
            frame=[0, 14],
            groupby=["monthdate(date)"],
        )
        .mark_line(size=1)
        .encode(
            alt.X("date", timeUnit="monthdate").title("date"),
            alt.Y("avg_wind:Q").title("average wind past 2 weeks (m/s)"),
            alt.Color("date:N", timeUnit="year").title("year"),
        )
        .configure_legend(orient="bottom")
    )

with cols[1].container(border=True, height="stretch"):
    "##### Precipitation"

    st.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("date:N", timeUnit="month").title("date"),
            alt.Y("precipitation:Q").aggregate("sum").title("precipitation (mm)"),
            alt.Color("date:N", timeUnit="year").title("year"),
        )
        .configure_legend(orient="bottom")
    )

cols = st.columns(2)

with cols[0].container(border=True, height="stretch"):
    "##### Monthly weather breakdown"
    ""

    st.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("month(date):O", title="month"),
            alt.Y("count():Q", title="days").stack("normalize"),
            alt.Color("weather:N"),
        )
        .configure_legend(orient="bottom")
    )

with cols[1].container(border=True, height="stretch"):
    "##### Raw data"

    st.dataframe(df)
