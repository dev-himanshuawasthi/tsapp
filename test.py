import streamlit as st
import pandas as pd
import altair as alt

# Load your data
df = pd.read_csv('CaseStudy_FraudIdentification.csv')

# Define your interactive widgets
filter_col = st.sidebar.selectbox("Filter by column:", df.columns)
filter_val = st.sidebar.multiselect("Filter by value:", df[filter_col].unique())

# Filter the data
if filter_val:
    df = df[df[filter_col].isin(filter_val)]

# Define your charts
chart_type = st.selectbox("Select chart type:", ["Line Chart", "Bar Chart", "Scatter Plot"])

if chart_type == "Line Chart":
    chart = alt.Chart(df).mark_line().encode(
        x='date',
        y='value',
        color='category'
    ).interactive()
elif chart_type == "Bar Chart":
    chart = alt.Chart(df).mark_bar().encode(
        x='date',
        y='value',
        color='category'
    ).interactive()
else:
    chart = alt.Chart(df).mark_circle().encode(
        x='date',
        y='value',
        color='category'
    ).interactive()

# Show your dashboard
st.title("My Interactive Dashboard")
st.write("Here's some data to explore:")
st.write(df)

st.write("Here's a chart to explore:")
st.altair_chart(chart, use_container_width=True)
