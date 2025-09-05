# streamlit run ex41_dashboard.py
import pandas as pd
import streamlit as st
import plotly.express as px

# Load data from CSV
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_sensor_data.csv")

df = load_data()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Dashboard UI
st.title('Predictive Maintenance Dashboard - Oil & Gas Industry')

# Filters
st.sidebar.header('Filters')
equipment_type = st.sidebar.multiselect('Select Equipment Type', df['Equipment Type'].unique(), default=df['Equipment Type'].unique())
failure_occurred = st.sidebar.multiselect('Select Failure Occurred', df['Failure Occurred'].unique(), default=df['Failure Occurred'].unique())

# Apply filters
filtered_df = df[(df['Equipment Type'].isin(equipment_type)) & (df['Failure Occurred'].isin(failure_occurred))]

# Visualizations
st.subheader('Failure Rate Over Time')
failure_trend = filtered_df.groupby('Timestamp')['Failure Occurred'].sum().reset_index()
fig1 = px.line(failure_trend, x='Timestamp', y='Failure Occurred', title='Failures Over Time')
st.plotly_chart(fig1)

st.subheader('Sensor Readings Distribution')
sensor_columns = ["Temperature", "Pressure", "Vibration", "Humidity", "Flow Rate"]
for sensor in sensor_columns:
    fig2 = px.histogram(filtered_df, x=sensor, color='Failure Occurred', marginal='box', title=f'{sensor} Distribution')
    st.plotly_chart(fig2)

st.subheader('Equipment Type vs Failure')
failure_breakdown = filtered_df.groupby('Equipment Type')['Failure Occurred'].sum().reset_index()
fig3 = px.bar(failure_breakdown, x='Equipment Type', y='Failure Occurred', title='Failure Count by Equipment Type')
st.plotly_chart(fig3)
