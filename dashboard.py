import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data():
    if os.path.exists("emotion_data.csv"):
        df = pd.read_csv("emotion_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        st.warning("No data found.")
        return pd.DataFrame(columns=['Date', 'Emotion', 'MusicRecommendation'])

df = load_data()

# Dashboard layout
st.title("Interactive Dashboard")

# Display recent emotions
st.header("Recent Emotions")
if not df.empty:
    st.dataframe(df[['Date', 'Emotion']].tail(10))
else:
    st.write("No recent emotion data.")

# Display music recommendations
st.header("Music Recommendations")
if not df.empty:
    st.dataframe(df[['Date', 'MusicRecommendation']].tail(10))
else:
    st.write("No recent music recommendations.")

# Plot emotion distribution as a pie chart
st.header("Emotion Distribution")
if not df.empty:
    emotion_counts = df['Emotion'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    ax.set_title('Emotion Distribution')
    st.pyplot(fig)
else:
    st.write("No data to show emotion distribution.")

# Plot emotional trends over time
st.header("Emotional Trends Over Time")
if not df.empty:
    # Create a DataFrame for trend analysis
    df.set_index('Date', inplace=True)
    emotion_trend = df.groupby([df.index.date, 'Emotion']).size().unstack().fillna(0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    emotion_trend.plot(ax=ax)
    ax.set_title('Emotional Trends Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.legend(title='Emotion')
    st.pyplot(fig)
else:
    st.write("No data to show trends.")
