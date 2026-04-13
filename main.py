import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from keras.models import load_model  # Corrected import statement
import streamlit as st
import plotly.graph_objs as go

# start_date = '2018-08-10'
# end_date = '2023-09-06'

st.title('Stock Trend Prediction')
#user_input
user_input = st.text_input('Enter Stock Ticker','AAPL')
start_date = st.date_input("Select a start date:", value=pd.to_datetime('2018-08-10').date())

# Create a date input field for the user to select an end date
end_date = st.date_input("Select an end date:", value=pd.to_datetime('2023-09-06').date())

df = None  # Initialize df variable

if start_date <= end_date:
    # Fetch historical stock data using yfinance
    try:
        # Download the historical data
        df = yf.download(user_input, start=start_date, end=end_date)
        
        # Check if data was actually downloaded
        if df.empty:
            st.error(f"No data found for {user_input} between {start_date} and {end_date}. Please check the ticker symbol and date range.")
            st.stop()
        
        # Display the historical data
        st.write(f"Historical data for {user_input} from {start_date} to {end_date}:")
        st.write(df)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()
else:
    st.error("End date must be greater than or equal to start date.")
    st.stop()

# Only proceed if we have data
if df is None or df.empty:
    st.error("No data available to process.")
    st.stop()

st.subheader('Closing Price vs Time Chart')

# Create matplotlib figure for closing price
fig_close, ax_close = plt.subplots(figsize=(12, 6))
ax_close.plot(df.index, df['Close'], color='blue', linewidth=1)
ax_close.set_title(f'{user_input} Closing Price')
ax_close.set_xlabel('Date')
ax_close.set_ylabel('Price (USD)')
ax_close.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig_close)

# df.reset_index(inplace=True)
# df.head()

#Z-score plot
st.subheader('Z-Score of log return Standardization')

# calculate the log-returns
df['Log-Return'] = np.log(df['Close'] / df['Close'].shift(1))

# calculate the mean and variance
mean = np.mean(df['Log-Return'])
variance = np.var(df['Log-Return'])
std_dev = np.sqrt(variance)

# normalize to z-score
df['Z-Score'] = (df['Log-Return'] - mean) / std_dev

first_close_price = df.iloc[0]['Close']
last_close_price = df.iloc[-1]['Close']
percentage_increase = (last_close_price) / first_close_price * 100


# plot the results
fig, ax = plt.subplots(figsize=(20, 8))
sns.set_style('whitegrid')

# ax.plot(df['Date'], df['Z-Score'])
# ax.set_title('Z-Score of Log-Returns (Standardisation)',fontsize=18)
# ax.set_xlabel('Date',fontsize=18)
# ax.set_ylabel('Z-Score',fontsize=18)
# st.pyplot(fig)
trace = go.Scatter(x=df.index, y=df["Z-Score"], mode='lines', name='Z-Score of Log-Returns (Standardisation')
layout = go.Layout(
    title=f"{user_input} Z-Score of Log-Returns (Standardisation)",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Z-Score")
)

fig1 = go.Figure(data=[trace], layout=layout)
st.plotly_chart(fig1, use_container_width=True)

#metric information
st.subheader('Mean, Variance, Std Deviation & Percentage Increase')
# st.write(df.head())
st.write("Mean = " + str(mean))
st.write("Variance = " +str(variance))
st.write("Std Deviation = " +str(std_dev))
st.write("Percentage Increase = "+str(percentage_increase)+"%")

# st.subheader('Closing Price vs Time Chart ')
# fig=plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# st.pyplot(fig)
df.reset_index(inplace=True)
df.head()

#100ma plot
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df['Date'], df['Close'], label='Close Price', color='blue')
ax2.plot(df['Date'], ma100, label='100MA', color='orange')
ax2.set_title(f'{user_input} Closing Price with 100-Day Moving Average')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (USD)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)


# st.subheader('Closing Price vs Time Chart with 100MA')
# ma100 = df.Close.rolling(100).mean()
# fig=plt.figure(figsize=(12,6))
# plt.plot(ma100)
# plt.plot(df.Close)
# st.pyplot(fig)


#200ma plot
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(df['Date'], df['Close'], label='Close Price', color='blue')
ax3.plot(df['Date'], ma100, label='100MA', color='orange')
ax3.plot(df['Date'], ma200, label='200MA', color='red')
ax3.set_title(f'{user_input} Closing Price with 100MA & 200MA')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price (USD)')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)


# st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()
# fig=plt.figure(figsize=(12,6))
# plt.plot(ma100)
# plt.plot(ma200)
# plt.plot(df.Close)
# st.pyplot(fig)

#training data and testing data creating the model
data_training =pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Custom object scope to handle deprecated parameters
def custom_lstm(**kwargs):
    # Remove deprecated parameters
    kwargs.pop('time_major', None)
    return tf.keras.layers.LSTM(**kwargs)

# Load model with custom objects
try:
    model = load_model("keras_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("The model may be incompatible with the current TensorFlow version. Please retrain the model or use a compatible version.")
    st.stop()

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler_scale = scaler.scale_

scale_factor = 1/scaler_scale[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#predicted vs observed plot
st.subheader('Predictions vs Original(LSTM)')

# Create time index for predictions
time_index = list(range(len(y_test)))

# Create Plotly traces
trace_original = go.Scatter(x=time_index, y=y_test.flatten(), mode='lines', name='Original Price', line=dict(color='blue'))
trace_predicted = go.Scatter(x=time_index, y=y_predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='red'))

layout = go.Layout(
    title=f"{user_input} LSTM Predictions vs Original Prices",
    xaxis=dict(title="Time"),
    yaxis=dict(title="Price"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig_pred = go.Figure(data=[trace_original, trace_predicted], layout=layout)
st.plotly_chart(fig_pred, use_container_width=True)
