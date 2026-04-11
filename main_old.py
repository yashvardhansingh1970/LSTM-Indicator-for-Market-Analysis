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
start_date = st.date_input("Select a start date:")

# Create a date input field for the user to select an end date
end_date = st.date_input("Select an end date:")

if start_date <= end_date:
    # Fetch historical stock data using yfinance
    try:
        # Download the historical data
        df = yf.download(user_input, start=start_date, end=end_date)

        # Display the historical data
        st.write(f"Historical data for {user_input} from {start_date} to {end_date}:")
        st.write(df)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.error("End date must be greater than or equal to start date.")
# df = yf.download(user_input, start=start_date, end=end_date)

st.subheader('Closing Price vs Time Chart')
trace = go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Closing Price')

layout = go.Layout(
    title=f"{user_input} Closing Price",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Price (USD)")
)

fig = go.Figure(data=[trace], layout=layout)

# Show the interactive plot in the Jupyter Notebook
st.plotly_chart(fig)

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
    title=f"Closing Price",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Z-Score")
)

fig1 = go.Figure(data=[trace], layout=layout)
st.plotly_chart(fig1)

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
fig2=plt.figure(figsize=(20,12))
ma100 = df.Close.rolling(100).mean()
trace_ma100 = go.Scatter(x=df.index, y=ma100, mode='lines',name='100ma')
trace_close = go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Price')
layout = go.Layout(
    title=f"Closing Price", legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
)
    # xaxis=dict(title="Number of Days "),
    # yaxis=dict(title="Price(USD)")
)

fig2 = go.Figure(data=[trace_close,trace_ma100], layout=layout)
st.plotly_chart(fig2)


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
trace_ma100 = go.Scatter(x=df.index, y=ma100, mode='lines',name='100ma')
trace_ma200 = go.Scatter(x=df.index, y=ma200, mode='lines',name='200ma')
trace_close = go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Price')

layout = go.Layout(
    title=f"Closing Price", legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    )
    # xaxis=dict(title="Price(USD)"),
    # yaxis=dict(title="Number of Days")
)

fig2 = go.Figure(data=[trace_close,trace_ma100, trace_ma200], layout=layout)
st.plotly_chart(fig2)


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
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.h5')

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
scaler = scaler.scale_

scale_factor = 1/0.01399972
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#predicted vs observed plot



st.subheader('Predictions vs Original(LSTM)')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label ='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
