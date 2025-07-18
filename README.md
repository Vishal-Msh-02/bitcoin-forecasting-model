📈 Bitcoin Price Forecasting App using Prophet

A Streamlit web application that forecasts Bitcoin prices using Facebook's open-source Prophet time series forecasting model. The app allows you to explore historical Bitcoin data, view 90-day price forecasts, and analyze seasonal components like weekly and yearly trends.

🚀 Features
📊 Visualizes historical Bitcoin prices

🤖 Forecasts the next 90 days using Prophet

🔍 Displays forecast components: trend, seasonality, holidays

📏 Evaluation using MAE and RMSE (on last 90 days)

📥 Downloadable forecast results as CSV

🔄 Automatically refreshes on data change

📦 Tech Stack
Technology	Purpose
Python	Programming language
Streamlit	Web app framework
Prophet	Time series forecasting
Pandas	Data manipulation
Plotly	Interactive plots
Scikit-learn	Evaluation metrics (MAE, RMSE)

📁 Dataset
Source: Historical Bitcoin prices (CSV file)

Columns Required:

Date: Daily timestamp (converted to Prophet format ds)

Close: Closing price (converted to Prophet format y)

Make sure your CSV file is named bitcoin_history.csv and located in the same directory as app.py.

📦 Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
streamlit run app.py

🖼 App Preview
Screenshots of the app can go here (forecast graph, components plot, evaluation metrics display, etc.)
<img width="1841" height="869" alt="image" src="https://github.com/user-attachments/assets/f5cb86b7-db95-4e72-9dd8-56a3f42e9190" />
<img width="1838" height="854" alt="image" src="https://github.com/user-attachments/assets/6e90a26d-6857-4999-81b2-027df99ceb49" />



📏 Model Evaluation
The model is evaluated using the last 90 days of historical data.

Metric	Value
MAE	Mean Absolute Error
RMSE	Root Mean Squared Error

These help you understand how accurate the Prophet model is on recent data.

📥 Export
You can export the forecast (date, predicted price, upper/lower bounds) as a CSV directly from the app.

🧠 Future Improvements
Allow user to upload custom Bitcoin datasets

Add more forecasting models (e.g., ARIMA, LSTM)

Show interactive tuning of Prophet hyperparameters

Add holiday effects (e.g., weekends, events)

🧑‍💻 Author
Vishal Maheshwary
