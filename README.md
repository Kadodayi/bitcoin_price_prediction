# Bitcoin Price Prediction with NeuralProphet

## Overview

This project aims to predict the hourly price of Bitcoin (BTC/USDT) for the next 24 hours using the NeuralProphet time series forecasting library. The model incorporates various external regressors, including market sentiment (Google Trends), network health (hash rate), and other financial indicators.

The project is structured into three main Python scripts:
1.  `Financial_time_series.py`: Fetches historical hourly Bitcoin price data from the Binance API.
2.  `data_preprocessing_(2).py`: Collects and preprocesses additional features, merges them with Bitcoin data, and performs scaling and transformations.
3.  yav bozmuşsun herşeyi

*   **Bitcoin Price (BTC/USDT):** Binance API (fetched via `requests`).
*   **Google Trends:** `pytrends` library.
*   **Bitcoin Hash Rate:** `api.blockchain.info`.
*   **Additional Financial Data (CSV files):**
    *   Gold: `XAU_USD Geçmiş Verileri (1).csv`
    *   DXY: `ABD Dolar Endeksi Geçmiş Verileri.csv`
    *   Federal Funds Rate: `Federal Fon Bileşik Faiz Oranı Geçmiş Verileri (1).csv` (Note: `fed_close` is dropped before final model training)
    *   BTC Trading Volume: `Btc_trade_volume.csv`
    *   Unique Bitcoin Addresses: `n_unique_addresss.csv`
    *   Miners' Revenue: `Miners_revenue_usdt.csv`
    *   BTC/ETH Correlation: `BTC_ETH Sentetik Geçmiş Verileri.csv`
    *   S&P 500 Index: `S&P 500 Geçmiş Verileri.csv`

    *Note: These CSV files are expected to be in the same directory as the scripts. Their original sources are likely financial data providers like Investing.com.*

## Workflow

The project follows a multi-step process:

1.  **Bitcoin Data Fetching (`Financial_time_series.py`):**
    *   Fetches hourly BTC/USDT k-line data from Binance for a specified date range (default start: Jan 10, 2024, end: current time).
    *   Processes the data into a DataFrame with `ds` (datetime, UTC, localized to None) and `y` (closing price).
    *   This script is intended to produce `basardık_aq.csv` (the `to_csv` line is commented out in the provided script but `data_preprocessing_(2).py` reads this file).

2.  **Data Preprocessing and Feature Engineering (`data_preprocessing_(2).py`):**
    *   Loads the Bitcoin price data (`basardık_aq.csv`).
    *   Fetches Google Trends data for "Bitcoin" (last 2 years, resampled to hourly).
    *   Fetches Bitcoin hash rate data (last 2 years, resampled to hourly).
    *   Merges these initial features with the Bitcoin price data.
    *   Loads additional financial data from various CSV files.
    *   Cleans, parses dates, and standardizes column names for these additional datasets.
    *   Resamples daily/other frequency data to hourly using linear interpolation (`expand_to_hourly` function).
    *   Merges all features into a single DataFrame using `pd.merge_asof` (nearest match within a 1-hour tolerance).
    *   Handles missing values (e.g., using `ffill`).
    *   Applies log transformation to the target variable `y` (Bitcoin price).
    *   Scales all features (Google Trends, hash rate, and other financial indicators) using `RobustScaler`.
    *   Saves the fully processed dataset to `robust_all_ham_veri.csv`.
    *   Includes exploratory data analysis (EDA) like box plots and correlation heatmaps.

3.  **Model Training and Prediction (`bitcoin_price_prediction_(1).py`):**
    *   Loads the preprocessed data from `robust_all_ham_veri.csv`.
    *   Drops the `fed_close` column.
    *   Initializes a NeuralProphet model with specified hyperparameters (see Model Details).
    *   Adds all selected features as lagged regressors to the model.
    *   Splits the data into training and validation sets (80/20 split).
    *   Trains the model using the training data, validating on the validation set, with early stopping enabled.
    *   Generates future predictions for the next 24 hours and historical predictions.
    *   Visualizes:
        *   Training and validation loss curves.
        *   Actual prices vs. predicted prices.
        *   A detailed 24-hour forecast plot with prices converted back from log scale to USD.
        *   Hourly predicted price changes for the 24-hour forecast.
    *   Prints a table of the 24-hour predicted prices (log and USD values).
    *   Calculates and prints the predicted percentage change in Bitcoin price over the next 24 hours.
    *   Outputs final model performance metrics (MAE_val, RMSE_val, Loss_val).

## Key Technologies and Libraries

*   **Python 3.x**
*   **Core Libraries:**
    *   `pandas`: Data manipulation and analysis.
    *   `numpy`: Numerical operations.
    *   `requests`: HTTP requests for API data fetching.
*   **Time Series & Forecasting:**
    *   `neuralprophet`: Core forecasting model.
    *   `pytrends`: Fetching Google Trends data.
*   **Machine Learning:**
    *   `scikit-learn`: For `RobustScaler`.
*   **Visualization:**
    *   `matplotlib`: Plotting.
    *   `seaborn`: Enhanced statistical visualizations.
*   **Date/Time:**
    *   `datetime`

## Setup and Usage

1.  **Prerequisites:**
    *   Python 3.x installed.
    *   `pip` package installer.

2.  **Installation:**
    Open your terminal or command prompt and install the necessary libraries:
    ```bash
    pip install pytrends neuralprophet pandas numpy matplotlib scikit-learn requests seaborn
    ```

3.  **Data Files:**
    *   Ensure all required CSV files for additional financial data (listed under "Data Sources") are present in the same directory as the Python scripts.

4.  **Running the Scripts:**
    Execute the scripts in the following order:

    *   **Step 1: Fetch Bitcoin Data**
        Run `Financial_time_series.py`.
        ```bash
        python Financial_time_series.py
        ```
        *   This script will fetch Bitcoin data. You might need to uncomment the line `df.to_csv("basardık_aq.csv", index=False)` or modify the script to save the output if it's not already doing so in a way that `data_preprocessing_(2).py` can read it (i.e., as `basardık_aq.csv`).
        *   The `start_date` is fixed to January 10, 2024. `end_date` is `datetime.now()`. Adjust if needed.

    *   **Step 2: Preprocess Data and Engineer Features**
        Run `data_preprocessing_(2).py`.
        ```bash
        python data_preprocessing_(2).py
        ```
        *   This script reads `basardık_aq.csv` and the other financial CSVs.
        *   It outputs `robust_all_ham_veri.csv`.

    *   **Step 3: Train Model and Predict**
        Run `bitcoin_price_prediction_(1).py`.
        ```bash
        python bitcoin_price_prediction_(1).py
        ```
        *   This script reads `robust_all_ham_veri.csv`.
        *   It will train the model, generate predictions, display plots, and print metrics.
        *   If running in an environment like Google Colab, the `!pip install` line at the beginning of the script will handle dependencies.

## Model Details

*   **Model:** `NeuralProphet`
*   **Key Hyperparameters:**
    *   `n_forecasts`: 24 (predicts 24 hours ahead)
    *   `n_lags`: 48 (uses 48 previous hours as input for AR-Net)
    *   `ar_layers`: `[8]` (A single hidden layer with 8 neurons for the AR-Net component)
    *   `learning_rate`: 0.0005
    *   `yearly_seasonality`: False (disabled as data might be less than 2 years or to simplify)
    *   `weekly_seasonality`: True
    *   `daily_seasonality`: True
    *   `epochs`: 200 (with early stopping)
*   **Lagged Regressors:** As listed in the "Features Used" section. Each regressor's past values are used to help predict the future target variable.

## Output

The final script (`bitcoin_price_prediction_(1).py`) produces:

*   **Console Output:**
    *   Training progress.
    *   A table comparing model metrics (MAE_val, RMSE_val, Loss_val).
    *   A table with 24-hour predicted Bitcoin prices (log and USD values).
    *   Predicted percentage change in Bitcoin price over the next 24 hours.
*   **Plots:**
    *   Model Loss (Training vs. Validation).
    *   Bitcoin Price Prediction (Actual vs. Predicted historical values).
    *   24-Hour Bitcoin Price Prediction (detailed forecast for the next 24 steps).
    *   Hourly Predicted Price Changes (%).

## File Structure
.
├── Financial_time_series.py # Fetches raw Bitcoin price data
├── data_preprocessing_(2).py # Preprocesses data, engineers features
├── bitcoin_price_prediction_(1).py # Trains model, predicts, and evaluates
├── basardık_aq.csv # Output of Financial_time_series.py (if saved)
├── robust_all_ham_veri.csv # Output of data_preprocessing_(2).py
├── XAU_USD Geçmiş Verileri (1).csv # Gold data (example input)
├── ABD Dolar Endeksi Geçmiş Verileri.csv # DXY data (example input)
├── Federal Fon Bileşik Faiz Oranı Geçmiş Verileri (1).csv # Fed rate data (example input)
├── Btc_trade_volume.csv # BTC volume data (example input)
├── n_unique_addresss.csv # Unique addresses data (example input)
├── Miners_revenue_usdt.csv # Miners revenue data (example input)
├── BTC_ETH Sentetik Geçmiş Verileri.csv # BTC/ETH correlation data (example input)
└── S&P 500 Geçmiş Verileri.csv # S&P 500 data (example input)


## Potential Improvements

*   Extensive hyperparameter tuning for the NeuralProphet model.
*   Exploration of different model architectures or alternative forecasting libraries.
*   Inclusion of more diverse or real-time features (e.g., social media sentiment, order book data).
*   Development of a real-time prediction pipeline for continuous forecasting.
*   More sophisticated handling of missing data or feature engineering techniques.
