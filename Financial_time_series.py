# coding: utf-8
"""
Created on Sat Mar 1 06:23:44 2025

@author: merto
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import requests

def fetch_bitcoin_data_hourly(start_date, end_date):
    base_url = "https://api.binance.com/api/v3/klines"
    symbol = "BTCUSDT"
    interval = "1h"
    limit = 1000

    all_data = []
    current_start = start_date.replace(tzinfo=timezone.utc)  # UTC bazlı başlangıç

    while current_start < end_date.replace(tzinfo=timezone.utc):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(end_date.timestamp() * 1000)
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if not data:
            break

        df_chunk = pd.DataFrame(data, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume", 
            "Close time", "Quote asset volume", "Number of trades", 
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ])
        all_data.append(df_chunk)

        # Sonraki başlangıç zamanını UTC'ye göre güncelle
        last_timestamp = df_chunk["Open time"].iloc[-1]
        current_start = datetime.utcfromtimestamp(last_timestamp / 1000).replace(tzinfo=timezone.utc) + timedelta(hours=1)

    if not all_data:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.concat(all_data, ignore_index=True)
    # Zaman dilimi bilgisi kaldırıldı
    df["ds"] = pd.to_datetime(df["Open time"], unit="ms", utc=True).dt.tz_localize(None)  # <button class="citation-flag" data-index="7">
    df["y"] = df["Close"].astype(float)
    return df[["ds", "y"]]

# UTC tarih aralığı (zaman dilimi belirtilmedi)
start_date = datetime(2024, 1, 10, 0, 0, 0)  # 10 Ocak 2024 00:00:00 UTC <button class="citation-flag" data-index="3">
end_date = datetime.now()      # 5 Mart 2025 00:00:00 UTC <button class="citation-flag" data-index="5">

df = fetch_bitcoin_data_hourly(start_date, end_date)
#df.drop(range(10033, 10209), inplace=True)
if not df.empty:
    print(f"Çekilen veri aralığı: {df['ds'].min()} - {df['ds'].max()}")
    #df.to_csv("basardık_aq.csv", index=False)
else:
    print("Belirtilen tarih aralığında veri bulunamadı.")

    



