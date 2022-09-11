import pandas as pd
import yfinance as yf

# Get dataframe of closing prices for SPY
df = yf.download(tickers='SPY',
		 interval='1d',
		 period='2y',
		 prepost=True)

# Drop all columns other than close prices
for col in df.columns:
	if col != 'Close':
		df.drop(col, axis=1, inplace=True)

# Create the daily arithmatic change in closing price ('delta'), and the log of the daily arithmatic change in closing price ('log_delta')
df['delta'] = df['Close'].diff()
df['log_delta'] = np.log10(df['Close']).diff()

# Drop first row as no delta can be computed for it
df = df[1:]

# Save dataframe to .csv file
df.to_csv('df_daily_2yr.csv')
