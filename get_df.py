import pandas as pd
import yfinance as yf


# Specify ticker symbols for desired securities
securities = ['SPY', 'DX-Y.NYB', '^VIX']		
df_main = pd.DataFrame()

for security in securities:

	# Get dataframe of prices for security
	df = yf.download(
		   tickers=security,
		   interval='2m',
		   start='2022-04-16',
		   end='2022-06-14',
		   prepost=True)

	# Rename close column to security-specific name to enable column joins 
	df = df.rename(columns={'Close': f'{security}_close'})

	# If df_main has no data, set equal to first df we create. Else, join with existing df_main.
	if len(df_main) == 0:
		df_main = df
	else:
		df_main = df_main.join(df[f'{security}_close'])

# Drop all columns other than close prices
for col in df_main.columns:
	if 'close' not in col:
		df_main = df_main.drop(col, axis=1)

# Drop rows with NaN and save to local csv file
df_main.dropna().to_csv('df.csv')

