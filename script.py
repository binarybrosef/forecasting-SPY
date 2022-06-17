import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from utils import *



def plot_features(df):
	'''
	Display violin plot of features in input DataFrame.

	Arguments
	---------
	df: Pandas DataFrame
		DataFrame comprising data to be plotted.
	'''

	df_sns = df.melt(var_name='Column', value_name='Normalized')
	plt.figure(figsize=(12,6))
	ax = sns.violinplot(x='Column', y='Normalized', data=df_sns)
	plt.show()


def compile_and_fit(model, window, epochs=50):
	'''
	Compile input model and train on data provided by window.
	
	Arguments
	---------
	model: Keras model
		Model used to forecast target variable. By default, an LSTM-based RNN.
	window: WindowGenerator class instance
		Provides dataset comprising sequences 
	epochs: int
		Number of epochs to train model
	'''

	model.compile(loss=tf.losses.MeanSquaredError(),
				  optimizer=tf.optimizers.Adam(),
				  metrics=[tf.metrics.MeanAbsoluteError()])

	tensorboard = TensorBoard(log_dir='logs')

	history = model.fit(window.train, 
						epochs=epochs,
						validation_data=window.val,
						callbacks=[tensorboard])


def preprocess_df(df, scaling='standard', features='raw', split=0.9):
	'''
	Scale and transform features to produce training and validation set DataFrames.

	Arguments
	---------
	df: Pandas DataFrame
		Full dataset comprising all training and validation data
	scaling: string
		'standard' applies standardization to values
		'minmax' applies minmax scaling to values
		None does not apply any scaling to values
	features: string
		'raw' results in not further processing features beyond scaling
		'percent' converts features into their percent change relative to last observation
		'diff' converts features into their subtractive difference relative to last observation
	split: float
		Ratio of training data to validation data

	Returns
	-------
	train_df: Pandas DataFrame
		Training data set
	val_df: Pandas DataFrame
		Validation data set
	'''

	columns = ['DX-Y.NYB_close', '^VIX_close', 'hour_sin', 
			   'hour_cos', 'day_sin', 'day_cos', 'SPY_close']

	securities = ['DX-Y.NYB_close', '^VIX_close', 'SPY_close']

	# Set index to Datetime and rearrange such that SPY_close is last column
	df = df.set_index('Datetime')
	df = df[columns]

	# Apply feature transformation to securities, not periodic features
	if features == 'percent':
		for security in securities:
			df[security] = df[security].pct_change()			
			df = df[1:]										# drop first row now that it comprises NaNs

	elif features == 'diff':
		for security in securities:
			df[security] = df[security].diff()					
			df = df[1:]										# drop first row now that it comprises NaNs

	elif features == 'log':
		for security in securities:
			df[security] = np.log(np.abs(df[security]))

	n = len(df)
	train_df = df[0:int(n*split)]		
	val_df = df[int(n*split):]			

	if scaling == 'standard':
		scaler = StandardScaler()
	elif scaling == 'minmax':
		scaler = MinMaxScaler()

	if scaling != None:

		train_df = scaler.fit_transform(train_df)
		val_df = scaler.transform(val_df)

	train_df = pd.DataFrame(train_df, columns=columns)
	val_df = pd.DataFrame(val_df, columns=columns)

	return train_df, val_df


# Columns: ['SPY_close', 'DX-Y.NYB_close', '^VIX_close']
df = pd.read_csv('df.csv')

# Compute time delta between each row
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['time_delta'] = df['Datetime'] - df['Datetime'].shift(periods=1)		# each element of 'time_delta' is of type pandas._libs.tslibs.timedeltas.Timedelta

# Remove rows that have time delta > 2 min
delta = pd.Timedelta('0 days 00:02:00')
df = df[df['time_delta'] == delta]

# Create timestamp feature for subsequent creation of periodic features
df['time'] = df['Datetime'].map(pd.Timestamp.timestamp)

hour = 60*60		# seconds/hour
day = hour*60		# seconds/2.5 days

# Create periodic features with sin/cos functions
df['hour_sin'] = np.sin(df['time'] * (2 * np.pi / hour))
df['hour_cos'] = np.cos(df['time'] * ((2 * np.pi) / hour))
df['day_sin'] = np.sin(df['time'] * ((2 * np.pi) / day))
df['day_cos'] = np.cos(df['time'] * ((2 * np.pi) / day))

# Drop time columns
df.drop(['time', 'time_delta'] , axis=1, inplace=True)

# Apply feature scaling and transformation to df to generate training and validation dfs
train_df, val_df = preprocess_df(df, scaling='minmax', features='diff')

# Display violin plot of features in training df
plot_features(train_df)

column_indices = {name: i for i, name in enumerate(train_df.columns)}

# Generate sequences of length 7 from training and validation dfs
# (input_width, label_width, shift, train_df, val_df, label_columns=None)
window = WindowGenerator(7, 1, 1, train_df, val_df, label_columns=['SPY_close'])

# Build RNN  
model = tf.keras.models.Sequential([
	tf.keras.layers.LSTM(32),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dense(1, activation=None)])

# Train RNN 
compile_and_fit(model, window, epochs=50)

# Evaluate baseline model against validation set
baseline = Baseline(label_index=column_indices['SPY_close'])
baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
				 metrics=[tf.keras.metrics.MeanAbsoluteError()])
baseline.evaluate(window.val)