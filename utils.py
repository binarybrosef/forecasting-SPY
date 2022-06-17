import tensorflow as tf
import numpy as np



class WindowGenerator():

	def __init__(self, input_width, label_width, shift,
	 			 train_df, val_df, 
	 			 label_columns=None):
		'''
		Initialize WindowGenerator class instance, set instance attributes.

		Arguments
		---------
		input_width: int
			Length of input sequence used to predict target label
		label_width: int
			Length of predicted output sequence. For predicting a single target, label_width=1.
		shift: int
			Temporal offset between target label index and end of input sequence. For predicting
			a target one time step after the end of an input sequence, shift=1.
		label_columns: list of strings
			Names of target label(s) to predict
		'''

		self.train_df = train_df
		self.val_df = val_df

		self.label_columns = label_columns

		# Build dict comprising {label_name: label_index}, where label_name is the (str) name of a target label,
		# and label_index is the integer index of that target label.
		if label_columns is not None:
			self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

		# Build dict comprising {column_name: column_index}, where column_name is the (str) name of a column/feature,
		# and column_index is the integer index of that column/feature.
		self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

		self.input_width = input_width
		self.label_width = label_width
		self.shift = shift

		# For input_width=7, shift=1, total_window_size=8
		self.total_window_size = input_width + shift

		# Get a set of indices at which to access input sequence data
		# For input_width=7, slice(0,7) = (0,1,2,3,4,5,6)
		self.input_slice = slice(0, input_width)

		# for total_window_size=8, np.arange(total_window_size) gives (0,1,2,3,4,5,6,7)
		# input_slice then is the set of indices that control which elements are obtained 
		# from the output of np.arange()
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]

		# For label_width=1, label_start is where the first target label is
		self.label_start = self.total_window_size - self.label_width

		# Specifying slice(start, None) means that indexing will begin with the element at start, and go
		# to the end of whatever is being indexed. Note that start is an index, so for start=8, we start at
		# the ninth element of whatever is being indexed.
		self.labels_slice = slice(self.label_start, None)
		self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

	def split_window(self, features):
		'''
		From a feature set, split input features from label(s).

		Arguments
		---------
		features: tf.data.Dataset
			feature set derived from a DataFrame (e.g., training or validation df)

		Returns
		-------
		inputs: tensor
			input features
		labels: tensor
			labels
		'''

		# features[batch_size, sequence_length, num_features]
		# Thus, the two subsequent lines pick out sequence elements belonging to an input sequence and to target
		# label(s), respectively
		inputs = features[:, self.input_slice, :]
		labels = features[:, self.labels_slice, :]

		# At this point, labels comprises only the timestep(s) for the target label(s), but includes all features
		# The code below uses column_indices to pick out only the feature(s) corresponding to the target label(s).
		# If only predicting SPY_close, the last axis should be of len 1.
		if self.label_columns is not None:
			labels = tf.stack(
				[labels[:, :, self.column_indices[name]] for name in self.label_columns], 
				axis=-1)

		inputs.set_shape([None, self.input_width, None])
		labels.set_shape([None, self.label_width, None])

		return inputs, labels

	
	def make_dataset(self, data):
		'''
		Create tf.data.Dataset from DataFrame. Map split_window() to obtain inputs, labels separately.

		Arguments
		---------
		data: Pandas DataFrame
			DataFrame comprising dataset such as training/validation set

		Returns
		-------
		ds: tf.data.Dataset 
			dataset comprising inputs, labels
		'''

	# timeseries_dataset_from_array() returns a tf.data.Dataset instance comprising sequences.
	# From TF docs: If targets was passed, the dataset yields tuple (batch_of_sequences, batch_of_targets). 
	# If not, the dataset yields only batch_of_sequences. 
		data = np.array(data, dtype=np.float32)
		ds = tf.keras.utils.timeseries_dataset_from_array(
			data=data,
			targets=None,
			sequence_length=self.total_window_size,
			sequence_stride=1,
			shuffle=True,
			batch_size=32)

		ds = ds.map(self.split_window)

		return ds

	@property
	def train(self):
		'''
		Create training set of sequences.

		Returns
		-------
		ds: tf.data.Dataset 
			dataset comprising inputs, labels
		'''

		return self.make_dataset(self.train_df)

	@property
	def val(self):
		'''
		Create validation set of sequences.

		Returns
		-------
		ds: tf.data.Dataset 
			dataset comprising inputs, labels
		'''

		return self.make_dataset(self.val_df)

	@property
	def example(self):
		result = getattr(self, '_example', None)

		if result is None:
			result = next(iter(self.train))
			self._example = result

		return result

	def __repr__(self):
		return '\n'.join([
			f'Total window size: {self.total_window_size}',
			f'Input indices: {self.input_indices}',
			f'Label indices: {self.label_indices}',
			f'Label column name(s): {self.label_columns}'])


class Baseline(tf.keras.Model):
	def __init__(self, label_index=None):
		'''
		Initialize Basline class instance to obtain baseline model, set instance attribute.

		Arguments
		---------
		label_index: int
			index of label(s)
		'''

		super().__init__()
		self.label_index = label_index


	def call(self, inputs):
		'''
		Evaluate baseline model on inputs.

		Arguments
		---------
		inputs: tensorflow.python.data.ops.dataset_ops.MapDataset
			dataset with which to evaluate baseline model

		Returns
		-------
		result: list
			Predicted output from baseline model. 
		'''

		if self.label_index is None:
			return inputs

		result = inputs[:, :, self.label_index]

		return result[:, :, tf.newaxis]