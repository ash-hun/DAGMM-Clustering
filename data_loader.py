from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ourDataLoader(Dataset):
	def __init__(self, data_path, mode="train", test_size=0.2):
		self.mode = mode
		data = np.load(data_path, allow_pickle=True)

		minmax = MinMaxScaler()
		features = data["kdd"][:, :-1]
		labels = data["kdd"][:, -1]

		features = minmax.fit_transform(features)
		# Splitting the data into training and testing sets
		num_samples = features.shape[0]
		num_train = int((1 - test_size) * num_samples)

		# Shuffling data indices
		indices = np.arange(num_samples)
		np.random.shuffle(indices)

		# Splitting indices for training and testing
		train_indices, test_indices = indices[:num_train], indices[num_train:]

		# Selecting training/testing data and labels
		self.train_features, self.train_labels = features[train_indices], labels[train_indices]
		self.test_features, self.test_labels = features[test_indices], labels[test_indices]

		# Use the training or testing data/labels depending on the mode
		self.features = self.train_features if mode == "train" else self.test_features
		self.labels = self.train_labels if mode == "train" else self.test_labels

	def __len__(self):
		return self.features.shape[0]

	def __getitem__(self, index):
		return np.float32(self.features[index]), np.float32(self.labels[index])

class KDD99Loader(object):
	def __init__(self, data_path, mode="train"):
		self.mode = mode
		data = np.load(data_path, allow_pickle=True)

		labels = data["kdd"][:, -1]
		features = data["kdd"][:, :-1]
		N, D = features.shape

		normal_data = features[labels == 1]
		normal_labels = labels[labels == 1]

		N_normal = normal_data.shape[0]

		attack_data = features[labels == 0]
		attack_labels = labels[labels == 0]

		N_attack = attack_data.shape[0]

		randIdx = np.arange(N_attack)
		np.random.shuffle(randIdx)
		N_train = N_attack // 2

		self.train = attack_data[randIdx[:N_train]]
		self.train_labels = attack_labels[randIdx[:N_train]]

		self.test = attack_data[randIdx[N_train:]]
		self.test_labels = attack_labels[randIdx[N_train:]]

		self.test = np.concatenate((self.test, normal_data), axis=0)
		self.test_labels = np.concatenate((self.test_labels, normal_labels), axis=0)

	def __len__(self):
		"""
		Number of images in the object dataset.
		"""
		if self.mode == "train":
			return self.train.shape[0]
		else:
			return self.test.shape[0]

	def __getitem__(self, index):
		if self.mode == "train":
			return np.float32(self.train[index]), np.float32(self.train_labels[index])
		else:
			return np.float32(self.test[index]), np.float32(self.test_labels[index])

def get_loader(data_path, batch_size, mode='train'):
	"""Build and return data loader."""

	# dataset = KDD99Loader(data_path, mode)
	dataset = ourDataLoader(data_path, mode)

	shuffle = False
	if mode == 'train':
		shuffle = True

	data_loader = DataLoader(dataset=dataset,
	                         batch_size=batch_size,
	                         shuffle=shuffle)
	return data_loader