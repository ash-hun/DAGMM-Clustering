from data_loader import *
from main import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정

class hyperparams():
	def __init__(self, config):
		self.__dict__.update(**config)

if __name__ == "__main__":
	seed_everything(2)
	data = np.load('./ourData.npz')
	print(pd.DataFrame(data['kdd']).shape)

	defaults = {
		'lr': 1e-4,
		'num_epochs': 200,
		'batch_size': 1024,
		'gmm_k': 9,
		'lambda_energy': 0.1,
		'lambda_cov_diag': 0.005,
		'pretrained_model': None,
		'mode': 'train',
		'use_tensorboard': False,
		'data_path': 'ourData.npz',
		'log_path': './dagmm/logs',
		'model_save_path': './dagmm/models',
		'sample_path': './dagmm/samples',
		'test_sample_path': './dagmm/test_samples',
		'result_path': './dagmm/results',
		'log_step': 194 // 4,
		'sample_step': 194,
		'model_save_step': 194,
	}

	solver = main(hyperparams(defaults))

	solver.data_loader.dataset.mode = "train"

	train_energy = []
	train_labels = []
	train_z = []

	features = data["kdd"][:, :-1]
	labels = data["kdd"][:, -1]

	for it, (input_data, labels) in enumerate(solver.data_loader):
		input_data_s = solver.to_var(torch.tensor(features, dtype=torch.float32))
		enc, dec, z, gamma = solver.dagmm(input_data_s)
		train_z.append(z.data.cpu().numpy())

	train_z = np.concatenate(train_z, axis=0)
	print("="*50)
	print(f' ► Train Z  : {train_z}')
	print("=" * 50)
	print(gamma)
	print(gamma.shape)

	max_indices = torch.argmax(gamma, dim=1)
	df = pd.DataFrame(max_indices.numpy(), columns=['Max_Index'])
	print("=" * 50)
	print(df.value_counts())
	# _, cluster_labels = torch.max(gamma, 1)

	# Plot ===========================================================================================================
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(train_z[:, 1], train_z[:, 0], train_z[:, 2], c=cluster_labels)
	ax.scatter(train_z[:, 1], train_z[:, 0], train_z[:, 2])
	ax.set_xlabel('Encoded')
	ax.set_ylabel('Euclidean')
	ax.set_zlabel('Cosine')
	plt.show()