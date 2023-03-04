import torch
import torch.utils.data as data
import numpy as np
import os

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def rotate_pc(batch_pc):
	"""rotate point cloud(pytorch tensor) by a small anngle"""

	for i in range(batch_pc.shape[0]):
		rotation_angle = np.random.uniform()*2*np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = torch.from_numpy(np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]],dtype=np.float32))
		# print(torch.reshape(batch_pc[i,:,:], [-1,3]).shape)
		# print(rotation_matrix.shape)
		tmp_pc = torch.mm(torch.reshape(batch_pc[i,:,:], [-1,3]),rotation_matrix)
		# print(batch_pc[i,:,:].shape)
		# print(tmp_pc.shape)
		batch_pc[i,:,:] = tmp_pc
	return batch_pc

def rotate_2pc(batch_pc1, batch_pc2):
	"""rotate point cloud(pytorch tensor) by a small anngle"""

	for i in range(batch_pc1.shape[0]):
		rotation_angle = np.random.uniform()*2*np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = torch.from_numpy(np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]],dtype=np.float32))
		tmp_pc1 = torch.mm(torch.reshape(batch_pc1[i,:,:], [-1,3]),rotation_matrix)
		batch_pc1[i,:,:] = tmp_pc1
		tmp_pc2 = torch.mm(torch.reshape(batch_pc2[i,:,:], [-1,3]),rotation_matrix)
		batch_pc2[i,:,:] = tmp_pc2
	return batch_pc1, batch_pc2

def jitter_pc(batch_pc, sigma=0.01, clip=0.05):
	"""per point jittering"""
	B,N,C = batch_pc.shape
	jittered_data = np.clip(sigma*np.random.randn(B,N,C), -1*clip, clip)
	jittered_data = torch.from_numpy(jittered_data)
	jittered_data += batch_pc
	return batch_pc

def jitter_2pc(batch_pc1, batch_pc2, sigma=0.01, clip=0.05):
	"""per point jittering"""
	B,N1,C = batch_pc1.shape
	_,N2,_ = batch_pc2.shape
	jittered_data = np.clip(sigma*np.random.randn(B,1,C), -1*clip, clip, dtype=np.float32)
	jittered_data1 = torch.from_numpy(jittered_data).repeat(1,N1,1)
	jittered_data2 = torch.from_numpy(jittered_data).repeat(1,N2,1)
	batch_pc1 += jittered_data1
	batch_pc2 += jittered_data2
	return batch_pc1, batch_pc2

def scale_2pc(batch_pc1, batch_pc2):
	B,N1,C = batch_pc1.shape
	_,N2,_ = batch_pc2.shape
	s = np.random.rand(B, 1, 1)*0.4+0.7
	s = torch.from_numpy(s.astype(np.float32))
	s1 = s.repeat(1,N1,1)
	s2 = s.repeat(1,N2,1)
	batch_pc1 = torch.mul(batch_pc1, s1)
	batch_pc2 = torch.mul(batch_pc2, s2)
	return batch_pc1, batch_pc2


class xyzDataSet_gt_nor(data.Dataset):
	def __init__(self, dataset_path, type_num, splits, AorB):
		self.AorB = AorB
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.%s.xyz'%(type_num,scene_num,self.AorB))
		gt = np.loadtxt(gt_path).astype(np.float32)
		partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_%s.xyz'%(type_num,scene_num,scan_num,self.AorB))
		partial = np.loadtxt(partial_path).astype(np.float32)
		partial = torch.from_numpy(partial).permute(1,0)
		gt = torch.from_numpy(gt)
		return partial, gt
	def __len__(self):
		return self.data_pairs.shape[0]


class xyzDataSet_partial_nor(data.Dataset):
	def __init__(self, dataset_path, type_num, splits, AorB):
		self.AorB = AorB
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.%d_%s.xyz'%(type_num,scene_num,scan_num,self.AorB))
		gt = np.loadtxt(gt_path).astype(np.float32)
		partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_%s.xyz'%(type_num,scene_num,scan_num,self.AorB))
		partial = np.loadtxt(partial_path).astype(np.float32)
		partial = torch.from_numpy(partial).permute(1,0)
		gt = torch.from_numpy(gt)
		return partial, gt
	def __len__(self):
		return self.data_pairs.shape[0]


class xyzDataSet(data.Dataset):
	def __init__(self, dataset_path, type_num, splits, AorB):
		self.AorB = AorB
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.%s.xyz'%(type_num,scene_num,self.AorB))
		gt = np.loadtxt(gt_path).astype(np.float32)
		partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_%s.xyz'%(type_num,scene_num,scan_num,self.AorB))
		partial = np.loadtxt(partial_path).astype(np.float32)
		partial = torch.from_numpy(partial).permute(1,0)
		gt = torch.from_numpy(gt)
		return partial, gt

	def __len__(self):
		return self.data_pairs.shape[0]

class xyzDataSet_test(data.Dataset):
	def __init__(self, dataset_path, type_num, splits, AorB):
		self.AorB = AorB
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.%s.xyz'%(type_num,scene_num,self.AorB))
		gt = np.loadtxt(gt_path).astype(np.float32)
		partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_%s.xyz'%(type_num,scene_num,scan_num,self.AorB))
		partial = np.loadtxt(partial_path).astype(np.float32)
		partial = torch.from_numpy(partial).permute(1,0)
		gt = torch.from_numpy(gt)
		return partial, gt, type_num, scene_num, scan_num

	def __len__(self):
		return self.data_pairs.shape[0]


class co_xyzDataSet_2partial_nor(data.Dataset):
	def __init__(self, dataset_path, type_num, splits):
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		A_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.%d_a.xyz'%(type_num,scene_num,scan_num))
		B_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.%d_b.xyz'%(type_num,scene_num,scan_num))
		A_gt = np.loadtxt(A_gt_path).astype(np.float32)
		B_gt = np.loadtxt(B_gt_path).astype(np.float32)

		A_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_a.xyz'%(type_num,scene_num,scan_num))
		B_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_b.xyz'%(type_num,scene_num,scan_num))
		A_partial = np.loadtxt(A_partial_path).astype(np.float32)
		B_partial = np.loadtxt(B_partial_path).astype(np.float32)

		partial = torch.from_numpy(np.concatenate((A_partial,B_partial),axis=0)).permute(1,0)
		gt = torch.from_numpy(np.concatenate((A_gt,B_gt),axis=0)).permute(1,0)#[n,3]->[3,n]

		return partial, gt
	
	def __len__(self):
		return self.data_pairs.shape[0]


class co_xyzDataSet(data.Dataset):
	def __init__(self, dataset_path, type_num, splits):
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		A_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.a.xyz'%(type_num,scene_num))
		B_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.b.xyz'%(type_num,scene_num))
		A_gt = np.loadtxt(A_gt_path).astype(np.float32)
		B_gt = np.loadtxt(B_gt_path).astype(np.float32)

		A_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_a.xyz'%(type_num,scene_num,scan_num))
		B_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_b.xyz'%(type_num,scene_num,scan_num))
		A_partial = np.loadtxt(A_partial_path).astype(np.float32)
		B_partial = np.loadtxt(B_partial_path).astype(np.float32)

		partial = torch.from_numpy(np.concatenate((A_partial,B_partial),axis=0)).permute(1,0)
		gt = torch.from_numpy(np.concatenate((A_gt,B_gt),axis=0)).permute(1,0)#[n,3]->[3,n]

		return partial, gt
	
	def __len__(self):
		return self.data_pairs.shape[0]


class co_xyzDataSet_test(data.Dataset):
	def __init__(self, dataset_path, type_num, splits):
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d_test.txt"%splits[0])).astype(np.int).reshape((-1,3))
		# self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"sss.txt")).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d_test.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		A_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.a.xyz'%(type_num,scene_num))
		B_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.b.xyz'%(type_num,scene_num))
		A_gt = np.loadtxt(A_gt_path).astype(np.float32)
		B_gt = np.loadtxt(B_gt_path).astype(np.float32)

		A_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_a.xyz'%(type_num,scene_num,scan_num))
		B_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_b.xyz'%(type_num,scene_num,scan_num))
		A_partial = np.loadtxt(A_partial_path).astype(np.float32)
		B_partial = np.loadtxt(B_partial_path).astype(np.float32)

		partial = torch.from_numpy(np.concatenate((A_partial,B_partial),axis=0)).permute(1,0)
		gt = torch.from_numpy(np.concatenate((A_gt,B_gt),axis=0)).permute(1,0)#[n,3]->[3,n]

		return partial, gt, type_num, scene_num, scan_num
	
	def __len__(self):
		return self.data_pairs.shape[0]

class co_xyzDataSet_test_value(data.Dataset):
	def __init__(self, dataset_path, type_num, splits):
		self.dataset_path = dataset_path
		self.data_pairs = np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(dataset_path,"type%d"%type_num,"fold%d.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		A_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.a.xyz'%(type_num,scene_num))
		B_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.b.xyz'%(type_num,scene_num))
		A_gt = np.loadtxt(A_gt_path).astype(np.float32)
		B_gt = np.loadtxt(B_gt_path).astype(np.float32)

		A_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_a.xyz'%(type_num,scene_num,scan_num))
		B_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_b.xyz'%(type_num,scene_num,scan_num))
		A_partial = np.loadtxt(A_partial_path).astype(np.float32)
		B_partial = np.loadtxt(B_partial_path).astype(np.float32)

		partial = torch.from_numpy(np.concatenate((A_partial,B_partial),axis=0)).permute(1,0)
		gt = torch.from_numpy(np.concatenate((A_gt,B_gt),axis=0)).permute(1,0)#[n,3]->[3,n]

		return partial, gt, type_num, scene_num, scan_num
	
	def __len__(self):
		return self.data_pairs.shape[0]


class co_xyzDataSet_test_selected(data.Dataset):
	def __init__(self, dataset_path, data_pairs_path, type_num, splits):
		self.dataset_path = dataset_path
		self.data_pairs_path = data_pairs_path
		self.data_pairs = np.loadtxt(os.path.join(self.data_pairs_path,"type%d"%type_num,"fold%d_test.txt"%splits[0])).astype(np.int).reshape((-1,3))
		for i in range(1,len(splits)):
			self.data_pairs = np.concatenate((self.data_pairs,
												np.loadtxt(os.path.join(self.data_pairs_path,"type%d"%type_num,"fold%d_test.txt"%splits[i])).astype(np.int).reshape((-1,3))), axis=0)
	
	def __getitem__(self, index):
		type_num = self.data_pairs[index][0]
		scene_num = self.data_pairs[index][1]
		scan_num = self.data_pairs[index][2]

		A_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.a.xyz'%(type_num,scene_num))
		B_gt_path = os.path.join(self.dataset_path, "type%d"%type_num, 'gt', 'scene%d.%d.b.xyz'%(type_num,scene_num))
		A_gt = np.loadtxt(A_gt_path).astype(np.float32)
		B_gt = np.loadtxt(B_gt_path).astype(np.float32)

		A_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_a.xyz'%(type_num,scene_num,scan_num))
		B_partial_path = os.path.join(self.dataset_path, "type%d"%type_num, 'partial', 'scene%d.%d.%d_b.xyz'%(type_num,scene_num,scan_num))
		A_partial = np.loadtxt(A_partial_path).astype(np.float32)
		B_partial = np.loadtxt(B_partial_path).astype(np.float32)

		partial = torch.from_numpy(np.concatenate((A_partial,B_partial),axis=0)).permute(1,0)
		gt = torch.from_numpy(np.concatenate((A_gt,B_gt),axis=0)).permute(1,0)#[n,3]->[3,n]

		return partial, gt, type_num, scene_num, scan_num
	
	def __len__(self):
		return self.data_pairs.shape[0]



if __name__ == '__main__':
	print()
	# dset_train = co_xyzDataSet(dataset_path='./data', 
	# 							type_num=1,
	# 							splits=[1,2,3,4])
	# dataloader_train = torch.utils.data.DataLoader(dset_train, 
	#                                                batch_size=32,
	#                                                shuffle=True,
	#                                                num_workers=8)
	# for AB_partial, gt_coarse in dataloader_train:
	# 	print(AB_partial.shape, gt_coarse.shape)
	# print(len(dset_train))