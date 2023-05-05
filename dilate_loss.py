import torch
from . import soft_dtw
from . import path_soft_dtw 

def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	try:
		batch_size1, N_output = outputs.shape
	except:
		batch_size1=1
		N_output= len(outputs)

	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size1, N_output, N_output)).to(device)
	for k in range(batch_size1):
		Dk = soft_dtw.pairwise_distances(targets[k, :].view(-1, 1), outputs[k,:].view(-1, 1)) #
		# print('DK的size是这样的{}'.format(Dk.shape))
		D[k:k+1, :, :] = Dk
	loss_shape = softdtw_batch(D, gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega = soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal = torch.sum(path*Omega) / (N_output*N_output)
	loss = alpha*loss_shape + (1-alpha)*loss_temporal
	return loss