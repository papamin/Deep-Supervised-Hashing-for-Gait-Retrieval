#########################################################################
#Script Name : Evluation.py
#Description : 1) Hamming Distance 
	       2) Compare Bit
	       3) Evaluation MAP
#Python Ver  : 3.6
#########################################################################

import numpy as np

def evaluate_macro(Rel, Ret):
		
	Rel_mat = np.mat(Rel)
	numTest = Rel_mat.shape[1]
	print ('numTest=',numTest)
	precisions = np.zeros((numTest))
	recalls    = np.zeros((numTest))

	retrieved_relevant_pairs = (Rel & Ret)

	for j in range(numTest):
		retrieved_relevant_num = len(retrieved_relevant_pairs[:,j][np.nonzero(retrieved_relevant_pairs[:,j])])
		retrieved_num = len(Ret[:, j][np.nonzero(Ret[:, j])])
		relevant_num  = len(Rel[:, j][np.nonzero(Rel[:, j])])
				
		if retrieved_num:
			precisions[j] = float(retrieved_relevant_num) / retrieved_num
		
		else:
			precisions[j] = 0.0

		if relevant_num:
			recalls[j]    = float(retrieved_relevant_num) / relevant_num
		
		else:
			recalls[j]    = 0.0

	p = np.mean(precisions)
	r = np.mean(recalls)
	return p,r

def hammingDist(B1, B2):
	
	n1 = B1.shape[0]
	n2, nwords = B2.shape

	Dh = np.zeros((n1, n2), dtype = np.uint16)
	for i in range(n1):
		for j in range(nwords):
			y = (B1[i, j] ^ B2[:, j]).T
			Dh[i, :] = Dh[i, :] + bit_in_char[y]
	return Dh



import numpy as np


def compactbit(b):
	'''
	b = bits array
	cb = compacted string of bits(using words of 'word' bits)
	'''
	b_mat = np.mat(b)
	[nSamples, nbits] = b_mat.shape
	nwords = int(np.ceil((float(nbits) / 8)))
	cb = np.zeros((nSamples,nwords),dtype = np.uint8)

	for i in range(nSamples):
		for j in range(nwords):
			temp = b[i , j * 8 : (j + 1) * 8]
			value = convert(temp)
			cb[i,j] = value

	return cb

def convert(arr):
	arr_mat = np.mat(arr)
	[_, col] = arr_mat.shape
	value = 0
	for i in range(col):
		value = value + (2 ** i) * arr[i]
	
	return value






