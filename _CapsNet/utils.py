import os
import scipy
from scipy import misc
import numpy as np
import tensorflow as tf
import random

def load_welding(batch_size, is_training=True):
	
	data_subset='test_4'
	datatype='active'
	train_ratio = 0.8 # 80% data for training, the rest for testing 
	validate_ratio = 0.2 # 20% training data for validation

	#############################################
	# reading images  
	#############################################

	# get the list of filenames and corresponding list of labels for training and testing
	train_filenames = [] 
	train_labels = []
	test_filenames = []
	test_labels = []

	# read files into label and frame lists
	with open('../../data/labels/' + data_subset + '_' + datatype + '_label.csv') as f:
		frames_labels = [(line.strip().split(',')[0], line.strip().split(',')[1]) for line in f]

	# re-organize data by labels
	file_dir = '../../data/' + data_subset + '/' + datatype + '/jpg/'
	file_format = '.jpeg'
	dict_frame = {} # key: label value, value: a list of indice in the original file
	for fr_lb in frames_labels:
		fr, lb  = fr_lb
		if (lb not in dict_frame):
			dict_frame[lb] = []
		dict_frame[lb].append(file_dir + fr + file_format)

	random.seed() # using current time as the seed 
	# generate filenames and labels for training and validation dataset
	for lb in dict_frame:
		# pick random indices for training data for lb in dict_frame
		train_index = random.sample(range(0, len(dict_frame[lb])), int(train_ratio*len(dict_frame[lb])))
		for index in range(len(dict_frame[lb])):
			# training data
			if (index in train_index):
				train_filenames.append(dict_frame[lb][index])
				train_labels.append(int(lb) - 1)
			# validation data 
			else:
				test_filenames.append(dict_frame[lb][index])
				test_labels.append(int(lb) - 1)

	#############################################
	# process training and validation images 
	#############################################
	# read training images
	for index,img_name in enumerate(train_filenames):
		img = misc.imread(img_name, mode='L')
		img_resize = misc.imresize(img, (28,28))
		img_reshape = img_resize.reshape((1, 28, 28, 1))
		if (index == 0):
			trainX = img_reshape / 255.
		else:
			trainX = np.concatenate((trainX, img_reshape), axis=0)
			
	# read training labels, just convert the list to array
	trainY = np.array(train_labels)

	# shuffle trainX and trainY
	trainXY = list(zip(trainX, trainY))
	random.shuffle(trainXY)
	trainX, trainY = zip(*trainXY)
	trX = np.float32(np.array(trainX))
	trY = np.array(trainY)

	#############################################
	# process testing images 
	#############################################
	# read testing images
	for index,img_name in enumerate(test_filenames):
		img = misc.imread(img_name, mode='L')
		img_resize = misc.imresize(img, (28,28))
		img_reshape = img_resize.reshape((1, 28, 28, 1))
		if (index == 0):
			valX = img_reshape / 255.
		else:
			valX = np.concatenate((valX, img_reshape), axis=0)
	valX = np.float32(np.array(valX))
	
	# read testing labels, just convert the list to array
	valY = np.array(test_labels)

	if is_training:
		num_tr_batch = len(trX) // batch_size
		num_val_batch = len(valX) // batch_size
		return trX, trY, num_tr_batch, valX, valY, num_val_batch
	else:
		num_te_batch = len(valX) // batch_size
		return valX, valY, num_te_batch


def load_mnist(batch_size, is_training=True):
	path = os.path.join('data', 'mnist')
	if is_training:
		fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

		fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		trainY = loaded[8:].reshape((60000)).astype(np.int32)

		trX = trainX[:55000] / 255.
		trY = trainY[:55000]

		valX = trainX[55000:, ] / 255.
		valY = trainY[55000:]

		num_tr_batch = 55000 // batch_size
		num_val_batch = 5000 // batch_size

		return trX, trY, num_tr_batch, valX, valY, num_val_batch
	else:
		fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

		fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		teY = loaded[8:].reshape((10000)).astype(np.int32)

		num_te_batch = 10000 // batch_size
		return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
	path = os.path.join('data', 'fashion-mnist')
	if is_training:
		fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

		fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		trainY = loaded[8:].reshape((60000)).astype(np.int32)

		trX = trainX[:55000] / 255.
		trY = trainY[:55000]

		valX = trainX[55000:, ] / 255.
		valY = trainY[55000:]

		num_tr_batch = 55000 // batch_size
		num_val_batch = 5000 // batch_size

		return trX, trY, num_tr_batch, valX, valY, num_val_batch
	else:
		fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

		fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		teY = loaded[8:].reshape((10000)).astype(np.int32)

		num_te_batch = 10000 // batch_size
		return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
	if dataset == 'mnist':
		return load_mnist(batch_size, is_training)
	elif dataset == 'fashion-mnist':
		return load_fashion_mnist(batch_size, is_training)
	elif dataset == 'welding':
		return load_welding(batch_size, is_training)
	else:
		raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
	if dataset == 'mnist':
		trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
	elif dataset == 'fashion-mnist':
		trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
	elif dataset == 'welding':
		trX, trY, num_tr_batch, valX, valY, num_val_batch = load_welding(batch_size, is_training=True)
	data_queues = tf.train.slice_input_producer([trX, trY])
	X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
								  batch_size=batch_size,
								  capacity=batch_size * 64,
								  min_after_dequeue=batch_size * 32,
								  allow_smaller_final_batch=False)

	return(X, Y)


def save_images(imgs, size, path):
	'''
	Args:
		imgs: [batch_size, image_height, image_width]
		size: a list with tow int elements, [image_height, image_width]
		path: the path to save images
	'''
	imgs = (imgs + 1.) / 2  # inverse_transform
	return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
	h, w = images.shape[1], images.shape[2]
	imgs = np.zeros((h * size[0], w * size[1], 3))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		imgs[j * h:j * h + h, i * w:i * w + w, :] = image

	return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
	try:
		return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
	except:
		return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
	try:
		return tf.nn.softmax(logits, axis=axis)
	except:
		return tf.nn.softmax(logits, dim=axis)
