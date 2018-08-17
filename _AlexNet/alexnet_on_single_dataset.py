import argparse
import os
import random

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='test_4', type=str)
parser.add_argument('--data_type', default='rei', type=str)
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=40, type=int)
parser.add_argument('--num_epochs2', default=400, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
	"""
	Check the accuracy of the model on either train or val (depending on dataset_init_op).
	"""
	# Initialize the correct dataset
	sess.run(dataset_init_op)
	num_correct, num_samples = 0, 0
	while True:
		try:
			correct_pred = sess.run(correct_prediction, {is_training: False})
			num_correct += correct_pred.sum()
			num_samples += correct_pred.shape[0]
		except tf.errors.OutOfRangeError:
			break

	# Return the fraction of datapoints that were correctly classified
	acc = float(num_correct) / num_samples
	return acc


def main(args):
	
	train_ratio = 0.8 # 80% data for training, the rest for testing 
	
	# get the list of filenames and corresponding list of labels for training et validation
	train_filenames = [] 
	train_labels = []
	val_filenames = []
	val_labels = []
	
	# read files into label and frame lists
	with open('../data/labels/' + args.dataset + '_' + args.data_type + '_label.csv') as f:
		frames_labels = [(line.strip().split(',')[0], line.strip().split(',')[1]) for line in f]

	# re-organize data by labels
	file_dir = '../data/' + args.dataset + '/' + args.data_type + '/jpg/'
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
				val_filenames.append(dict_frame[lb][index])
				val_labels.append(int(lb) - 1)

	assert set(train_labels) == set(val_labels),\
		   "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
																   set(val_labels))

	num_classes = len(set(train_labels))

	# --------------------------------------------------------------------------
	# In TensorFlow, you first want to define the computation graph with all the
	# necessary operations: loss, training op, accuracy...
	# Any tensor created in the `graph.as_default()` scope will be part of `graph`
	graph = tf.Graph()
	with graph.as_default():

		# Preprocessing (for both training and validation):
		# (1) Decode the image from jpg format
		# (2) Resize the image so its smaller side is 224 pixels long
		def _parse_function(filename, label):
			image_string = tf.read_file(filename)
			image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
			image = tf.cast(image_decoded, tf.float32)
			
			prefer_side = 224
			resized_image = tf.image.resize_image_with_crop_or_pad(image, prefer_side, prefer_side) # (2)
			
			return resized_image, label
			
		# Training dataset
		train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
		train_dataset = train_dataset.map(_parse_function)
		# train_dataset = train_dataset.map(training_preprocess)
		train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
		batched_train_dataset = train_dataset.batch(args.batch_size)

		# Validation dataset
		val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
		val_dataset = val_dataset.map(_parse_function)
		# val_dataset = val_dataset.map(val_preprocess)
		batched_val_dataset = val_dataset.batch(args.batch_size)


		# Now we define an iterator that can operator on either dataset.
		# The iterator can be reinitialized by calling:
		#     - sess.run(train_init_op) for 1 epoch on the training set
		#     - sess.run(val_init_op)   for 1 epoch on the valiation set
		# Once this is done, we don't need to feed any value for images and labels
		# as they are automatically pulled out from the iterator queues.

		# A reinitializable iterator is defined by its structure. We could use the
		# `output_types` and `output_shapes` properties of either `train_dataset`
		# or `validation_dataset` here, because they are compatible.
		iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
														   batched_train_dataset.output_shapes)
		images, labels = iterator.get_next()

		train_init_op = iterator.make_initializer(batched_train_dataset)
		val_init_op = iterator.make_initializer(batched_val_dataset)

		# Indicates whether we are in training or in test mode
		is_training = tf.placeholder(tf.bool)

		alexnet = tf.contrib.slim.nets.alexnet
		# alexnet.alexnet_v2.default_image_size = 28
		with slim.arg_scope(alexnet.alexnet_v2_arg_scope(weight_decay=args.weight_decay)):
			logits, _ = alexnet.alexnet_v2(images, num_classes=num_classes, is_training=is_training,
								   dropout_keep_prob=args.dropout_keep_prob)	
				
		# ---------------------------------------------------------------------
		# Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
		# We can then call the total loss easily
		tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		loss = tf.losses.get_total_loss()

		# Then we want to finetune the entire model for a few epochs.
		# We run minimize the loss only with respect to all the variables.
		full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
		full_train_op = full_optimizer.minimize(loss)

		# Evaluation metrics
		prediction = tf.to_int32(tf.argmax(logits, 1))
		correct_prediction = tf.equal(prediction, labels)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# finalizes this graph, making it read-only.
		# tf.get_default_graph().finalize()

	# --------------------------------------------------------------------------
	# Now that we have built the graph and finalized it, we define the session.
	# The session is the interface to *run* the computational graph.
	# We can call our training operations with `sess.run(train_op)` for instance

	# configure GPU
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5

	with tf.Session(graph=graph, config=config) as sess:
	# with tf.Session(graph=graph) as sess:
		sess.run(tf.global_variables_initializer())
		# Train the entire model for a few more epochs, continuing with the *same* weights.
		for epoch in range(args.num_epochs2):
			print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
			sess.run(train_init_op)
			while True:
				try:
					_ = sess.run(full_train_op, {is_training: True})
				except tf.errors.OutOfRangeError:
					break

			# Check accuracy on the train and val sets every epoch
			train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
			val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
			print('Train accuracy: %f' % train_acc)
			print('Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
