import argparse
import os
import random

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='thickness_2_speed_1', type=str)
parser.add_argument('--data_type', default='Active', type=str)
parser.add_argument('--model_path', default='./vgg_pre_trained/vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=40, type=int)
parser.add_argument('--num_epochs2', default=400, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]

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

def my_accuracy(sess, prediction, is_training, dataset_init_op):
	"""
	Check the accuracy of the model on either train or val (depending on dataset_init_op).
	"""
	# Initialize the correct dataset
	sess.run(dataset_init_op)
	num_correct, num_samples = 0, 0
	while True:
		try:
			prediction_print = sess.run(prediction, {is_training: False})
			print (prediction_print)
		except tf.errors.OutOfRangeError:
			break

def main(args):

	# get the list of filenames and corresponding list of labels for training et validation
	train_filenames = [] 
	train_labels = []
	val_filenames = []
	val_labels = []
	
	# load training file names and labels 
	with open('../../data/classification/train_test_labels/' + args.dataset + '_' + args.data_type + '_train' + '.txt') as f:
		for line in f:
			train_filenames.append(line.strip().split(',')[0])
			train_labels.append(int(line.strip().split(',')[1]))
		
	# load testing file names and labels 
	with open('../../data/classification/train_test_labels/' + args.dataset + '_' + args.data_type + '_test' + '.txt') as f:
		for line in f:
			val_filenames.append(line.strip().split(',')[0])
			val_labels.append(int(line.strip().split(',')[1]))

	num_classes = len(set(train_labels))

	# --------------------------------------------------------------------------
	# In TensorFlow, you first want to define the computation graph with all the
	# necessary operations: loss, training op, accuracy...
	# Any tensor created in the `graph.as_default()` scope will be part of `graph`
	graph = tf.Graph()
	with graph.as_default():
		# Standard preprocessing for VGG on ImageNet taken from here:
		# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
		# Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

		# Preprocessing (for both training and validation):
		# (1) Decode the image from jpg format
		# (2) Resize the image so its smaller side is 256 pixels long
		def _parse_function(filename, label):
			image_string = tf.read_file(filename)
			image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
			image = tf.cast(image_decoded, tf.float32)

			smallest_side = 256.0
			height, width = tf.shape(image)[0], tf.shape(image)[1]
			height = tf.to_float(height)
			width = tf.to_float(width)

			scale = tf.cond(tf.greater(height, width),
							lambda: smallest_side / width,
							lambda: smallest_side / height)
			new_height = tf.to_int32(height * scale)
			new_width = tf.to_int32(width * scale)

			resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
			return resized_image, label

		# Preprocessing (for training)
		# (3) Take a random 224x224 crop to the scaled image
		# (4) Horizontally flip the image with probability 1/2
		# (5) Substract the per color mean `VGG_MEAN`
		# Note: we don't normalize the data here, as VGG was trained without normalization
		def training_preprocess(image, label):
			crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
			means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
			centered_image = crop_image - means                                     # (5)

			return centered_image, label

		# Preprocessing (for validation)
		# (3) Take a central 224x224 crop to the scaled image
		# (4) Substract the per color mean `VGG_MEAN`
		# Note: we don't normalize the data here, as VGG was trained without normalization
		def val_preprocess(image, label):
			crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

			means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
			centered_image = crop_image - means                                     # (4)

			return centered_image, label

		# ----------------------------------------------------------------------
		# DATASET CREATION using tf.data.Dataset
		# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

		# The tf.data.Dataset framework uses queues in the background to feed in
		# data to the model.
		# We initialize the dataset with a list of filenames and labels, and then apply
		# the preprocessing functions described above.
		# Behind the scenes, queues will load the filenames, preprocess them with multiple
		# threads and apply the preprocessing in parallel, and then batch the data

		# Training dataset
		train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
		train_dataset = train_dataset.map(_parse_function)
		train_dataset = train_dataset.map(training_preprocess)
		train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
		batched_train_dataset = train_dataset.batch(args.batch_size)

		# Validation dataset
		val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
		val_dataset = val_dataset.map(_parse_function)
		val_dataset = val_dataset.map(val_preprocess)
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

		# ---------------------------------------------------------------------
		# Now that we have set up the data, it's time to set up the model.
		# For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
		# last fully connected layer (fc8) and replace it with our own, with an
		# output size num_classes=8
		# We will first train the last layer for a few epochs.
		# Then we will train the entire model on our dataset for a few epochs.

		# Get the pretrained model, specifying the num_classes argument to create a new
		# fully connected replacing the last one, called "vgg_16/fc8"
		# Each model has a different architecture, so "vgg_16/fc8" will change in another model.
		# Here, logits gives us directly the predicted scores we wanted from the images.
		# We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
		vgg = tf.contrib.slim.nets.vgg
		with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
			logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
								   dropout_keep_prob=args.dropout_keep_prob)

		# Specify where the model checkpoint is (pretrained weights).
		model_path = args.model_path
		assert(os.path.isfile(model_path))

		# Restore only the layers up to fc7 (included)
		# Calling function `init_fn(sess)` will load all the pretrained weights.
		variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

		# Initialization operation from scratch for the new "fc8" layers
		# `get_variables` will only return the variables whose name starts with the given pattern
		fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
		fc8_init = tf.variables_initializer(fc8_variables)

		# ---------------------------------------------------------------------
		# Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
		# We can then call the total loss easily
		tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		loss = tf.losses.get_total_loss()

		# First we want to train only the reinitialized last layer fc8 for a few epochs.
		# We run minimize the loss only with respect to the fc8 variables (weight and bias).
		fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
		fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

		# Then we want to finetune the entire model for a few epochs.
		# We run minimize the loss only with respect to all the variables.
		full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
		full_train_op = full_optimizer.minimize(loss)

		# Evaluation metrics
		prediction = tf.to_int32(tf.argmax(logits, 1))
		# print (prediction)
		# print (labels)
		correct_prediction = tf.equal(prediction, labels)
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		tf.get_default_graph().finalize()

		
	# --------------------------------------------------------------------------
	# Now that we have built the graph and finalized it, we define the session.
	# The session is the interface to *run* the computational graph.
	# We can call our training operations with `sess.run(train_op)` for instance

	# configure GPU
	# config = tf.ConfigProto()
	# config.gpu_options.per_process_gpu_memory_fraction = 0.5

	# with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		init_fn(sess)  # load the pretrained weights
		sess.run(fc8_init)  # initialize the new fc8 layer

		accuracy_file = 'result_' + args.dataset + '_' + args.data_type + '.txt'
		# remove accuracy file if it exists
		try:
			os.remove(accuracy_file)
		except OSError:
			pass

		# write validation file names into files 
		with open(accuracy_file, 'a') as f:
			f.write('validation_file_name:' + ','.join([f_name.split('/')[-1] for f_name in val_filenames]) + '\n' + '\n')
			
		# write real labels back to files
		with open(accuracy_file, 'a') as f:
			# real labels
			real_labels = []
			sess.run(val_init_op)
			while True:
				try:
					labels_batch = sess.run(labels, {is_training: False})
					for lb in labels_batch:
						real_labels.append(str(lb))
				except tf.errors.OutOfRangeError:
					break
			f.write('real labels:' + ','.join(real_labels) + '\n' + '\n')
					
		# Update only the last layer for a few epochs.
		for epoch in range(args.num_epochs1):
			# Run an epoch over the training data.
			print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
			# Here we initialize the iterator with the training set.
			# This means that we can go through an entire epoch until the iterator becomes empty.
			sess.run(train_init_op)
			while True:
				try:
					_ = sess.run(fc8_train_op, {is_training: True})
				except tf.errors.OutOfRangeError:
					break

			# Check accuracy on the train and val sets every epoch.
			train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
			val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
			
			# predicted labels
			predicted_labels = []
			sess.run(val_init_op)
			while True:
				try:
					predicted_labels_batch = sess.run(prediction, {is_training: False})
					for pre_lb in predicted_labels_batch:
						predicted_labels.append(str(pre_lb))
				except tf.errors.OutOfRangeError:
					break
					
			# write accuracy results back to files
			with open(accuracy_file, 'a') as f:
				f.write('Starting epoch ' + str(epoch + 1) + ' / ' + str(args.num_epochs2) + '\n')
				f.write('Train accuracy: ' + str(train_acc) + '\n')
				f.write('Validation accuracy: ' + str(val_acc) + '\n')
				f.write('predict:' + ','.join(predicted_labels) + '\n' + '\n')
			

					
			# print('Train accuracy: %f' % train_acc)
			# print('Val accuracy: %f\n' % val_acc)
			
			
			# sess.run(val_init_op, {is_training: False})
			# print(sess.run(prediction), {is_training: False})
			
			# acc = sess.run(accuracy)
			# acc = sess.run(prediction)
			# print('reduced_mean accuracy: %f\n' % acc)

		# Train the entire model for a few more epochs, continuing with the *same* weights.
		for epoch in range(args.num_epochs2):
			print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
			sess.run(train_init_op, {is_training: False})
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
			# sess.run(val_init_op, {is_training: False})
			# print(sess.run(prediction), {is_training: False})
			# acc = sess.run(prediction)
			# sess.run(prediction)
			# print('reduced_mean accuracy: %f\n' % acc)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
