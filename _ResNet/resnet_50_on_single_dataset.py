import math, json, os, sys, random, shutil
import argparse
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import ResNet50

# specify the dataset and data_type
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='test_4', type=str)
parser.add_argument('--data_type', default='rei', type=str)

def main(args):

	dataset = args.dataset
	data_type = args.data_type

	# data dir, image size and batch size
	DATA_DIR = 'data'
	TRAIN_DIR = os.path.join(DATA_DIR, 'train')
	VALID_DIR = os.path.join(DATA_DIR, 'valid')
	SIZE = (224, 224)
	BATCH_SIZE = 2

	# remove files
	try:
		shutil.rmtree(DATA_DIR)
	except:
		pass

	train_ratio = 0.8 # 80% data for training, the rest for testing 

	# get the list of filenames and corresponding list of labels for training et validation
	train_filenames = [] 
	train_labels = []
	val_filenames = []
	val_labels = []

	# read files into label and frame lists
	with open('../data/labels/' + dataset + '_' + data_type + '_label.csv') as f:
		frames_labels = [(line.strip().split(',')[0], line.strip().split(',')[1]) for line in f]

	# re-organize data by labels
	file_dir = '../data/' + dataset + '/' + data_type + '/jpg/'
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

	assert set(train_labels) == set(val_labels), "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels), set(val_labels))

	# create new dir data/train/label_x and data/valid/label_x
	for label in set(train_labels):
		os.makedirs(os.path.join(TRAIN_DIR, str(label)), exist_ok=True)
		os.makedirs(os.path.join(VALID_DIR, str(label)), exist_ok=True)

	# copy files
	for tr_file, label in zip(train_filenames, train_labels):
		shutil.copy2(tr_file, os.path.join(TRAIN_DIR, str(label)))
	for val_file, label in zip(val_filenames, val_labels):
		shutil.copy2(val_file, os.path.join(VALID_DIR, str(label)))
		
	# train models	
	num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
	num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

	num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
	num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

	gen = image.ImageDataGenerator()
	val_gen = image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

	batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
	val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

	model = ResNet50()

	classes = list(iter(batches.class_indices))
	model.layers.pop()
	for layer in model.layers:
		layer.trainable=False
	last = model.layers[-1].output
	x = Dense(len(classes), activation="softmax")(last)
	finetuned_model = Model(model.input, x)
	finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
	for c in batches.class_indices:
		classes[batches.class_indices[c]] = c
	finetuned_model.classes = classes

	early_stopping = EarlyStopping(patience=450)
	checkpointer = ModelCheckpoint('./resnet_model/resnet_50_best.h5', verbose=1, save_best_only=True)

	finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=450, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
	finetuned_model.save('./resnet_model/resnet_50_final.h5')

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)	