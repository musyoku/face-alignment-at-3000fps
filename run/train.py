import argparse, os, sys
import cv2
import numpy as np
import lbf

class BoundingBox:
	def __init__(self):
		self.left = 0
		self.top = 0
		self.right = 0
		self.bottom = 0

	def move_y(self, move):
		self.top += move
		self.bottom += move

	def move_x(self, move):
		self.top += move
		self.bottom += move

	def cast(self):
		self.left = int(self.left)
		self.top = int(self.top)
		self.right = int(self.right)
		self.bottom = int(self.bottom)

	def width(self):
		return self.right - self.left

	def height(self):
		return self.bottom - self.top

def load_annotations(directory):
	annotations = {}
	fs = os.listdir(directory)
	for filename in fs:
		if filename.endswith(".pts"):
			with open(os.path.join(directory, filename), "r") as f:
				annotation = f.read().strip().split("\n")
				assert len(annotation) == 72
				landmarks = []
				for location_str in annotation[3:-1]:
					location = location_str.split(" ")
					landmarks.append((float(location[0]), float(location[1])))
				annotations[filename.replace(".pts", "")] = landmarks
	return annotations

# expand the bounding box
def expand_bounding_box(bbox, image_height, image_width, padding):
	padding = min(padding, bbox.left)
	padding = min(padding, bbox.top)
	padding = min(padding, image_width - bbox.right)
	padding = min(padding, image_height - bbox.bottom)

	bbox.left = bbox.left - padding
	bbox.top = bbox.top - padding
	bbox.right = bbox.right + padding
	bbox.bottom = bbox.bottom + padding

	bbox.cast()

def get_bounding_box(landmarks, image_height, image_width):
	bbox = BoundingBox()
	bbox.left = landmarks[0][0]
	bbox.top = landmarks[0][1]
	bbox.right = landmarks[0][0]
	bbox.bottom = landmarks[0][1]

	for (x, y) in landmarks:
		if x > bbox.right:
			bbox.right = x
		if x < bbox.left:
			bbox.left = x
		if y > bbox.bottom:
			bbox.bottom = y
		if y < bbox.top:
			bbox.top = y

	bbox.cast()

	# make the bounding box square
	bbox_width = bbox.width()
	bbox_height = bbox.height()

	if bbox_width > bbox_height:
		diff = (bbox_width - bbox_height) // 2
		mod = (bbox_width - bbox_height) % 2
		bbox.top = bbox.top - diff
		bbox.bottom = bbox.bottom + diff + mod
		if bbox.top < 0:
			bbox.move_y(-bbox.top)
			assert bbox.bottom <= image_height
		elif bbox.bottom > image_height:
			bbox.move_y(image_height - bbox.bottom)
			assert bbox.top >= 0

	elif bbox_width < bbox_height:
		diff = (bbox_height - bbox_width) // 2
		mod = (bbox_height - bbox_width) % 2
		bbox.left = bbox.left - diff
		bbox.right = bbox.right + diff + mod
		if bbox.left < 0:
			bbox.move_x(-bbox.left)
			assert bbox.right <= image_width
		elif bbox.right > image_width:
			bbox.move_x(image_width - bbox.right)
			assert bbox.left >= 0

	assert bbox.width() == bbox.height()
	return bbox

def preprocess_images(directory):
	print("processing", directory)
	annotations = load_annotations(directory)
	fs = os.listdir(directory)
	num_total_images = 0
	dataset_images = []
	dataset_landmarks = []
	mean_normalized_landmarks = []
	for _ in range(68):
		mean_normalized_landmarks.append([0, 0])

	for filename in fs:
		if filename.endswith(".png") or filename.endswith("jpg"):
			image_rgb = cv2.imread(os.path.join(directory, filename))
			image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

			filename = filename.replace(".png", "")
			filename = filename.replace(".jpg", "")
			assert filename in annotations
			landmarks = annotations[filename]

			image_height = image_rgb.shape[0]
			image_width = image_rgb.shape[1]

			try:
				bbox = get_bounding_box(landmarks, image_height, image_width)
				padding = bbox.width() * 0.3
				expand_bounding_box(bbox, image_height, image_width, padding)
			except Exception as e:
				continue

			image_gray = image_gray[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
			if bbox.width() > args.max_image_size:
				image_gray = cv2.resize(image_gray, (args.max_image_size, args.max_image_size))

			dataset_images.append(image_gray)

			# localize landmark location
			# x: [-1, 1]
			# y: [-1, 1]
			localized_landmarks = []
			for feature_index, (x, y) in enumerate(landmarks):
				x = (x - bbox.left) / bbox.width() * 2 - 1
				y = (y - bbox.top) / bbox.height() * 2 - 1
				localized_landmarks.append((x, y))

			dataset_landmarks.append(localized_landmarks)

	return dataset_images, dataset_landmarks

def read_images_and_shapes(targets):
	image_list = []
	shape_list = []

	mean_shape = []
	for _ in range(68):
		mean_shape.append([0, 0])

	for target in targets:
		images, shape = preprocess_images(os.path.join(args.dataset_directory, target))
		image_list += images
		shape_list += shape

	# calculate mean shape
	for shape in shape_list:
		for feature_index, (x, y) in enumerate(shape):
			mean_shape[feature_index][0] += x
			mean_shape[feature_index][1] += y
		
	for feature_index in range(len(mean_shape)):
		mean_shape[feature_index][0] /= len(shape_list)
		mean_shape[feature_index][1] /= len(shape_list)

	mean_shape = np.asarray(mean_shape, dtype=np.float64)

	return image_list, shape_list, mean_shape

def build_corpus(targets, mean_shape=None):
	corpus = lbf.corpus()
	image_list, shape_list, _mean_shape = read_images_and_shapes(targets)
	if mean_shape is None:
		mean_shape = _mean_shape

	for image, shape in zip(image_list, shape_list):
		shape = np.asarray(shape, dtype=np.float64)

		# normalize shape
		mat = cv2.estimateRigidTransform(shape, mean_shape, False)
		if mat is None:
			continue

		rotation = mat[:, :2]
		shift = mat[:, 2]
		normalized_shape = np.transpose(np.dot(rotation, shape.T) + shift[:, None], (1, 0))

		mat = cv2.estimateRigidTransform(normalized_shape, shape, False)
		if mat is None:
			continue

		rotation_inv = mat[:, :2]
		shift_inv = mat[:, 2]

		corpus.add(image, shape, normalized_shape, rotation, rotation_inv, shift, shift_inv)

	return corpus, mean_shape

def imwrite(image_gray, shape, filename):
	image_height = image_gray.shape[0]
	image_width = image_gray.shape[1]
	image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
	color = (0, 255, 255)
	for (x, y) in shape:
		x = int(image_width / 2 + x * image_width / 2)
		y = int(image_height / 2 + y * image_height / 2)
		
		cv2.line(image_bgr, (x - 4, y), (x + 4, y), color, 1)
		cv2.line(image_bgr, (x, y - 4), (x, y + 4), color, 1)

	cv2.imwrite(os.path.join(args.debug_directory, filename), image_bgr)

def main():
	assert args.dataset_directory is not None

	try:
		os.mkdir(args.output_directory)
	except:
		pass

	# build corpus
	training_targets = ["afw", "ibug", "helen/trainset", "lfpw/trainset"]
	training_targets = ["afw"]
	validation_targets = ["helen/testset", "lfpw/testset"]
	validation_targets = ["ibug"]
	training_corpus, mean_shape = build_corpus(training_targets)
	validation_corpus, _ = build_corpus(validation_targets, mean_shape=mean_shape)
	print("#images (train):", training_corpus.get_num_images())
	print("#images (val):", validation_corpus.get_num_images())

	# save mean shape
	mean_shape_image = np.zeros((500, 500), dtype=np.uint8)
	white = (255, 255, 255)
	for (x, y) in mean_shape:
		x = int(250 + x * 250)
		y = int(250 + y * 250)
		cv2.line(mean_shape_image, (x - 4, y), (x + 4, y), white, 1)
		cv2.line(mean_shape_image, (x, y - 4), (x, y + 4), white, 1)
	cv2.imwrite("mean.jpg", mean_shape_image)

	# initialize dataset
	dataset = lbf.dataset(training_corpus=training_corpus, 
						  validation_corpus=validation_corpus,
						  augmentation_size=args.augmentation_size)

	# initlaize model
	feature_radius = [0.29, 0.21, 0.16, 0.12, 0.08, 0.04]
	assert len(feature_radius) == args.num_stages
	model = lbf.model(num_stages=args.num_stages,
					  num_trees_per_forest=args.num_trees,
					  tree_depth=args.tree_depth,
					  num_landmarks=len(mean_shape),
					  mean_shape_ndarray=mean_shape, 
					  feature_radius=feature_radius)
	model.save(args.model_filename)

	# training
	trainer = lbf.trainer(dataset=dataset, 
						  model=model,
						  num_features_to_sample=args.num_training_features)

	# debug
	if args.debug_directory is not None:
		for data_index in range(50):
			augmented_data_index = data_index
			image = training_corpus.get_image(data_index)

			estimated_shape = trainer.get_current_estimated_shape(augmented_data_index, transform=True)
			imwrite(image.copy(), estimated_shape, os.path.join(args.debug_directory, "{}_initial_shape_stage_{}.jpg".format(data_index, 0)))

	for stage in range(args.num_stages):
		trainer.train_stage(stage)
		model.save(args.model_filename)
		trainer.evaluate_stage(stage)

		# debug
		if args.debug_directory is not None:
			for data_index in range(50):
				augmented_data_index = data_index
				image = training_corpus.get_image(data_index)

				estimated_shape = trainer.estimate_shape_only_using_local_binary_features(stage, augmented_data_index, transform=True)
				imwrite(image.copy(), estimated_shape, os.path.join(args.debug_directory, "{}_local_stage_{}.jpg".format(data_index, stage)))
				
				estimated_shape = trainer.get_current_estimated_shape(augmented_data_index, transform=True)
				imwrite(image.copy(), estimated_shape, os.path.join(args.debug_directory, "{}_estimated_stage_{}.jpg".format(data_index, stage)))
				
				target_shape = trainer.get_target_shape(augmented_data_index, transform=True)
				imwrite(image.copy(), target_shape, os.path.join(args.debug_directory, "{}_target_stage_{}.jpg".format(data_index, stage)))

				# validation
				estimated_shape = trainer.get_validation_estimated_shape(data_index, transform=True)
				imwrite(image.copy(), estimated_shape, os.path.join(args.debug_directory, "validation_{}_stage_{}.jpg".format(data_index, stage)))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset-directory", "-dataset", type=str, default=None)
	parser.add_argument("--debug-directory", "-debug", type=str, default=None)
	parser.add_argument("--model-filename", "-model", type=str, default="lbf.model")
	parser.add_argument("--max-image-size", "-size", type=int, default=500)
	parser.add_argument("--augmentation-size", "-augment", type=int, default=20)
	parser.add_argument("--num-stages", "-stages", type=int, default=6)
	parser.add_argument("--num-trees", "-trees", type=int, default=12)
	parser.add_argument("--num-training-features", "-features", type=int, default=500)
	parser.add_argument("--tree-depth", "-depth", type=int, default=7)
	args = parser.parse_args()
	main()