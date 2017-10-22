import argparse, os, sys
import cv2
import numpy as np

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
			scale = 1.0
			if bbox.width() > args.max_image_size:
				scale = args.max_image_size / bbox.width()
				image_gray = cv2.resize(image_gray, (args.max_image_size, args.max_image_size))

			# normalize landmark location
			# x: [-1, 1]
			# y: [-1, 1]
			normalized_landmarks = []
			for feature_index, (x, y) in enumerate(landmarks):
				x = scale * (x - bbox.left) / bbox.width() * 2 - 1
				y = scale * (y - bbox.top) / bbox.height() * 2 - 1
				normalized_landmarks.append((x, y))

				mean_normalized_landmarks[feature_index][0] += x
				mean_normalized_landmarks[feature_index][1] += y

			num_total_images += 1

	for feature_index in range(len(mean_normalized_landmarks)):
		mean_normalized_landmarks[feature_index][0] /= num_total_images
		mean_normalized_landmarks[feature_index][1] /= num_total_images

	return num_total_images, mean_normalized_landmarks

def main():
	assert args.dataset_directory is not None

	try:
		os.mkdir(args.output_directory)
	except:
		pass

	num_total_images = 0
	mean_normalized_landmarks = []
	for _ in range(68):
		mean_normalized_landmarks.append([0, 0])
	targets = ["01_Indoor", "02_Outdoor"]

	for target in targets:
		_num_total_images, _mean_normalized_landmarks = preprocess_images(os.path.join(args.dataset_directory, target))
		num_total_images += _num_total_images

		for feature_index in range(len(_mean_normalized_landmarks)):
			mean_normalized_landmarks[feature_index][0] += _mean_normalized_landmarks[feature_index][0]
			mean_normalized_landmarks[feature_index][1] += _mean_normalized_landmarks[feature_index][1]

	for feature_index in range(len(mean_normalized_landmarks)):
		mean_normalized_landmarks[feature_index][0] /= len(targets)
		mean_normalized_landmarks[feature_index][1] /= len(targets)

	mean_shape_image = np.zeros((500, 500), dtype=np.uint8)
	white = (255, 255, 255)
	for (x, y) in mean_normalized_landmarks:
		x = int(250 + x * 250)
		y = int(250 + y * 250)
		cv2.line(mean_shape_image, (x - 4, y), (x + 4, y), white, 1)
		cv2.line(mean_shape_image, (x, y - 4), (x, y + 4), white, 1)

	cv2.imwrite("mean.jpg", mean_shape_image)
	print(mean_normalized_landmarks)

	print("#images", num_total_images)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset-directory", "-dataset", type=str, default=None)
	parser.add_argument("--max-image-size", "-size", type=int, default=500)
	args = parser.parse_args()
	main()