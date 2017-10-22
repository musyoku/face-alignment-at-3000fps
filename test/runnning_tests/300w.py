import argparse, os, sys
import cv2

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

def load_images(base_directory, category):
	directory = os.path.join(base_directory, category)
	images = []
	fs = os.listdir(directory)
	for filename in fs:
		print("loading", filename)
		if filename.endswith(".png"):
			image_rgb = cv2.imread(os.path.join(directory, filename))
			image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
			images.append((image_rgb, image_gray, filename.replace(".png", "")))
	return images

def load_annotations(base_directory, category):
	directory = os.path.join(base_directory, category)
	annotations = {}
	fs = os.listdir(directory)
	for filename in fs:
		print("loading", filename)
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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset-directory", "-dataset", type=str, default=None)
	parser.add_argument("--output-directory", "-out", type=str, default=None)
	args = parser.parse_args()

	assert args.dataset_directory is not None
	assert args.output_directory is not None

	try:
		os.mkdir(args.output_directory)
	except:
		pass

	annotations = {}
	annotations = dict(annotations, **load_annotations(args.dataset_directory, "01_Indoor"))
	annotations = dict(annotations, **load_annotations(args.dataset_directory, "02_Outdoor"))

	images = []
	images += load_images(args.dataset_directory, "01_Indoor")
	images += load_images(args.dataset_directory, "02_Outdoor")

	for (image_rgb, image_gray, filename) in images:
		print("processing", filename)
		assert filename in annotations
		landmarks = annotations[filename]
		white = (255, 255, 255)

		image_height = image_rgb.shape[0]
		image_width = image_rgb.shape[1]

		landmark_center = [0, 0]
		for (x, y) in landmarks:
			landmark_center[0] += x
			landmark_center[1] += y

		# import pdb;pdb.set_trace()

		try:
			bbox = get_bounding_box(landmarks, image_height, image_width)
			padding = bbox.width() * 0.3
			expand_bounding_box(bbox, image_height, image_width, padding)
		except Exception as e:
			import traceback
			traceback.print_exc()
			continue

		landmark_center[0] = int(landmark_center[0] / len(landmarks))
		landmark_center[1] = int(landmark_center[1] / len(landmarks))
		
		for (x, y) in landmarks:
			x = int(x)
			y = int(y)
			cv2.line(image_rgb, (x - 4, y), (x + 4, y), white, 1)
			cv2.line(image_rgb, (x, y - 4), (x, y + 4), white, 1)

		cv2.rectangle(image_rgb, (bbox.left, bbox.top), (bbox.right, bbox.bottom), white, thickness=2)
		cv2.imwrite(os.path.join(args.output_directory, "%s.jpg" % filename), image_rgb)

if __name__ == "__main__":
	main()