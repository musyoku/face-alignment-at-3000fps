import argparse, os, sys
import cv2
import dlib

def load_images(base_directory, number):
	directory = os.path.join(base_directory, "helen_%d" % number)
	images = []
	fs = os.listdir(directory)
	for filename in fs:
		print("loading", filename)
		image_rgb = cv2.imread(os.path.join(directory, filename))
		image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
		images.append((image_rgb, image_gray, filename.replace(".jpg", "")))
	return images

def load_annotations(base_directory):
	annotations = {}
	directory = os.path.join(base_directory, "annotation")
	fs = os.listdir(directory)
	for filename in fs:
		print("loading", filename)
		with open(os.path.join(directory, filename), "r") as f:
			annotation = f.read().strip().split("\n")
			assert len(annotation) == 195
			image_name = annotation[0]
			landmarks = []
			for location_str in annotation[1:]:
				location = location_str.split(" , ")
				landmarks.append((float(location[0]), float(location[1])))
			annotations[image_name] = landmarks
	return annotations

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

	dlib_detector = dlib.get_frontal_face_detector()
	annotations = load_annotations(args.dataset_directory)
	images = []
	images += load_images(args.dataset_directory, 1)
	images += load_images(args.dataset_directory, 2)
	images += load_images(args.dataset_directory, 3)
	images += load_images(args.dataset_directory, 4)
	images += load_images(args.dataset_directory, 5)

	for (image_rgb, image_gray, filename) in images:
		print("processing", filename)
		assert filename in annotations
		landmarks = annotations[filename]
		white = (255, 255, 255)

		image_width = image_rgb.shape[1]
		landmark_center = [0, 0]
		for (x, y) in landmarks:
			landmark_center[0] += x
			landmark_center[1] += y
		landmark_center[0] = int(landmark_center[0] / len(landmarks))
		landmark_center[1] = int(landmark_center[1] / len(landmarks))
		
		for (x, y) in landmarks:
			x = int(x)
			y = int(y)
			cv2.line(image_rgb, (x - 4, y), (x + 4, y), white, 1)
			cv2.line(image_rgb, (x, y - 4), (x, y + 4), white, 1)

		face_rects = dlib_detector(image_gray, 0)
		if len(face_rects) > 0:
			for rect in face_rects:
				if rect.contains(landmark_center[0], landmark_center[1]):
					cv2.rectangle(image_rgb, (rect.left(), rect.top()), (rect.right(), rect.bottom()), white, thickness=2)

		cv2.imwrite(os.path.join(args.output_directory, "%s.jpg" % filename), image_rgb)

if __name__ == "__main__":
	main()