import argparse, os, sys
import dlib
import cv2

def load_images(base_directory, number):
	images = []
	directory = os.path.join(base_directory, "helen_%d" % number)
	fs = os.listdir(directory)
	for filename in fs:
		print("loading", filename)
		image_rgb = cv2.imread(os.path.join(directory, filename))
		image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
		images.append((image_gray, filename))
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
			landmarks = annotation[1:]
			annotations[image_name] = landmarks
	return annotations

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--helen-directory", "-helen", type=str, default=None)
	args = parser.parse_args()

	assert args.helen_directory is not None

	dlib_detector = dlib.get_frontal_face_detector()

	annotations = load_annotations(args.helen_directory)

	images = []
	images += load_images(args.helen_directory, 1)
	images += load_images(args.helen_directory, 2)
	images += load_images(args.helen_directory, 3)
	images += load_images(args.helen_directory, 4)
	images += load_images(args.helen_directory, 5)


	for (image, filename) in images:
		print("processing", filename)
		face_rects = dlib_detector(image, 0)
		if len(face_rects) > 0:
			for rect in face_rects:
				pass

if __name__ == "__main__":
	main()