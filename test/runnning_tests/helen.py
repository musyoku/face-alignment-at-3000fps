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
		images.append((image_gray, filename))
	return images

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--helen-directory", "-helen", type=str, default=None)
	parser.add_argument("--output-directory", "-out", type=str, default=None)
	args = parser.parse_args()

	assert args.helen_directory is not None
	assert args.output_directory is not None

	try:
		os.mkdir(args.output_directory)
	except:
		pass

	cv_cascade_path = "../../cv_haarcascade.xml"
	cv_cascade = cv2.CascadeClassifier(cv_cascade_path)

	dlib_detector = dlib.get_frontal_face_detector()

	images = []
	images += load_images(args.helen_directory, 1)
	images += load_images(args.helen_directory, 2)
	images += load_images(args.helen_directory, 3)
	images += load_images(args.helen_directory, 4)
	images += load_images(args.helen_directory, 5)

	for (image, filename) in images:
		print("processing", filename)

		# OpenCV
		_image = image.copy()
		face_rects = cv_cascade.detectMultiScale(_image, scaleFactor=1.71, minNeighbors=2, minSize=(30, 30))
		color = (255, 255, 255)
		if len(face_rects) > 0:
			for rect in face_rects:
				cv2.rectangle(_image, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), color, thickness=2)
			cv2.imwrite(os.path.join(args.output_directory, "%s.cv.jpg" % filename), _image)

		# Dlib
		_image = image.copy()
		face_rects = dlib_detector(_image, 0)
		if len(face_rects) > 0:
			for rect in face_rects:
				cv2.rectangle(_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), color, thickness=2)
			cv2.imwrite(os.path.join(args.output_directory, "%s.dlib.jpg" % filename), _image)

if __name__ == "__main__":
	main()