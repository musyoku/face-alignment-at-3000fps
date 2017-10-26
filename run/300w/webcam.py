from imutils.video import VideoStream
import argparse, imutils, time, dlib, cv2
import lbf

def main():
	detector = dlib.get_frontal_face_detector()
	model = lbf.model(args.model_filename)

	vs = VideoStream().start()
	time.sleep(2.0)

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		rects = detector(gray, 0)

		for rect in rects:
			face = frame[rect.top():rect.bottom(), rect.left():rect.right()]
			shape = model.estimate_shape(face)
			# cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 255), 1)
			cv2.imshow("face", face)
		  
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	cv2.destroyAllWindows()
	vs.stop()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-filename", "-model", type=str, default="lbf.model")
	args = parser.parse_args()
	main()