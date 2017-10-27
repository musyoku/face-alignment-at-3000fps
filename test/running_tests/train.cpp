#include <boost/python/numpy.hpp>
#include <opencv/opencv.hpp>
#include <vector>
#include <set>
#include "../../src/lbf/sampler.h"
#include "../../src/python/corpus.h"
#include "../../src/python/dataset.h"
#include "../../src/python/model.h"
#include "../../src/python/trainer.h"

using namespace lbf;
using namespace lbf::randomforest;
using namespace lbf::python;
using std::cout;
using std::endl;
namespace p = boost::python;
namespace np = boost::python::numpy;

void load_training_data(std::vector<cv::Mat1b> &images,
						std::vector<cv::Mat1d> &shapes)
{
	std::string fn_haar = "./../haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	bool yes = haar_cascade.load(fn_haar);
	std::cout << "face detector loaded : " << yes << std::endl;
	std::cout << "loading images..." << std::endl;

	int count = 0;
	for (int i = 0; i < image_path_prefix.size(); i++) {
		int c = 0;
		std::ifstream fin;
		fin.open((image_lists[i]).c_str(), std::ifstream::in);
		std::string path_prefix = image_path_prefix[i];
		std::string image_file_name, image_pts_name;
		std::cout << "loading images in folder: " << path_prefix << std::endl;
		while (fin >> image_file_name >> image_pts_name){
			std::string image_path, pts_path;
			if (path_prefix[path_prefix.size()-1] == '/') {
				image_path = path_prefix + image_file_name;
				pts_path = path_prefix + image_pts_name;
			}
			else{
				image_path = path_prefix + "/" + image_file_name;
				pts_path = path_prefix + "/" + image_pts_name;
			}
			cv::Mat_<uchar> image = cv::imread(image_path.c_str(), 0);
			// std::cout << "image size: " << image.size() << std::endl;
			cv::Mat_<double> ground_truth_shape = LoadGroundTruthShape(pts_path.c_str());

			if (image.cols > 2000){
				cv::resize(image, image, cv::Size(image.cols / 4, image.rows / 4), 0, 0, cv::INTER_LINEAR);
				ground_truth_shape /= 4.0;
			}
			else if (image.cols > 1400 && image.cols <= 2000){
				cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
				ground_truth_shape /= 3.0;
			}

			std::vector<cv::Rect> faces;
			haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));

			 for (int i = 0; i < faces.size(); i++){
				cv::Rect faceRec = faces[i];
				if (ShapeInRect(ground_truth_shape, faceRec)){
					images.push_back(image);
					ground_truth_shapes.push_back(ground_truth_shape);
					BoundingBox bbox;

					bbox.start_x = faceRec.x;
					bbox.start_y = faceRec.y;
					bbox.width = faceRec.width;
					bbox.height = faceRec.height;
					bbox.center_x = bbox.start_x + bbox.width / 2.0;
					bbox.center_y = bbox.start_y + bbox.height / 2.0;
					bboxes.push_back(bbox);
					count++;
					c++;
					if (count%100 == 0){
						std::cout << count << " images loaded\n";
					}
					break;
				}
			 }
		}
		std::cout << "get " << c << " faces in " << path_prefix << std::endl;
		fin.close();
	}

	std::cout << "get " << bboxes.size() << " faces in total" << std::endl;

}

int main(){
	Py_Initialize();
	np::initialize();
	Corpus* training_corpus = new Corpus();
	Corpus* validation_corpus = new Corpus();
	cv::Mat1b image(100, 100);
	cv::Mat1d shape(68, 2);
	int num_data = 10;
	for(int data_index = 0;data_index < num_data;data_index++){
		for(int h = 0;h < 100;h++){
			for(int w = 0;w < 100;w++){
				image(h, w) = sampler::uniform_int(0, 255);
			}
		}
		for(int landmark_index = 0;landmark_index < 68;landmark_index++){
			shape(landmark_index, 0) = sampler::uniform(-1, 1);
			shape(landmark_index, 1) = sampler::uniform(-1, 1);
		}
		training_corpus->_images.push_back(image);
		training_corpus->_shapes.push_back(shape);
		training_corpus->_normalized_shapes.push_back(shape);
		cv::Mat1d rotation(2, 2);
		rotation(0, 0) = 1;
		rotation(0, 1) = 0;
		rotation(1, 0) = 0;
		rotation(1, 1) = 1;
		training_corpus->_rotation.push_back(rotation);
		training_corpus->_rotation_inv.push_back(rotation);
		cv::Point2d shift;
		shift.x = 1;
		shift.y = 1;
		training_corpus->_shift.push_back(shift);
		training_corpus->_shift_inv.push_back(shift);
	}
	double* mean_shape = new double[68 * 2];
	for(int landmark_index = 0;landmark_index < 68;landmark_index++){
		mean_shape[landmark_index * 2 + 0] = sampler::uniform(-1, 1);
		mean_shape[landmark_index * 2 + 1] = sampler::uniform(-1, 1);
	}
	auto stride = p::make_tuple(sizeof(double) * 2, sizeof(double));
	np::ndarray mean_shape_ndarray = np::from_data(mean_shape, np::dtype::get_builtin<double>(), p::make_tuple(68, 2), stride, p::object());

	int augmentation_size = 5;
	int num_stages = 6;
	int num_trees_per_forest = 5;
	int tree_depth = 7;
	int num_landmarks = 68;
	int num_features_to_sample = 500;
	std::vector<double> feature_radius{0.29, 0.21, 0.16, 0.12, 0.08, 0.04};

	Dataset* dataset = new Dataset(training_corpus, validation_corpus, mean_shape_ndarray, augmentation_size);
	Model* model = new Model(num_stages, num_trees_per_forest, tree_depth, num_landmarks, feature_radius);
	Trainer* trainer = new Trainer(dataset, model, num_features_to_sample);
	trainer->train();

	delete[] mean_shape;
	delete training_corpus;
	delete dataset;
	delete model;
	delete trainer;
}