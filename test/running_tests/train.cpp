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
	for(int data_index = 0;data_index < training_corpus->get_num_images();data_index++){
		trainer->get_predicted_shape(data_index, true);
	}
	trainer->train();

	delete[] mean_shape;
	delete training_corpus;
	delete dataset;
	delete model;
	delete trainer;
}