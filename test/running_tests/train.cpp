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
	Corpus* corpus = new Corpus();
	cv::Mat_<uint8_t> image(100, 100);
	cv::Mat_<uint8_t> shape(68, 2);
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
		corpus->_images_train.push_back(image);
		corpus->_shapes_train.push_back(shape);
		corpus->_normalized_shapes_train.push_back(shape);
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

	Dataset* dataset = new Dataset(corpus, mean_shape_ndarray, augmentation_size);
	Model* model = new Model(num_stages, num_trees_per_forest, tree_depth, num_landmarks, feature_radius);
	Trainer* trainer = new Trainer(dataset, model, num_features_to_sample);
	trainer->train();

	delete[] mean_shape;
	delete corpus;
	delete dataset;
	delete model;
	delete trainer;
}