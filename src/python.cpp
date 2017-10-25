#include "python/corpus.h"
#include "python/dataset.h"
#include "python/model.h"
#include "python/trainer.h"

using namespace lbf::python;
using boost::python::arg;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(lbf){
	Py_Initialize();
	np::initialize();

	boost::python::class_<Corpus>("corpus")
	.def("get_training_image", &Corpus::get_training_image)
	.def("get_num_training_images", &Corpus::get_num_training_images)
	.def("get_num_test_images", &Corpus::get_num_test_images)
	.def("add_test_data", &Corpus::add_test_data)
	.def("add_training_data", &Corpus::add_training_data);

	boost::python::class_<Dataset>("dataset", boost::python::init<Corpus*, np::ndarray, int>((arg("corpus"), arg("mean_shape_ndarray"), arg("augmentation_size"))))
	.def("get_num_training_images", &Dataset::get_num_training_images);
	
	boost::python::class_<Model>("model", boost::python::init<int, int, int, int, boost::python::list>((arg("num_stages"), arg("num_trees_per_forest"), arg("tree_depth"), arg("num_landmarks"), arg("feature_radius"))));

	boost::python::class_<Trainer>("trainer", boost::python::init<Dataset*, Model*, int>((arg("dataset"), arg("model"), arg("num_features_to_sample"))))
	.def("get_predicted_shape", &Trainer::get_predicted_shape)
	.def("train", &Trainer::train)
	.def("train_stage", &Trainer::train_stage);
}