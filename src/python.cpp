#include "python/corpus.h"
#include "python/dataset.h"

using namespace lbf::python;
using boost::python::arg;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(lbf){
	np::initialize();

	boost::python::class_<Corpus>("corpus")
	.def("get_num_training_images", &Corpus::get_num_training_images)
	.def("get_num_test_images", &Corpus::get_num_test_images)
	.def("add_test_data", &Corpus::add_test_data)
	.def("add_training_data", &Corpus::add_training_data);

	boost::python::class_<Dataset>("dataset", boost::python::init<Corpus*, np::ndarray, int>((arg("corpus"), arg("mean_shape_ndarray"), arg("augmentation_size"))));
}