#include <opencv/opencv.hpp>
#include <vector>
#include <set>
#include "../../src/lbf/sampler.h"
#include "../../src/python/corpus.h"
#include "../../src/python/dataset.h"
#include "../../src/python/model.h"
#include "../../src/python/trainer.h"

using namespace lbf::randomforest;
using namespace lbf::python;
using std::cout;
using std::endl;

int main(){
	Corpus* corpus = new Corpus();
	cv::Mat_<uint8_t> image(500, 500);

}