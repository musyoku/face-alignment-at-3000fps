#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <cassert>
#include <set>
#include "../sampler.h"
#include "forest.h"

namespace lbf {
	namespace randomforest {
		Forest::Forest(int stage, int landmark_index, int num_trees, double radius, int tree_depth){
			_stage = stage;
			_num_trees = num_trees;
			_radius = radius;
			_landmark_index = landmark_index;
			_num_total_leaves = 0;

			_trees.reserve(num_trees);
			for(int n = 0;n < num_trees;n++){
				Tree* tree = new Tree(tree_depth, _landmark_index, this);
				_trees.push_back(tree);
			}
		}
		Forest::~Forest(){
			for(Tree* tree: _trees){
				delete tree;
			}
		}
		void Forest::train(std::vector<FeatureLocation> &feature_locations, 
						   cv::Mat_<int> &pixel_differences, 
						   std::vector<cv::Mat1d> &regression_targets)
		{
			assert(feature_locations.size() == pixel_differences.rows);
			assert(pixel_differences.cols == regression_targets.size());
			int num_data = pixel_differences.cols;
			assert(num_data > 0);
			for(int tree_index = 0;tree_index < get_num_trees();tree_index++){
				// bootstrap
				std::set<int> sampled_indices;
				for(int n = 0;n < num_data;n++){
					int index = sampler::uniform_int(0, num_data - 1);
					sampled_indices.insert(index);
				}
				assert(sampled_indices.size() > 0);
				// build tree
				Tree* tree = _trees[tree_index];
				tree->train(sampled_indices, feature_locations, pixel_differences, regression_targets);
				assert(tree->get_num_leaves() > 0);
				_num_total_leaves += tree->get_num_leaves();
			}
		}
		void Forest::predict(cv::Mat1d &shape, cv::Mat1b &image, std::vector<Node*> &leaves){
			leaves.clear();
			leaves.reserve(_num_trees);
			for(int tree_index = 0;tree_index < get_num_trees();tree_index++){
				Tree* tree = _trees[tree_index];
				Node* leaf = tree->predict(shape, image);
				assert(leaf->is_leaf() == true);
				assert(0 <= leaf->identifier() && leaf->identifier() < tree->get_num_leaves());
				leaves.push_back(leaf);
			}
		}
		Tree* Forest::get_tree_at(int tree_index){
			assert(tree_index < _num_trees);
			return _trees[tree_index];	
		}
		int Forest::get_num_trees(){
			assert(_num_trees == _trees.size());
			return _trees.size();
		}
		int Forest::get_num_total_leaves(){
			return _num_total_leaves;
		}
		template <class Archive>
		void Forest::serialize(Archive &ar, unsigned int version){
			ar & _stage;
			ar & _landmark_index;
			ar & _num_trees;
			ar & _num_total_leaves;
			ar & _radius;
			ar & _trees;
		}
		template void Forest::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void Forest::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
	}
}