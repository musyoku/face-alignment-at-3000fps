#pragma once
#include <opencv/opencv.hpp>

namespace lbf {
	class FeatureLocation {
	public:
		cv::Point2d a;
		cv::Point2d b;
		FeatureLocation(cv::Point2d a, cv::Point2d b){
			a = a;
			b = b;
		}
		FeatureLocation(){
			a = cv::Point2d(0.0, 0.0);
			b = cv::Point2d(0.0, 0.0);
		};
	};
}