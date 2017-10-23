#pragma once
#include <opencv/opencv.hpp>

namespace lbf {
	class FeatureLocation {
	public:
		cv::Point2d start;
		cv::Point2d end;
		FeatureLocation(cv::Point2d a, cv::Point2d b){
			start = a;
			end = b;
		}
		FeatureLocation(){
			start = cv::Point2d(0.0, 0.0);
			end = cv::Point2d(0.0, 0.0);
		};
	};
}