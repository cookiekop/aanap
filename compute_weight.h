//
// Created by Michael Zhang on 2018/8/2.
//

#ifndef AANAP_COMPUTE_WEIGHT_H
#define AANAP_COMPUTE_WEIGHT_H

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

vector<float> compute_weight(const Point2f &p, const Point2f &K1, const Point2f &K2);

#endif //AANAP_COMPUTE_WEIGHT_H
