//
// Created by Michael Zhang on 2018/8/1.
//

#ifndef AANAP_HOMOGRAPHY_WARP_H
#define AANAP_HOMOGRAPHY_WARP_H

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Point2f homography_warp(const Mat& src, const Mat& H, const Point2f offset, const Size s, Mat& dst);

Mat homography_linearization(const Mat &H, const Point &center_point, const vector<Point2f> &anchor_points);

#endif //AANAP_HOMOGRAPHY_WARP_H
