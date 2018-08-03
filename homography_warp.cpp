//
// Created by Michael Zhang on 2018/8/1.
//

#include "homography_warp.h"

Point2f homography_warp(const Mat& src, const Mat& H, const Point2f offset, const Size s, Mat& dst)
{
    vector<Point2f> src_p;
    for (int i=0; i<src.rows; ++i) {
        for(int j=0; j<src.cols; ++j) src_p.emplace_back(Point2f(i, j));
    }

    vector<Point2f> dst_p;
    perspectiveTransform(src_p, dst_p, H);

    Point2f Ot = dst_p[src.rows / 2 + src.cols /2];

    for (int i=0; i<dst_p.size(); ++i) {
        dst_p[i] -= offset;
    }

    Mat h = findHomography(src_p, dst_p);

    warpPerspective(src, dst, h, s);
    return Ot;
}

Mat taylor_series(const Mat &H, const Point anchor) {
    double dy1dx1 = H.at<double>(0,0)/(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y)
                   - (H.at<double>(2,0)*(H.at<double>(0,2) + H.at<double>(0,0)*anchor.x + H.at<double>(0,1)*anchor.y))/pow(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y, 2);

    double dy1dx2 = H.at<double>(0,1)/(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y)
                   - (H.at<double>(2,1)*(H.at<double>(0,2) + H.at<double>(0,0)*anchor.x + H.at<double>(0,1)*anchor.y))/pow(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y, 2);

    double dy2dx1 = H.at<double>(1,0)/(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y)
                   - (H.at<double>(2,0)*(H.at<double>(1,2) + H.at<double>(1,0)*anchor.x + H.at<double>(1,1)*anchor.y))/pow(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y, 2);

    double dy2dx2 = H.at<double>(1,1)/(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y)
                   - (H.at<double>(2,1)*(H.at<double>(1,2) + H.at<double>(1,0)*anchor.x + H.at<double>(1,1)*anchor.y))/pow(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y, 2);

    double y1 = (H.at<double>(0,2) + H.at<double>(0,0)*anchor.x + H.at<double>(0,1)*anchor.y)/(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y);
    double y2 = (H.at<double>(1,2) + H.at<double>(1,0)*anchor.x + H.at<double>(1,1)*anchor.y)/(H.at<double>(2,2) + H.at<double>(2,0)*anchor.x + H.at<double>(2,1)*anchor.y);

    Mat A(Size(3,3), CV_64F);

    A.at<double>(0,0) = dy1dx1;
    A.at<double>(0,1) = dy1dx2;
    A.at<double>(0,2) = y1 - anchor.x*dy1dx1 - anchor.y*dy1dx2;
    A.at<double>(1,0) = dy2dx1;
    A.at<double>(1,1) = dy2dx2;
    A.at<double>(1,2) = y2 - anchor.x*dy2dx1 - anchor.y*dy2dx2;
    A.at<double>(2,0) = A.at<double>(2,1) = 0;
    A.at<double>(2,2) = 1;

    //cout << A << endl;

    return A.clone();
}

Mat homography_linearization(const Mat &H, const Point &center_point, const vector<Point2f> &anchor_points) {
    const int vega = 5;

    Mat alpha(Size(1,anchor_points.size()), CV_64F);

    double sum_alpha = 0;
    for (int i=0; i<anchor_points.size(); ++i) {
        double dist = pow(center_point.x - anchor_points[i].x,2) + pow(center_point.y - anchor_points[i].y,2);

        double temp = pow(1 + dist / vega, -(vega+1)/2);

        alpha.at<double>(i) = temp;
        sum_alpha += temp;
    }

    alpha = alpha / sum_alpha;

    //cout << alpha << endl;


    Mat h_out(Size(3,3), CV_64F);
    h_out.setTo(Scalar::all(0));
    for (int i=0; i<anchor_points.size(); ++i) {
        Mat A = taylor_series(H, anchor_points[i]);
        h_out = h_out + alpha.at<double>(i) * A;
    }

    //cout << h_out << endl;
    return h_out.clone();
}