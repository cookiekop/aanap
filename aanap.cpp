//
// Created by Michael Zhang on 2018/7/27.
//

#include <iostream>
#include <cmath>

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "math.h"
#include "homography_warp.h"
#include "aanap.h"
#include "compute_weight.h"
#include "samplePoint.h"



using namespace std;

AANAPWarper::AANAPWarper()
{
    cell_height_ = 10;
    cell_width_ = 10;
    gamma_ = 0.1;
    sigma_ = 8.5;

    H_s_ = NULL;
    H_r_ = NULL;
}

/*
 *
 */
int AANAPWarper::buildMaps(Mat src_img, ImageFeatures src_features, ImageFeatures dst_features,
                          MatchesInfo matches_info, Mat &xmap, Mat &ymap, Point &corner , Mat dst_img)
{
    int matches_size = matches_info.matches.size();

    /*
     * 初始化矩阵A
     */
    //InitA(src_features, dst_features, matches_info);

    FindS(src_features, dst_features, matches_info, src_img);

    std::vector<Point2d> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( src_img.cols, 0 );
    obj_corners[2] = cvPoint( src_img.cols, src_img.rows ); obj_corners[3] = cvPoint( 0, src_img.rows );
    std::vector<Point2d> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, Hg_);

    double cw = 0, ch = 0;

    offset.x = 0;
    offset.y = 0;
    const int anchor_num = 20;
    double anchor_dist_w = src_img.cols / (anchor_num + 1);
    double anchor_dist_h = src_img.rows / (anchor_num + 1);



    for (int i=0; i<4; ++i) {
        if(dst_img.cols > cw) cw = dst_img.cols;
        if(dst_img.rows > ch) ch = dst_img.rows;

        /*if (obj_corners[i].x < offset.x) offset.x = obj_corners[i].x;
        if (obj_corners[i].y < offset.y) offset.y = obj_corners[i].y;*/
    }

    for (int i=0; i<4; ++i) {
        if(scene_corners[i].x > cw) cw = scene_corners[i].x;
        if(scene_corners[i].y > ch) ch = scene_corners[i].y;

        if (scene_corners[i].x < offset.x) offset.x = scene_corners[i].x;
        if (scene_corners[i].y < offset.y) offset.y = scene_corners[i].y;
    }

    cw = cw - offset.x;
    ch = ch - offset.y;

    for(int i=1; i<anchor_num+1; ++i)
        anchor_points.emplace_back(Point2d(anchor_dist_w * i, 0));

    for(int i=1; i<anchor_num+1; ++i)
        anchor_points.emplace_back(Point2d(anchor_dist_w * i, src_img.rows));

    for(int i=1; i<anchor_num+1; ++i)
        anchor_points.emplace_back(Point2d(src_img.cols, anchor_dist_h * i));

    for(int i=1; i<anchor_num+1; ++i)
        anchor_points.emplace_back(Point2d(0, anchor_dist_h * i));


    Mat h_pano;

    Point2d Ot = homography_warp(src_img, Hg_, offset, Size(cvCeil(cw), cvCeil(ch)), h_pano);

    for (int i=0; i<dst_img.rows; ++i)
    {
        for (int j=0; j<dst_img.cols; ++j)
        {
            h_pano.at<Vec3b>(cvRound(i - offset.y),cvRound(j - offset.x)) = dst_img.at<Vec3b>(i,j);
        }
    }

    for (int i=0; i<anchor_points.size(); ++i)
        circle(h_pano, anchor_points[i]-offset, 3,Scalar(255,0,0),-1);

    imwrite("homography.jpg", h_pano);

    Point2d Or(dst_img.cols /2, dst_img.rows /2);

    double k,b;
    k = (Or.y - Ot.y) / (Or.x - Ot.x);
    b = Or.y - k * Or.x;

    cout << k << " " << b << endl;

    K_max_.x = 0;
    for (int i=0; i<4; ++i)
        if (scene_corners[i].x > K_max_.x) K_max_.x = scene_corners[i].x;

    K_min_.x = 99999;
    for (int i=0; i<4; ++i)
        if (scene_corners[i].x < K_min_.x) K_min_.x = scene_corners[i].x;

    K_1_.x = dst_img.cols;
    K_2_.x =  K_1_.x + (K_max_.x - K_1_.x)/2;

    K_1_.y = k*K_1_.x + b;
    K_2_.y = k*K_2_.x + b;
    K_min_.y = k*K_min_.x + b;
    K_max_.y = k*K_max_.x + b;

    cout << K_max_ << " " << K_min_ << endl;
    cout << K_1_ << " " << K_2_ << endl;

    /*
     * 每一个cell做一次DLT
     */
    int inliner_nums = matches_info.num_inliers;
    // 初始化W_
    W_ = Mat_<apap_float>::zeros(2 * matches_size, 2 * matches_size);

    // 存放每一个cell对应的H
    if(H_s_ != NULL)
    {
        for(int i = 0; i < cell_rows_; i++)
            delete []H_s_[i];
        delete []H_s_;
        H_s_ = NULL;
    }
    if(H_r_ != NULL)
    {
        for(int i = 0; i < cell_rows_; i++)
            delete []H_r_[i];
        delete []H_r_;
        H_r_ = NULL;
    }
    int src_rows = cvCeil(ch);
    int src_cols = cvCeil(cw);
    cell_rows_ = cvCeil((double)(src_rows) / cell_height_);
    cell_cols_ = cvCeil((double)(src_cols) / cell_width_);

    H_s_ = new Mat_<apap_float> *[cell_rows_];
    for(int i = 0; i < cell_rows_; i++)
        H_s_[i] = new Mat_<apap_float> [cell_cols_];

    H_r_ = new Mat_<apap_float> *[cell_rows_];
    for(int i = 0; i < cell_rows_; i++)
        H_r_[i] = new Mat_<apap_float> [cell_cols_];


    //Mat_<apap_float> H(3, 3);

    for(int x = 0; x < src_cols; x += cell_width_)
        for(int y = 0; y < src_rows; y += cell_height_)
            CellDLT(x, y, src_features, dst_features, matches_info, H_s_[y / cell_height_][x / cell_width_], H_r_[y / cell_height_][x / cell_width_], dst_img);

    cout << "aanap: MDLT completed" << endl;

    /*
     * Remap
     * 反向映射（Reverse mapping，结果图片中的每个点对应src的坐标）
     */
    //	1、计算映射结果图片的尺寸，只需要把四条边映射一下就可以了
    /*Point2d corner_2d(H_s_[0][0](0, 2) / H_s_[0][0](2, 2), H_s_[0][0](1, 2) / H_s_[0][0](2, 2));
    Point2d br(corner_2d);
    for(int x = 0; x < src_cols; x ++)
    {
        //上面的边
        updateMappedSize(H_s_[0][x / cell_width_], x, 0, corner_2d, br);
        //下面的边
        updateMappedSize(H_s_[(src_rows-1) / cell_height_][x / cell_width_],
                         x, src_rows - 1, corner_2d, br);
    }

    for(int y = 0; y < src_rows; y ++)
    {
        // 左边
        updateMappedSize(H_s_[y / cell_height_][0], 0, y, corner_2d, br);
        // 右边
        updateMappedSize(H_s_[y / cell_height_][(src_cols-1) / cell_width_],
                         src_cols - 1, y, corner_2d, br);
    }

    cout << corner_2d << endl;
    cout << br << endl;

    //	2、反向映射
    corner.x = cvFloor(corner_2d.x);
    corner.y = cvFloor(corner_2d.y);
    int result_width = cvCeil(br.x) + 1 - corner.x;
    int result_height = cvCeil(br.y) + 1 - corner.y;*/


    xmap.create(src_rows, src_cols, CV_32F);
    ymap.create(src_rows, src_cols, CV_32F);
    float *xmap_rev_ptr = xmap.ptr<float>(0);
    float *ymap_rev_ptr = ymap.ptr<float>(0);
    for(int i = 0; i < src_cols * src_rows; i++)
        xmap_rev_ptr[i] = ymap_rev_ptr[i] = -1;

    corner = offset;
    for(int x = 0; x < src_cols; x += cell_width_)
    {
        for(int y = 0; y < src_rows; y += cell_height_)
        {
            int xx_max = std::min(x + cell_width_, src_cols);
            int yy_max = std::min(y + cell_height_, src_rows);
            BuildCellMap(x, y, xx_max, yy_max, H_s_[y / cell_height_][x / cell_width_], corner,
                         xmap, ymap);
        }
    }

    Mat src_out;
    remap(src_img, src_out, xmap, ymap, INTER_LINEAR);
    imwrite("src_out.jpg", src_out);

    cout << "aanap: build map completed" << endl;

    return 0;
}

int AANAPWarper::buildMaps( vector<Mat> imgs, vector<ImageFeatures> features,
                           vector<MatchesInfo> pairwise_matches, vector<Mat> &xmaps, vector<Mat> &ymaps, vector<Point> &corners )
{
    int n = imgs.size();

    int mid_idx = n / 2;



    for(int i = 0; i < n; ++i)
    {
        if(i == mid_idx)
            continue;
        int pair_idx = i * n + mid_idx;
        buildMaps(imgs[i], features[i], features[mid_idx], pairwise_matches[pair_idx],
                  xmaps[i], ymaps[i], corners[i], imgs[mid_idx]);
    }

    // 基准图片的map

    xmaps[mid_idx].create(xmaps[0].size(), CV_32F);
    ymaps[mid_idx].create(xmaps[0].size(), CV_32F);
    float *xmap_rev_ptr = xmaps[mid_idx].ptr<float>(0);
    float *ymap_rev_ptr = ymaps[mid_idx].ptr<float>(0);
    for(int i = 0; i < xmaps[0].rows * xmaps[0].cols; i++)
        xmap_rev_ptr[i] = ymap_rev_ptr[i] = -1;

    corners[mid_idx] = offset;
    for(int y = 0; y < xmaps[0].rows; y++)
    {
        for(int x = 0; x < xmaps[0].cols; x++)
        {
            int xx_max = std::min(x + cell_width_, xmaps[0].cols);
            int yy_max = std::min(y + cell_height_, xmaps[0].rows);
            BuildCellMap(x, y, xx_max, yy_max, H_r_[y / cell_height_][x / cell_width_], corners[mid_idx],
                         xmaps[mid_idx], ymaps[mid_idx]);
        }
    }

    cout << "aanap: dst map completed" << endl;

    Mat dst_out;
    remap(imgs[mid_idx], dst_out, xmaps[mid_idx], ymaps[mid_idx], INTER_LINEAR);
    imwrite("dst_out.jpg", dst_out);

    /*for(int i = mid_idx+1; i >= (mid_idx-1); i--)
    {
        if(i == mid_idx)
            continue;
        int pair_idx = i * n + mid_idx;
        buildMaps(imgs[i], features[i], features[mid_idx], pairwise_matches[pair_idx],
                  xmaps[i], ymaps[i], corners[i]);
    }*/

    if(n < 4)
        return 0;

    /*ImageFeatures features_warped;
    int feat_num = features[1].keypoints.size();
    features_warped.keypoints.resize(feat_num);
    features_warped.img_idx = features[1].img_idx;
    features_warped.img_size = features[1].img_size;
    Mat_<apap_float> tmp_X(3, 1), tmp_res_X;
    for(int i = 0; i < feat_num; i++)
    {
        tmp_X(0, 0) = features[1].keypoints[i].pt.x;
        tmp_X(1, 0) = features[1].keypoints[i].pt.y;
        tmp_X(2, 0) = 1;
        int x = cvRound(tmp_X(0, 0));
        int y = cvRound(tmp_X(1, 0));
        tmp_res_X = H_s_[y / cell_height_][x / cell_width_] * tmp_X;
        features_warped.keypoints[i].pt.x = (double)(tmp_res_X(0, 0) / tmp_res_X(2, 0) - corners[1].x);
        features_warped.keypoints[i].pt.y = (double)(tmp_res_X(1, 0) / tmp_res_X(2, 0) - corners[1].y);
    }
    buildMaps(imgs[0], features[0], features_warped, pairwise_matches[1],
              xmaps[0], ymaps[0], corners[0], );
    corners[0].x += corners[1].x;
    corners[0].y += corners[1].y;*/

    return 0;
}




int AANAPWarper::CellDLT( int offset_x, int offset_y, ImageFeatures src_features,
                         ImageFeatures dst_features, MatchesInfo matches_info, Mat_<apap_float> &H, Mat_<apap_float> &H_r, const Mat &dst_img)
{
    int matches_size = matches_info.matches.size();
    int inliner_nums = matches_info.num_inliers;

    double center_x = offset_x + cell_width_ / 2 + offset.x;
    double center_y = offset_y + cell_height_ / 2 + offset.y;

    for(int j = 0, inliner_idx = 0; j < matches_size; j ++)
    {
        /*if (!matches_info.inliers_mask[j])
            continue;*/

        const DMatch& m = matches_info.matches[j];
        Point2d p1 = src_features.keypoints[m.queryIdx].pt;
        Point2d p2 = dst_features.keypoints[m.trainIdx].pt;

        double weight = exp(-(pow(center_x - p1.x,2) + pow(center_y - p1.y,2)) / pow(sigma_,2));

        int i = inliner_idx * 2;
        W_(i, i) = W_(i+1, i+1) = std::max(weight, gamma_);
        inliner_idx ++;
    }


    //cout << W_.size() << " " << A_.size() << endl;

    Mat WA = W_ * A_;
    Mat_<apap_float> h(9, 1);
    SVD::solveZ(WA, h);
    //eigen2cv(h_e, h);

    vector<double> ab = compute_weight(Point2d(center_x, center_y), K_min_, K_max_);
    vector<double> cd = compute_weight(Point2d(center_x, center_y), K_1_, K_2_);


    /*
    printf("(");
    for(int i = 0; i < 9; i++)
        printf("%lf, ", H(i, 0) / H(8, 0));
    printf(")\n");
    */
    if(H.empty())
        H.create(3,3);
    apap_float *H_ptr = H.ptr<apap_float>(0);
    apap_float *h_ptr = h.ptr<apap_float>(0);
    for(int i = 0; i < 9; i++)
        H_ptr[i] = h_ptr[i];

    H = T_dst_.inv() * (H * T_src_);
    H = H / H.at<double>(2,2);
    //Mat H_inv = H.inv();
    //H_inv = H_inv / H_inv.at<double>(8);

    //cout << H_inv << endl;

    if (center_x > dst_img.cols || center_y > dst_img.rows || center_x < 0 || center_y < 0)
    {
        Mat h_linear = homography_linearization(H, Point2d(center_x, center_y), anchor_points);
        H = cd[0] * h_linear + cd[1] * H;
    }

    Mat h_invert = H.inv();

    //cout << ab[0] << " " << ab[1] << endl;

    H = S_ * ab[0] + H * ab[1];

    if(H_r.empty())
        H_r.create(3,3);
    H_r = H * h_invert;

    if (center_x > dst_img.cols) H_r.setTo(Scalar::all(0));

    //H = H.inv();
    //H_r = H_r.inv();
    return 0;
}


void AANAPWarper::FindS(ImageFeatures src_features,	ImageFeatures dst_features, MatchesInfo matches_info, const Mat src)
{
    cout << "init S" << endl;
    vector<Point2d> src_p, dst_p;

    Point2d center_src(0,0), center_dst(0,0);

    for (int i = 0; i < matches_info.matches.size(); ++i)
    {
        //if (!matches_info.inliers_mask[i]) continue;
        Point2d tmp = src_features.keypoints[matches_info.matches[i].queryIdx].pt;
        src_p.push_back(tmp);
        center_src.x += tmp.x;
        center_src.y += tmp.y;

        tmp = dst_features.keypoints[matches_info.matches[i].trainIdx].pt;
        dst_p.push_back(tmp);
        center_dst.x += tmp.x;
        center_dst.y += tmp.y;
    }

    Mat drawing = src.clone();
    for (int j=0; j< src_p.size(); ++j)
        circle(drawing, src_p[j], 3,Scalar(255,0,0),-1);
    imwrite("pivot.jpg", drawing);

    center_dst.x /= src_p.size();
    center_dst.y /= src_p.size();
    center_src.y /= src_p.size();
    center_src.x /= src_p.size();

    double mean_dist_src = 0, mean_dist_dst = 0;
    for (int i=0; i<src_p.size(); ++i) {
        double dist = sqrt(pow(src_p[i].x - center_src.x, 2) + pow(src_p[i].y - center_src.y, 2));
        mean_dist_src += dist;

        dist = sqrt(pow(dst_p[i].x - center_dst.x, 2) + pow(dst_p[i].y - center_dst.y, 2));
        mean_dist_dst += dist;
    }
    mean_dist_src /= src_p.size();
    mean_dist_dst /= src_p.size();

    double scale_src = sqrt(2)/mean_dist_src;
    double scale_dst = sqrt(2)/mean_dist_dst;
    
    T_src_.create(Size(3,3), CV_64F);
    T_dst_.create(Size(3,3), CV_64F);

    T_src_.at<double>(0,0) = T_src_.at<double>(1,1) = scale_src;
    T_src_.at<double>(0,1) = T_src_.at<double>(1,0) = T_src_.at<double>(2,0) = T_src_.at<double>(2,1) = 0;
    T_src_.at<double>(0,2) = -scale_src*center_src.x;
    T_src_.at<double>(1,2) = -scale_src*center_src.y;
    T_src_.at<double>(2,2) = 1;

    T_dst_.at<double>(0,0) = T_dst_.at<double>(1,1) = scale_dst;
    T_dst_.at<double>(0,1) = T_dst_.at<double>(1,0) = T_dst_.at<double>(2,0) = T_dst_.at<double>(2,1) = 0;
    T_dst_.at<double>(0,2) = -scale_dst*center_dst.x;
    T_dst_.at<double>(1,2) = -scale_dst*center_dst.y;
    T_dst_.at<double>(2,2) = 1;

    A_ = Mat_<apap_float>::zeros(2 * matches_info.matches.size(), 9);
    //vector<Point2d> in_src, in_dst, out_src, out_dst;
    for (int i=0 , inliner_idx = 0; i<src_p.size(); ++i) {
        Mat temp_pt(Size(1,3), CV_64F);
        temp_pt.at<double>(0) = src_p[i].x;
        temp_pt.at<double>(1) = src_p[i].y;
        temp_pt.at<double>(2) = 1;
        temp_pt = T_src_ * temp_pt;
        src_p[i].x = temp_pt.at<double>(0);
        src_p[i].y = temp_pt.at<double>(1);

        temp_pt.at<double>(0) = dst_p[i].x;
        temp_pt.at<double>(1) = dst_p[i].y;
        temp_pt.at<double>(2) = 1;
        temp_pt = T_dst_ * temp_pt;
        dst_p[i].x = temp_pt.at<double>(0);
        dst_p[i].y = temp_pt.at<double>(1);

        int j = inliner_idx * 2;

        A_(j, 0) = A_(j, 1) = A_(j, 2) = 0;
        A_(j, 3) = -src_p[i].x;
        A_(j, 4) = -src_p[i].y;
        A_(j, 5) = -1;
        A_(j, 6) = dst_p[i].y * src_p[i].x;
        A_(j, 7) = dst_p[i].y * src_p[i].y;
        A_(j, 8) = dst_p[i].y;

        A_(j + 1, 0) = src_p[i].x;
        A_(j + 1, 1) = src_p[i].y;
        A_(j + 1, 2) = 1;
        A_(j + 1, 3) = A_(j + 1, 4) = A_(j + 1, 5) = 0;
        A_(j + 1, 6) = -dst_p[i].x * src_p[i].x;
        A_(j + 1, 7) = -dst_p[i].x * src_p[i].y;
        A_(j + 1, 8) = -dst_p[i].x;

        inliner_idx ++;

        /*if (matches_info.inliers_mask[i]) {
            in_src.push_back(src_p[i]);
            in_dst.push_back(dst_p[i]);

        }
        else {
            out_src.push_back(src_p[i]);
            out_dst.push_back(dst_p[i]);
        }*/
    }

    //cout << A_ << endl;
    //vector<uchar> RansacStatus;
    Mat_<apap_float> hg(9, 1);
    SVD::solveZ(A_, hg);
    if(Hg_.empty())
        Hg_.create(Size(3,3), CV_64F);
    for(int i = 0; i < 9; i++)
        Hg_.at<double>(i) = hg.at<double>(i);
    //Hg_ = findHomography(in_src, in_dst);
    Hg_ = T_dst_.inv() * (Hg_ * T_src_);
    Hg_ = Hg_ / Hg_.at<double>(2,2);
    cout << Hg_ << endl;





    vector<Mat> S_segment;
    vector<double> theta;
    int t = 0;
    while (1) {
        vector<Point2d> RR_keypoint01, RR_keypoint02;
        samplePoint(src_p, dst_p, 50, 1.5, RR_keypoint01, RR_keypoint02);
        /*vector<uchar> RansacStatus;
        Mat Fundamental = findHomography(src_p, dst_p, FM_RANSAC, 1, RansacStatus, 2000, 0.995);

        int i = 0;
        RR_keypoint01.clear();
        RR_keypoint02.clear();
        while (1)
        {
            if (i >= src_p.size()) break;
            if (RansacStatus[i] != 0)
            {
                RR_keypoint01.push_back(src_p[i]);
                RR_keypoint02.push_back(dst_p[i]);
                src_p.erase(src_p.begin() + i);
                dst_p.erase(dst_p.begin() + i);
            }
            ++i;
        }*/

        if (RR_keypoint01.size() < 50) break;

        Mat A, b;
        A.create(RR_keypoint01.size() * 2, 4, CV_64F);
        b.create(RR_keypoint01.size() * 2, 1, CV_64F);
        for (int j=0, index = 0; j<RR_keypoint01.size(); ++j) {

            double  *ptr_A1 = A.ptr<double>(index);
            double  *ptr_A2 = A.ptr<double>(index+1);
            double *ptr_b1 = b.ptr<double>(index);
            double *ptr_b2 = b.ptr<double>(index+1);

            ptr_A1[0] = RR_keypoint01[j].x;
            ptr_A1[1] = -RR_keypoint01[j].y;
            ptr_A1[2] = 1;
            ptr_A1[3] = 0;

            ptr_A2[0] = RR_keypoint01[j].y;
            ptr_A2[1] = RR_keypoint01[j].x;
            ptr_A2[2] = 0;
            ptr_A2[3] = 1;

            ptr_b1[0] = RR_keypoint02[j].x;
            ptr_b2[0] = RR_keypoint02[j].y;

            index += 2;
        }

        //cout << A << endl;
        Mat invert_A = A.inv(DECOMP_SVD);
        //cv::invert(A, invert_A, CV_SVD);
        Mat beta = invert_A * b;
        //cout << beta << endl;

        theta.push_back(atan(beta.at<double>(1) / beta.at<double>(0)));

        Mat S_temp(Size(3,3), CV_64F);
        S_temp.at<double>(0,0) = beta.at<double>(0);
        S_temp.at<double>(0,1) = -beta.at<double>(1);
        S_temp.at<double>(0,2) = beta.at<double>(2);
        S_temp.at<double>(1,0) = beta.at<double>(1);
        S_temp.at<double>(1,1) = beta.at<double>(0);
        S_temp.at<double>(1,2) = beta.at<double>(3);
        S_temp.at<double>(2,0) = S_temp.at<double>(2,1) = 0;
        S_temp.at<double>(2,2) = 1;

        S_segment.push_back(S_temp);

        Mat drawing = src.clone();
        for (int j=0; j< RR_keypoint01.size(); ++j) {
            Point2d point_norm;
            Mat temp_pt(Size(1,3), CV_64F);
            temp_pt.at<double>(0) = RR_keypoint01[j].x;
            temp_pt.at<double>(1) = RR_keypoint01[j].y;
            temp_pt.at<double>(2) = 1;
            temp_pt = T_src_.inv() * temp_pt;
            point_norm.x = temp_pt.at<double>(0);
            point_norm.y = temp_pt.at<double>(1);
            circle(drawing, point_norm, 3,Scalar(255,0,0),-1);
        }

        imwrite("ransac"+to_string(t++)+".jpg", drawing);
    }

    double theta_min = 9999;
    int index = 0;
    for (int i=0; i<t; ++i) {
        if (abs(theta[i]) < theta_min) {
            theta_min = abs(theta[i]);
            index = i;
        }
    }

    cout << index << " " << theta_min << endl;

    S_ = S_segment[index];

    S_ = T_dst_.inv() * (S_ * T_src_);
    cout << S_ << endl;
}


void AANAPWarper::InitA( ImageFeatures src_features, ImageFeatures dst_features, MatchesInfo matches_info )
{
    printf("init A and H\n");

    int matches_size = matches_info.matches.size();
    int inliner_nums = matches_info.num_inliers;
    A_ = Mat_<apap_float>::zeros(2 * inliner_nums, 9);

    Mat src_points(1, inliner_nums, CV_32FC2);
    Mat dst_points(1, inliner_nums, CV_32FC2);

    for(int j = 0, inliner_idx = 0; j < matches_size; j++)
    {
        if (!matches_info.inliers_mask[j])
            continue;

        const DMatch& m = matches_info.matches[j];
        Point2f p1 = src_features.keypoints[m.queryIdx].pt;
        Point2f p2 = dst_features.keypoints[m.trainIdx].pt;
        src_points.at<Point2f>(0, inliner_idx) = p1;
        dst_points.at<Point2f>(0, inliner_idx) = p2;

        int i = inliner_idx * 2;

        A_(i, 0) = A_(i, 1) = A_(i, 2) = 0;
        A_(i, 3) = -p1.x;
        A_(i, 4) = -p1.y;
        A_(i, 5) = -1;
        A_(i, 6) = p2.y * p1.x;
        A_(i, 7) = p2.y * p1.y;
        A_(i, 8) = p2.y;

        A_(i + 1, 0) = p1.x;
        A_(i + 1, 1) = p1.y;
        A_(i + 1, 2) = 1;
        A_(i + 1, 3) = A_(i + 1, 4) = A_(i + 1, 5) = 0;
        A_(i + 1, 6) = -p2.x * p1.x;
        A_(i + 1, 7) = -p2.x * p1.y;
        A_(i + 1, 8) = -p2.x;

        inliner_idx ++;
    }

    Hg_ = findHomography(src_points, dst_points);

    //cout << A_ << endl;
    cout << Hg_ << endl;
}

void AANAPWarper::testh( ImageFeatures src_features, ImageFeatures dst_features,
                        MatchesInfo matches_info, Mat h )
{
    int matches_size = matches_info.matches.size();
    double *h_ptr = h.ptr<double>(0);
    for(int i = 0; i < matches_size; i ++)
    {
        if (!matches_info.inliers_mask[i])
            continue;
        const DMatch& m = matches_info.matches[i];
        Point2f p1 = src_features.keypoints[m.queryIdx].pt;
        Point2f p2 = dst_features.keypoints[m.trainIdx].pt;

        double _z = h_ptr[6] * p1.x + h_ptr[7] * p1.y + h_ptr[8];
        double _y = h_ptr[3] * p1.x + h_ptr[4] * p1.y + h_ptr[5];
        double _x = h_ptr[0] * p1.x + h_ptr[1] * p1.y + h_ptr[2];

        _x = _x / _z;
        _y = _y / _z;

        if(i < 20)
            printf("%.7lf, %.7lf => %.7lf, %.7lf\n", p2.x, p2.y, _x, _y);
    }
    printf("\n");
}

void AANAPWarper::updateMappedSize( Mat_<apap_float> H, double x, double y,
                                   Point2d &corner, Point2d &br )
{
    double _z = H(2, 0) * x + H(2, 1) * y + H(2, 2);
    double _y = (H(1, 0) * x + H(1, 1) * y + H(1, 2)) / _z;
    double _x = (H(0, 0) * x + H(0, 1) * y + H(0, 2)) / _z;

    corner.x = std::min(corner.x, _x);
    corner.y = std::min(corner.y, _y);
    br.x = std::max(br.x, _x);
    br.y = std::max(br.y, _y);
}

void AANAPWarper::BuildCellMap( int x_1, int y_1, int x_2, int y_2, Mat H, Point corner, Mat &xmap, Mat &ymap )
{
    float *xmap_rev_ptr = xmap.ptr<float>(0);
    float *ymap_rev_ptr = ymap.ptr<float>(0);
    int res_width = xmap.size().width;
    int res_height = xmap.size().height;

    Mat_<apap_float> tmp_corner_X[4], tmp_corner_res_X[4];	// 四个角的坐标
    for(int i = 0; i < 4; i++)
        tmp_corner_X[i] = Mat_<apap_float>::ones(3, 1);
    tmp_corner_X[0](0, 0) = tmp_corner_X[3](0, 0) = x_1;
    tmp_corner_X[0](1, 0) = tmp_corner_X[1](1, 0) = y_1;
    tmp_corner_X[1](0, 0) = tmp_corner_X[2](0, 0) = x_2;
    tmp_corner_X[2](1, 0) = tmp_corner_X[3](1, 0) = y_2 ;

    for(int i = 0; i < 4; i++)
    {
        tmp_corner_res_X[i] = H * tmp_corner_X[i];
        tmp_corner_res_X[i] = tmp_corner_res_X[i] / tmp_corner_res_X[i](2, 0);
    }
    //cout << "1" << endl;
    // 先找到x和y的范围
    double y_min_d, y_max_d;
    double x_min_d, x_max_d;
    y_min_d = y_max_d = tmp_corner_res_X[0](1, 0);
    x_min_d = x_max_d = tmp_corner_res_X[0](0, 0);
    for(int i = 1; i < 4; i++)
    {
        double y = tmp_corner_res_X[i](1, 0);
        double x = tmp_corner_res_X[i](0, 0);
        y_min_d = std::min(y, y_min_d);
        y_max_d = std::max(y, y_max_d);
        x_min_d = std::min(x, x_min_d);
        x_max_d = std::max(x, x_max_d);
    }

    //cout << "2" << endl;
    // 扫描线
    int y_min = cvFloor(y_min_d), y_max = cvCeil(y_max_d);
    double x1, x2, y1, y2, x_start_d, x_end_d;
    int x_start, x_end;
    Mat_<apap_float> H_inv = H.inv(), tmp_X = Mat_<apap_float>::ones(3, 1), tmp_mapped_X(3, 1);
    double delta = 1.2;
    for(int y = y_min; y <= y_max; y++)
    {
        // 与4条线的交点
        x_start_d = x_max_d;
        x_end_d = x_min_d;
        for(int i = 0; i < 4; i++)
        {
            x2 = tmp_corner_res_X[i](0, 0);
            y2 = tmp_corner_res_X[i](1, 0);
            x1 = tmp_corner_res_X[(i+1) % 4](0, 0);
            y1 = tmp_corner_res_X[(i+1) % 4](1, 0);
            if(std::abs(y1 - y2) < 0.2)
                continue;

            if((y > (y1+delta) && y > (y2+delta)) || (y < (y1-delta) && y < (y2-delta)))
                continue;
            double p_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
            x_start_d = std::min(x_start_d, p_x);
            x_end_d = std::max(x_end_d, p_x);
        }
        x_start = cvFloor(x_start_d);
        x_end = cvCeil(x_end_d);
        tmp_X(1, 0) = y;
        //cout << "3" << endl;
        for(int x = x_start; x <= x_end; x++)
        {
            int x_idx = x - corner.x;
            int y_idx = y - corner.y;
            if(x_idx >= res_width || y_idx >= res_height || x_idx < 0 || y_idx < 0)
                continue;
            tmp_X(0, 0) = x;
            tmp_mapped_X = H_inv * tmp_X;
            int idx = y_idx * res_width + x_idx;
            xmap_rev_ptr[idx] = (float)(tmp_mapped_X(0, 0) / tmp_mapped_X(2, 0));
            ymap_rev_ptr[idx] = (float)(tmp_mapped_X(1, 0) / tmp_mapped_X(2, 0));
        }
    }

}

AANAPWarper::~AANAPWarper()
{
    if(H_s_ != NULL)
    {
        for(int i = 0; i < cell_rows_; i++)
            delete []H_s_[i];
        delete []H_s_;
        H_s_ = NULL;
    }
    if(H_s_ != NULL)
    {
        for(int i = 0; i < cell_rows_; i++)
            delete []H_r_[i];
        delete []H_r_;
        H_r_ = NULL;
    }
}