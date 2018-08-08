#include <iostream>
#include "aanap.h"
#include "opencv2/opencv.hpp"
#include "blender.h"
#include "opencv2/xfeatures2d.hpp"




using namespace cv;

#define IMG_NUM 2

vector<Mat> xmaps_, ymaps_, final_warped_masks_;
Rect dst_roi_;
vector<Point> corners_;
vector<Size> sizes_;
const int parallel_num_ = 4;
vector<Mat> final_blend_masks_;

vector<Mat> blend_weight_maps_;

int StitchFrame( vector<Mat> &src, Mat &dst)
{
    vector<Mat> final_warped_images_(IMG_NUM);
    bool time_debug = false;//true;//
    long start_clock, end_clock;

    if(time_debug)
        start_clock = clock();

    int64 t;
    int num_images = src.size();

    int dst_width = dst_roi_.width;
    int dst_height = dst_roi_.height;
    if(dst.empty())
        dst.create(dst_roi_.size(), CV_8UC3);
    uchar *dst_ptr_00 = dst.ptr<uchar>(0);
    memset(dst_ptr_00, 0, dst_width * dst_height * 3);

    double warp_time[100], feed_time[100];

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        if(time_debug)
            t = getTickCount();

        // Warp the current image
        remap(src[img_idx], final_warped_images_[img_idx], xmaps_[img_idx], ymaps_[img_idx],
              INTER_LINEAR);//, BORDER_REFLECT);
        if(time_debug)
            warp_time[img_idx] = 1000 * (getTickCount() - t) / getTickFrequency();

        if(time_debug)
            t = getTickCount();
        int dx = corners_[img_idx].x - dst_roi_.x;
        int dy = corners_[img_idx].y - dst_roi_.y;
        int img_rows = sizes_[img_idx].height;
        int img_cols = sizes_[img_idx].width;
        int src_rows = src[img_idx].rows;
        int src_cols = src[img_idx].cols;

        int rows_per_parallel = img_rows / parallel_num_;
#pragma omp parallel for
        for(int parallel_idx = 0; parallel_idx < parallel_num_; parallel_idx++)
        {
            int row_start = parallel_idx * rows_per_parallel;
            int row_end = row_start + rows_per_parallel;
            if(parallel_idx == parallel_num_ - 1)
                row_end = img_rows;

            uchar *dst_ptr;
            uchar *warped_img_ptr	= final_warped_images_[img_idx].ptr<uchar>(row_start);
            float *total_weight_ptr	= blend_weight_maps_[img_idx].ptr<float>(row_start);
            for(int y = row_start; y < row_end; y++)
            {
                dst_ptr = dst_ptr_00 + ((dy + y) * dst_width + dx) * 3;
                for(int x = 0; x < img_cols; x++)
                {
                    (*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
                    warped_img_ptr++;
                    dst_ptr++;

                    (*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
                    warped_img_ptr++;
                    dst_ptr++;

                    (*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
                    warped_img_ptr++;
                    dst_ptr++;

                    total_weight_ptr++;
                }
            }
        }


        if(time_debug)
            feed_time[img_idx] = 1000 * (getTickCount() - t) / getTickFrequency();
    }

    if(time_debug)
        for(int i = 0; i < num_images; i++)
            cout << "\twarp " << warp_time[i] << "ms, feed " << feed_time[i] << "ms" << endl;

    if(time_debug)
        cout << "(=" << clock()-start_clock << "ms)"<<endl;

    return 0;
}


int main() {
    vector<Mat> src;
    for(int i=5; i<7; ++i)
        src.push_back(imread("./temple_"+to_string(i)+".jpg"));

    /*for(int i=10; i<13; ++i)
        src.push_back(imread("./"+to_string(i)+".jpg"));*/

    Ptr<cv::xfeatures2d::SIFT> sift=cv::xfeatures2d::SIFT::create(0, 3, 0.01, 40);
    vector<ImageFeatures> features(IMG_NUM);
    for(int i=0; i<IMG_NUM;++i) {
        /*UMat descriptors;
        detectPoints(src[i], features[i].keypoints, descriptors);
        features[i].img_idx = i;
        features[i].img_size = src[i].size();*/
        //resize(src[i], src[i], Size(1000,800));
        features[i].img_idx = i;
        features[i].img_size = src[i].size();
        sift->detect(src[i], features[i].keypoints);
        sift->compute(src[i], features[i].keypoints, features[i].descriptors);
        cout << features[i].keypoints.size() << endl;
    }

    vector<MatchesInfo> matches_info;
    BestOf2NearestMatcher matcher(false, 0.5f);
    matcher(features, matches_info);
    matcher.collectGarbage();
    for (int i=0; i< IMG_NUM; ++i) {
        for (int j = i + 1; j < IMG_NUM; ++j) {
            int pair_idx = i*IMG_NUM + j;
            Mat display_match;
            //cout << matches_info[pair_idx].num_inliers << endl;
            drawMatches(src[i], features[i].keypoints, src[j], features[j].keypoints, matches_info[pair_idx].matches, display_match);
            cout << matches_info[pair_idx].matches.size() << endl;
            imwrite("matches" + to_string(pair_idx) + ".jpg", display_match);
        }
    }

    /*BFMatcher matcher;
    vector<vector< DMatch >>  matches;
    for (int i=0; i< IMG_NUM; ++i) {
        for (int j=i+1; j<IMG_NUM; ++j) {
            vector< DMatch > t_match, good_match;
            matcher.match( features[i].descriptors, features[j].descriptors, t_match);
            findGoodMatches(t_match, good_match);
            cout << good_match.size() << " " << t_match.size() <<endl;
            matches.push_back(good_match);

            Mat display_match;
            drawMatches(src[i], features[i].keypoints, src[j], features[j].keypoints, good_match, display_match);
            imwrite("matches" + to_string(i * IMG_NUM + j) + ".jpg", display_match);
        }
    }*/






    corners_.resize(IMG_NUM);
    sizes_.resize(IMG_NUM);
    xmaps_.resize(IMG_NUM);
    ymaps_.resize(IMG_NUM);
    Ptr<AANAPWarper> warper = new AANAPWarper();
    warper->buildMaps(src, features, matches_info, xmaps_, ymaps_, corners_);

    for(int i = 0; i < IMG_NUM; ++i)
        sizes_[i] = xmaps_[i].size();
    dst_roi_ = resultRoi(corners_, sizes_);

    vector<Mat> seamed_masks(IMG_NUM);
    vector<Mat> images_warped(IMG_NUM);
    vector<Mat> init_masks(IMG_NUM);
    final_warped_masks_.resize(IMG_NUM);
    for(int i = 0; i < IMG_NUM; i++)
    {
        init_masks[i].create(src[i].size(), CV_8U);
        init_masks[i].setTo(Scalar::all(255));
        remap(src[i], images_warped[i], xmaps_[i], ymaps_[i], INTER_LINEAR);
        remap(init_masks[i], final_warped_masks_[i], xmaps_[i], ymaps_[i], INTER_NEAREST, BORDER_CONSTANT);
        seamed_masks[i] = final_warped_masks_[i].clone();
    }

    Ptr<SeamFinder> seam_finder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    vector<UMat> images_warped_f(IMG_NUM), seamed_masks_u(IMG_NUM);
    for (int i = 0; i < IMG_NUM; ++i) {
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
        seamed_masks[i].convertTo(seamed_masks_u[i], CV_8U);
    }
    seam_finder->find(images_warped_f, corners_, seamed_masks_u);
    for (int i = 0; i < IMG_NUM; ++i) {
        seamed_masks_u[i].convertTo(seamed_masks[i], CV_8U);
    }


    images_warped.clear();

    MyFeatherBlender blender;
    Size dst_sz = dst_roi_.size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * 5 / 100.f;
    blender.setSharpness(1.f / blend_width);
    final_blend_masks_.resize(IMG_NUM);
    for(int i = 0; i < IMG_NUM; i++) {
        final_blend_masks_[i] = final_warped_masks_[i] & seamed_masks[i];
        //imwrite("./mask"+to_string(i)+".jpg", final_blend_masks_[i]);
    }

    blender.createWeightMaps(dst_roi_, corners_, final_blend_masks_, blend_weight_maps_);

    Mat result_img;
    StitchFrame(src, result_img);

    imwrite("result.jpg", result_img);

    return 0;
}