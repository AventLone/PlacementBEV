#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

struct BevConfig
{
    float x_min = -5.0f;
    float x_max = 5.0f;
    float y_min = 0.0f;
    float y_max = 10.0f;
    float resolution = 0.05f; // meter / pixel
};

struct CameraModel
{
    Eigen::Matrix3f K;
    Eigen::Matrix3f Rcw; // world(or forklift) -> camera
    Eigen::Vector3f tcw;
    cv::Mat image;
};

// inline std::pair<cv::Mat, cv::Mat> bevFusion(const std::vector<CameraModel>& cams, const BevConfig& cfg)
// {
//     const int bev_w = std::ceil((cfg.x_max - cfg.x_min) / cfg.resolution);
//     const int bev_h = std::ceil((cfg.y_max - cfg.y_min) / cfg.resolution);

//     cv::Mat bev(bev_h, bev_w, CV_8UC3, cv::Scalar(0, 0, 0));
//     cv::Mat bev_binary = cv::Mat::zeros(bev_h, bev_w, CV_8UC1);

//     for (int r = 0; r < bev_h; r++)
//     {
//         for (int c = 0; c < bev_w; c++)
//         {
//             const double x = cfg.x_min + (c + 0.5) * cfg.resolution;
//             const double y = cfg.y_max - (r + 0.5) * cfg.resolution;
//             cv::Vec3d p_w(x, y, 0.0);

//             for (const auto& [K, Rcw, tcw, image] : cams)
//             {
//                 cv::Vec3d p_c = Rcw * p_w + tcw;

//                 if (p_c[2] <= 1e-3)
//                 {
//                     continue;
//                 }

//                 cv::Vec3d p = K * p_c;
//                 const int u = static_cast<int>(std::round(p[0] / p[2]));
//                 const int v = static_cast<int>(std::round(p[1] / p[2]));

//                 if (u < 0 || u >= image.cols || v < 0 || v >= image.rows)
//                 {
//                     continue;
//                 }

//                 if (const auto& img_pixel = image.ptr<cv::Vec3b>(v)[u]; img_pixel != cv::Vec3b(0, 0, 0))
//                 {
//                     bev.ptr<cv::Vec3b>(r)[c] = img_pixel;
//                     bev_binary.ptr<uchar>(r)[c] = 255;
//                 }
//             }
//         }
//     }

//     return {bev, bev_binary};
// }

inline cv::Mat bevFusionBina(const std::vector<CameraModel>& cams, const BevConfig& cfg)
{
    const int bev_w = std::ceil((cfg.x_max - cfg.x_min) / cfg.resolution);
    const int bev_h = std::ceil((cfg.y_max - cfg.y_min) / cfg.resolution);

    cv::Mat bev = cv::Mat::zeros(bev_h, bev_w, CV_8UC1);

    for (int r = 0; r < bev_h; r++)
    {
        for (int c = 0; c < bev_w; c++)
        {
            const float x = cfg.x_min + (c + 0.5) * cfg.resolution;
            const float y = cfg.y_max - (r + 0.5) * cfg.resolution;
            Eigen::Vector3f p_w(x, y, 0.0f);

            for (const auto& [K, Rcw, tcw, image] : cams)
            {
                Eigen::Vector3f p_c = Rcw * p_w + tcw;

                if (p_c[2] <= 1e-3)
                {
                    continue;
                }

                Eigen::Vector3f p = K * p_c;
                const int u = static_cast<int>(std::round(p[0] / p[2]));
                const int v = static_cast<int>(std::round(p[1] / p[2]));

                if (u < 0 || u >= image.cols || v < 0 || v >= image.rows)
                {
                    continue;
                }

                if (const auto& img_pixel = image.ptr<uchar>(v)[u]; img_pixel != 0)
                {
                    bev.ptr<uchar>(r)[c] = img_pixel;
                }
            }
        }
    }

    return bev;
}

//生成可放置区域（Minkowski erosion）
inline cv::Mat computeFeasibleRegion(const cv::Mat& free_space, const cv::Size box_size)
{
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, box_size);
    cv::Mat feasible;
    cv::erode(free_space, feasible, kernel); // freeSpace: 1 = free, 0 = obstacle
    return feasible;
}
