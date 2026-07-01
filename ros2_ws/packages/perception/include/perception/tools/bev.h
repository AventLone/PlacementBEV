#pragma once
#include <opencv2/opencv.hpp>

struct BevConfig
{
    double x_min = -5.0;
    double x_max = 5.0;
    double y_min = 0.0;
    double y_max = 10.0;
    double resolution = 0.05; // meter / pixel
};

struct BevResult
{
    cv::Mat bev_image; // CV_8UC3
    cv::Mat bev_ground_mask; // CV_8UC1
    cv::Mat valid_mask; // CV_8UC1
    cv::Mat map_x; // CV_32FC1
    cv::Mat map_y; // CV_32FC1
};

static cv::Matx33d matToMatx33d(const cv::Mat& mat)
{
    CV_Assert(mat.rows == 3 && mat.cols == 3 && mat.type() == CV_64F);

    cv::Matx33d out;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            out(r, c) = mat.at<double>(r, c);
        }
    }
    return out;
}

static cv::Vec3d matToVec3d(const cv::Mat& mat)
{
    CV_Assert(mat.total() == 3 && mat.type() == CV_64F);

    cv::Mat t = mat.reshape(1, 3);
    return {t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)};
}

inline BevResult ipmInverseMapping(const cv::Mat& image, const cv::Mat& ground_mask, const cv::Matx33d& K,
                                   const cv::Matx33d& Rcw, const cv::Vec3d& tcw, const BevConfig& cfg)
{
    CV_Assert(image.type() == CV_8UC3);
    CV_Assert(ground_mask.type() == CV_8UC1);
    CV_Assert(image.rows == ground_mask.rows);
    CV_Assert(image.cols == ground_mask.cols);

    const int bev_w = static_cast<int>((cfg.x_max - cfg.x_min) / cfg.resolution);
    const int bev_h = static_cast<int>((cfg.y_max - cfg.y_min) / cfg.resolution);

    cv::Mat map_x(bev_h, bev_w, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat map_y(bev_h, bev_w, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat geometry_valid(bev_h, bev_w, CV_8UC1, cv::Scalar(0));

    for (int row = 0; row < bev_h; ++row)
    {
        for (int col = 0; col < bev_w; ++col)
        {
            double x = cfg.x_min + (col + 0.5) * cfg.resolution;
            double y = cfg.y_max - (row + 0.5) * cfg.resolution;

            cv::Vec3d p_w(x, y, 0.0);
            cv::Vec3d p_c = Rcw * p_w + tcw; // world -> camera

            if (p_c[2] <= 1e-6)
            {
                continue;
            }

            cv::Vec3d p = K * p_c; // camera -> image

            double u = p[0] / p[2];
            double v = p[1] / p[2];

            if (u < 0.0 || u >= image.cols || v < 0.0 || v >= image.rows)
            {
                continue;
            }

            map_x.at<float>(row, col) = static_cast<float>(u);
            map_y.at<float>(row, col) = static_cast<float>(v);
            geometry_valid.at<uchar>(row, col) = 255;
        }
    }

    cv::Mat bev_image;
    cv::Mat bev_ground_mask;

    cv::remap(image, bev_image, map_x, map_y, cv::INTER_LINEAR);
    cv::remap(ground_mask, bev_ground_mask, map_x, map_y, cv::INTER_NEAREST);

    // 只保留：几何投影有效 && 原图中属于 ground mask 的区域
    cv::Mat valid_mask;
    cv::bitwise_and(geometry_valid, bev_ground_mask, valid_mask);

    cv::Mat invalid_mask;
    cv::bitwise_not(valid_mask, invalid_mask);

    bev_image.setTo(cv::Scalar(0, 0, 0), invalid_mask);
    bev_ground_mask.setTo(0, invalid_mask);

    return {bev_image, bev_ground_mask, valid_mask, map_x, map_y};
}

inline cv::Mat getBev(const cv::Mat& img, const cv::Matx33d& K,
                      const cv::Matx33d& Rcw, const cv::Vec3d& tcw, const BevConfig& cfg)
{
    /* 1. 计算BEV图像尺寸 */
    const int bev_w = static_cast<int>((cfg.x_max - cfg.x_min) / cfg.resolution);
    const int bev_h = static_cast<int>((cfg.y_max - cfg.y_min) / cfg.resolution);

    /* 2. 计算单应性矩阵 H (BEV物理坐标 -> 图像像素坐标) */
    const cv::Matx33d H_ground_to_img(K * cv::Matx33d(
                                          Rcw(0, 0), Rcw(0, 1), tcw[0],
                                          Rcw(1, 0), Rcw(1, 1), tcw[1],
                                          Rcw(2, 0), Rcw(2, 1), tcw[2]
                                      ));
    const cv::Matx33d H_bev_to_ground(0.0, -cfg.resolution, cfg.x_max - 0.5 * cfg.resolution,
                                      cfg.resolution, 0.0, cfg.y_min + 0.5 * cfg.resolution,
                                      0.0, 0.0, 1.0);
    const cv::Matx33d H_bev_to_img = H_ground_to_img * H_bev_to_ground;

    /* 3. 执行反向映射 */
    cv::Mat bev;
    cv::warpPerspective(img, bev, H_bev_to_img, cv::Size(bev_w, bev_h), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
    return bev;
}

// def get_bev_image(img, K, R, T, bev_params):
//     """
//     img: 去畸变后的原图
//     K: (3, 3) 内参矩阵
//     R: (3, 3) 旋转矩阵 (World -> Camera)
//     T: (3, 1) 平移向量 (World -> Camera)
//     bev_params: BEV范围参数
//     """
//
//     # 1. 定义BEV参数
//     x_min, x_max = bev_params['x_range']  # 左右范围 (米)
//     y_min, y_max = bev_params['y_range']  # 前后范围 (米)
//     resolution = bev_params['resolution'] # 米/像素
//
//     # 计算BEV图像尺寸
//     width = int((x_max - x_min) / resolution)
//     height = int((y_max - y_min) / resolution)
//
//     # 2. 计算单应性矩阵 H (BEV物理坐标 -> 图像像素坐标)
//     # 假设 Z=0，我们取 R 的第1列、第2列和 T 组成 3x3 矩阵
//     # 注意：这取决于你的世界坐标系定义。
//     # 通常设定：X为右，Y为前，Z为上。此时地面是Z=0。
//     # 如果你的定义是：X为前，Y为左，Z为上，则需要取 R 的对应列。
//     # 这里假设标准惯例：Target World Z=0.
//
//     # 构建投影矩阵：仅保留 R 的第1、2列 (对应X, Y) 和 T
//     # target_matrix = K * [r1, r2, t]
//     target_matrix = np.zeros((3, 3))
//     target_matrix[:, 0] = R[:, 0] # r1
//     target_matrix[:, 1] = R[:, 1] # r2
//     target_matrix[:, 2] = T[:, 0] # t
//     H = np.dot(K, target_matrix)
//
//     # 3. 这里的 H 是把 (X_w, Y_w, 1) 映射到 (u, v, 1)
//     # 但我们需要把 BEV 像素坐标 (u_bev, v_bev) 映射到 物理坐标 (X_w, Y_w)
//     # 再映射到 原图 (u, v)。
//     # 我们可以构建一个从 BEV_Pixel -> World 的变换矩阵 M_scale
//
//     # BEV像素 (0,0) -> (x_min, y_max) (通常BEV图左上角对应物理空间的最远最左)
//     # BEV像素 (u, v) -> World (x, y) = (x_min + u*res, y_max - v*res)
//     # 这是一个简单的缩放和平移矩阵
//     M_bev2world = np.array([
//         [resolution, 0, x_min],
//         [0, -resolution, y_max], # 注意Y轴方向，图像向下，物理坐标通常向前
//         [0, 0, 1]
//     ])
//
//     # 综合矩阵：BEV像素 -> 原图像素
//     # H_total = H_world2img * M_bev2pixel
//     H_total = np.dot(H, M_bev2world)
//
//     # 4. 执行反向映射
//     # 使用 cv2.warpPerspective 并指定 WARP_INVERSE_MAP
//     # 或者直接传入 H_total 的逆矩阵 (因为warpPerspective默认是 Src->Dst)
//     # 但这里我们定义的 H_total 就是 Dst(BEV) -> Src(Img)，所以要用逆变换标志
//
//     bev_img = cv2.warpPerspective(
//         img,
//         H_total,
//         (width, height),
//         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
//     )
//
//     return bev_img
