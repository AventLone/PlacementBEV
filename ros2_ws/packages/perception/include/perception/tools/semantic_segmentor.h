#pragma once
#include "perception/types/cuda_buffer.hpp"
#include <NvInfer.h>
#include <array>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

struct SemanticResult
{
    cv::Mat class_map;      // CV_32SC1, label id per pixel
    cv::Mat confidence_map; // CV_32FC1, max probability per pixel
};

class SegFormerSegmentor
{
    class Logger final : public nvinfer1::ILogger
    {
    public:
        void log(const Severity severity, const char* message) noexcept override
        {
            if (severity <= Severity::kWARNING)
            {
                std::cerr << "[TensorRT] " << message << std::endl;
            }
        }
    };

public:
    explicit SegFormerSegmentor(const std::string& model_path, std::string input_name = "input", std::string output_name = "logits");
    ~SegFormerSegmentor();

    SegFormerSegmentor(const SegFormerSegmentor&) = delete;
    SegFormerSegmentor& operator=(const SegFormerSegmentor&) = delete;
    SegFormerSegmentor(SegFormerSegmentor&&) = delete;
    SegFormerSegmentor& operator=(SegFormerSegmentor&&) = delete;

    SemanticResult segment(const cv::Mat& image, bool swap_rb = true,
                           const cv::Scalar& mean = {0.485, 0.456, 0.406}, const cv::Scalar& std_ = {0.229, 0.224, 0.225});

    std::vector<SemanticResult> segmentBatch(const std::vector<cv::Mat>& images, bool swap_rb = true,
                                             const cv::Scalar& mean = {0.485, 0.456, 0.406},
                                             const cv::Scalar& std_ = {0.229, 0.224, 0.225});

    [[nodiscard]] int batchSize() const noexcept { return mBatchSize; }
    [[nodiscard]] nvinfer1::DataType inputType() const noexcept { return mInputType; }
    [[nodiscard]] nvinfer1::DataType outputType() const noexcept { return mOutputType; }

    // Backward-compatible API name to simplify transition from the previous wrapper.
    // SemanticResult seg(const cv::Mat& image,
    //                    const bool swap_rb = true,
    //                    const cv::Scalar& mean = {0.485, 0.456, 0.406},
    //                    const cv::Scalar& std_ = {0.229, 0.224, 0.225})
    // {
    //     return segment(image, swap_rb, mean, std_);
    // }

private:
    struct TensorShapeInfo
    {
        int channels{0};
        int height{0};
        int width{0};
        bool is_label_map{false};
    };

    inline static Logger sLogger{};
    void* mCudaStream{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mCudaEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::string mInputName;
    std::string mOutputName;

    cv::Size mInputSize;
    int mBatchSize{0};
    nvinfer1::DataType mInputType{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mOutputType{nvinfer1::DataType::kFLOAT};
    TensorShapeInfo mOutputShape;

    std::unordered_map<std::string, CudaBuffer> mCudaBuffers;
    std::vector<float> mOutputTensor;
    std::vector<uint16_t> mOutputTensorHalf;

    static std::vector<char> readModel(const std::string& file_path);

    static size_t getTensorSize(const nvinfer1::Dims& dims);
    static cv::Size extractHW(const nvinfer1::Dims& dims);
    static TensorShapeInfo parseOutputShape(const nvinfer1::Dims& dims);

    // bool infer(const cv::Mat& input_image);

    static std::vector<float> preprocess(const cv::Mat& image,
                                         const cv::Size& input_size,
                                         const cv::Scalar& mean,
                                         const cv::Scalar& std_,
                                         bool swap_rb);

    SemanticResult postprocess(const cv::Size& original_size, int batch_index) const;
};

inline std::vector<cv::Vec3b> makePascalLikeColorMap(const int num_classes)
{
    std::vector<cv::Vec3b> color_map;
    color_map.reserve(static_cast<size_t>(std::max(num_classes, 0)));

    for (int class_id = 0; class_id < num_classes; ++class_id)
    {
        int label = class_id;
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;

        for (int bit = 0; bit < 8; ++bit)
        {
            r |= static_cast<uint8_t>(((label >> 0) & 1) << (7 - bit));
            g |= static_cast<uint8_t>(((label >> 1) & 1) << (7 - bit));
            b |= static_cast<uint8_t>(((label >> 2) & 1) << (7 - bit));
            label >>= 3;
        }

        color_map.emplace_back(b, g, r);
    }

    return color_map;
}

inline cv::Mat colorizeClassMap(const cv::Mat& class_map, const std::vector<cv::Vec3b>& color_map)
{
    CV_Assert(class_map.type() == CV_32SC1);
    cv::Mat colorized(class_map.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    for (int y = 0; y < class_map.rows; ++y)
    {
        const int* class_ptr = class_map.ptr<int>(y);
        auto* out_ptr = colorized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < class_map.cols; ++x)
        {
            if (const int class_id = class_ptr[x]; class_id >= 0 && static_cast<size_t>(class_id) < color_map.size())
            {
                out_ptr[x] = color_map[static_cast<size_t>(class_id)];
            }
        }
    }

    return colorized;
}

inline cv::Mat overlaySegmentation(const cv::Mat& image,
                                   const cv::Mat& class_map,
                                   const std::vector<cv::Vec3b>& color_map,
                                   const double alpha = 0.45)
{
    CV_Assert(image.type() == CV_8UC3);
    CV_Assert(class_map.type() == CV_32SC1);
    CV_Assert(image.size() == class_map.size());

    const cv::Mat seg_vis = colorizeClassMap(class_map, color_map);
    cv::Mat blended;
    cv::addWeighted(image, 1.0 - alpha, seg_vis, alpha, 0.0, blended);
    return blended;
}

using SemanticSeg = SegFormerSegmentor;
