#include "perception/tools/semantic_segmentor.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <limits>

std::vector<char> SegFormerSegmentor::readModel(const std::string& file_path)
{
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg); // 将读指针从文件末尾开始移动0个字节
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Failed to read file: " + file_path);
    }
    return buffer;
}

size_t SegFormerSegmentor::getTensorSize(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= static_cast<size_t>(dims.d[i] > 0 ? dims.d[i] : 1);
    }
    return size;
}

cv::Size SegFormerSegmentor::extractHW(const nvinfer1::Dims& dims)
{
    if (dims.nbDims < 2)
    {
        throw std::runtime_error("Input tensor rank is too small. Expected at least 2 dims.");
    }

    const int h = dims.d[dims.nbDims - 2];
    const int w = dims.d[dims.nbDims - 1];
    if (h <= 0 || w <= 0)
    {
        throw std::runtime_error("Input tensor has invalid H/W dimensions.");
    }

    return {w, h};
}

SegFormerSegmentor::TensorShapeInfo SegFormerSegmentor::parseOutputShape(const nvinfer1::Dims& dims)
{
    if (dims.nbDims == 4)
    {
        TensorShapeInfo info;
        info.channels = static_cast<int>(dims.d[1]);
        info.height = static_cast<int>(dims.d[2]);
        info.width = static_cast<int>(dims.d[3]);
        info.is_label_map = false;
        return info;
    }

    if (dims.nbDims == 3)
    {
        // Supports [1,H,W] label maps.
        TensorShapeInfo info;
        info.channels = 1;
        info.height = static_cast<int>(dims.d[1]);
        info.width = static_cast<int>(dims.d[2]);
        info.is_label_map = true;
        return info;
    }

    throw std::runtime_error("Unsupported output tensor shape. Expected [N,C,H,W] logits or [N,H,W] labels.");
}

SegFormerSegmentor::SegFormerSegmentor(const std::string& model_path,
                                       std::string input_name,
                                       std::string output_name)
    : mInputName(std::move(input_name)),
      mOutputName(std::move(output_name))
{
    cudaSetDeviceFlags(cudaDeviceScheduleYield);

    const std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(mLogger)};
    if (!runtime)
    {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    const std::vector<char> model_data = readModel(model_path);
    mCudaEngine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_data.data(), model_data.size()));
    if (!mCudaEngine)
    {
        throw std::runtime_error("Failed to deserialize TensorRT engine plan.");
    }

    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
    if (!mContext)
    {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }

    for (int i = 0; i < mCudaEngine->getNbIOTensors(); ++i)
    {
        const char* name = mCudaEngine->getIOTensorName(i);
        const nvinfer1::Dims dims = mCudaEngine->getTensorShape(name);
        mCudaBuffers[name] = CudaBuffer(mCudaEngine->getTensorDataType(name), getTensorSize(dims));
        mContext->setTensorAddress(name, mCudaBuffers[name].data());
    }

    if (!mCudaBuffers.contains(mInputName))
    {
        throw std::runtime_error("Input tensor not found in engine: " + mInputName);
    }
    if (!mCudaBuffers.contains(mOutputName))
    {
        throw std::runtime_error("Output tensor not found in engine: " + mOutputName);
    }

    mInputSize = extractHW(mCudaEngine->getTensorShape(mInputName.c_str()));
    mOutputShape = parseOutputShape(mCudaEngine->getTensorShape(mOutputName.c_str()));

    if (mOutputShape.height <= 0 || mOutputShape.width <= 0)
    {
        throw std::runtime_error("Output tensor shape contains invalid spatial dimensions.");
    }

    mOutputTensor.resize(mCudaBuffers[mOutputName].size());

    if (auto* stream = reinterpret_cast<cudaStream_t*>(&mCudaStream); cudaStreamCreate(stream) != cudaSuccess)
    {
        throw std::runtime_error("Failed to create CUDA stream.");
    }
}

SegFormerSegmentor::~SegFormerSegmentor()
{
    if (mCudaStream != nullptr)
    {
        cudaStreamDestroy(static_cast<cudaStream_t>(mCudaStream));
        mCudaStream = nullptr;
    }
}

// bool SegFormerSegmentor::infer(const cv::Mat& input_image)
// {
//     const std::vector<float> input_tensor = preprocess(input_image, mInputSize, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}, true);
//     const auto stream = static_cast<cudaStream_t>(mCudaStream);
//     cudaMemcpyAsync(mCudaBuffers[mInputName].data(), input_tensor.data(),
//                     mCudaBuffers[mInputName].bytes(), cudaMemcpyHostToDevice, stream);
//
//     if (!mContext->enqueueV3(stream))
//     {
//         return false;
//     }
//
//     cudaMemcpyAsync(mOutputTensor.data(), mCudaBuffers[mOutputName].data(),
//                     mCudaBuffers[mOutputName].bytes(), cudaMemcpyDeviceToHost, stream);
//     return cudaStreamSynchronize(stream) == cudaSuccess;
// }

std::vector<float> SegFormerSegmentor::preprocess(const cv::Mat& image, const cv::Size& input_size,
                                                  const cv::Scalar& mean, const cv::Scalar& std_, const bool swap_rb)
{
    cv::Mat resized;
    cv::resize(image, resized, input_size, 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat fp32;
    resized.convertTo(fp32, CV_32FC3, 1.0 / 255.0);
    if (swap_rb)
    {
        cv::cvtColor(fp32, fp32, cv::COLOR_BGR2RGB);
    }

    std::vector<cv::Mat> channels(3);
    cv::split(fp32, channels);

    for (int c = 0; c < 3; ++c)
    {
        channels[c] = (channels[c] - mean[c]) / std_[c];
    }

    const auto plane = static_cast<size_t>(input_size.area());
    std::vector<float> nchw(plane * 3);

    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(nchw.data() + c * plane, channels[c].ptr<float>(), plane * sizeof(float));
    }

    return nchw;
}

SemanticResult SegFormerSegmentor::postprocess(const cv::Size& original_size) const
{
    SemanticResult result;

    if (mOutputShape.is_label_map)
    {
        cv::Mat labels_small(mOutputShape.height, mOutputShape.width, CV_32SC1);
        for (int y = 0; y < mOutputShape.height; ++y)
        {
            int* out_ptr = labels_small.ptr<int>(y);
            const float* in_ptr = mOutputTensor.data() + static_cast<size_t>(y) * static_cast<size_t>(mOutputShape.width);
            for (int x = 0; x < mOutputShape.width; ++x)
            {
                out_ptr[x] = static_cast<int>(in_ptr[x]);
            }
        }

        cv::resize(labels_small, result.class_map, original_size, 0.0, 0.0, cv::INTER_NEAREST);
        result.confidence_map = cv::Mat::ones(original_size, CV_32FC1);
        return result;
    }

    // PyTorch-style order for SegFormer: upsample logits first, then argmax.
    result.class_map = cv::Mat(original_size, CV_32SC1, cv::Scalar(0));
    cv::Mat best_logit(original_size, CV_32FC1, cv::Scalar(-std::numeric_limits<double>::infinity()));

    const size_t plane = static_cast<size_t>(mOutputShape.height) * static_cast<size_t>(mOutputShape.width);

    for (int c = 0; c < mOutputShape.channels; ++c)
    {
        cv::Mat logits_small(mOutputShape.height, mOutputShape.width, CV_32FC1,
                             const_cast<float*>(mOutputTensor.data() + static_cast<size_t>(c) * plane));
        cv::Mat logits_up;
        cv::resize(logits_small, logits_up, original_size, 0.0, 0.0, cv::INTER_LINEAR);

        for (int y = 0; y < original_size.height; ++y)
        {
            const float* up_ptr = logits_up.ptr<float>(y);
            auto* best_ptr = best_logit.ptr<float>(y);
            int* cls_ptr = result.class_map.ptr<int>(y);
            for (int x = 0; x < original_size.width; ++x)
            {
                if (up_ptr[x] > best_ptr[x])
                {
                    best_ptr[x] = up_ptr[x];
                    cls_ptr[x] = c;
                }
            }
        }
    }

    cv::Mat denom(original_size, CV_32FC1, cv::Scalar(0.0F));
    cv::Mat best_exp(original_size, CV_32FC1, cv::Scalar(0.0F));

    for (int c = 0; c < mOutputShape.channels; ++c)
    {
        cv::Mat logits_small(mOutputShape.height, mOutputShape.width, CV_32FC1,
                             const_cast<float*>(mOutputTensor.data() + static_cast<size_t>(c) * plane));
        cv::Mat logits_up;
        cv::resize(logits_small, logits_up, original_size, 0.0, 0.0, cv::INTER_LINEAR);

        for (int y = 0; y < original_size.height; ++y)
        {
            const float* up_ptr = logits_up.ptr<float>(y);
            const float* best_ptr = best_logit.ptr<float>(y);
            auto* denom_ptr = denom.ptr<float>(y);
            auto* best_exp_ptr = best_exp.ptr<float>(y);
            const int* cls_ptr = result.class_map.ptr<int>(y);

            for (int x = 0; x < original_size.width; ++x)
            {
                const float e = std::exp(up_ptr[x] - best_ptr[x]);
                denom_ptr[x] += e;
                if (cls_ptr[x] == c)
                {
                    best_exp_ptr[x] = e;
                }
            }
        }
    }

    result.confidence_map = cv::Mat(original_size, CV_32FC1, cv::Scalar(0.0F));
    for (int y = 0; y < original_size.height; ++y)
    {
        const float* denom_ptr = denom.ptr<float>(y);
        const float* best_exp_ptr = best_exp.ptr<float>(y);
        auto* conf_ptr = result.confidence_map.ptr<float>(y);
        for (int x = 0; x < original_size.width; ++x)
        {
            conf_ptr[x] = denom_ptr[x] > 0.0F ? best_exp_ptr[x] / denom_ptr[x] : 0.0F;
        }
    }

    return result;
}

SemanticResult SegFormerSegmentor::segment(const cv::Mat& image,
                                           const bool swap_rb,
                                           const cv::Scalar& mean,
                                           const cv::Scalar& std_)
{
    if (image.empty())
    {
        throw std::invalid_argument("Input image is empty.");
    }

    const std::vector<float> input_tensor = preprocess(image, mInputSize, mean, std_, swap_rb);
    const auto stream = static_cast<cudaStream_t>(mCudaStream);
    cudaMemcpyAsync(mCudaBuffers[mInputName].data(), input_tensor.data(),
                    mCudaBuffers[mInputName].bytes(), cudaMemcpyHostToDevice, stream);

    if (!mContext->enqueueV3(stream))
    {
        throw std::runtime_error("TensorRT enqueue failed.");
    }

    cudaMemcpyAsync(mOutputTensor.data(), mCudaBuffers[mOutputName].data(),
                    mCudaBuffers[mOutputName].bytes(), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess)
    {
        throw std::runtime_error("CUDA stream synchronization failed.");
    }

    return postprocess(image.size());
}
