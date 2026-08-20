// #include "perception/nodes/LidarCameraFusion.h"
#include "perception/nodes/BEV.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto options = rclcpp::NodeOptions();
    rclcpp::spin(std::make_shared<BEV>("BevNode", options));
    rclcpp::shutdown();
    return 0;
}

// #include "perception/tools/semantic_segmentor.h"
// #include <chrono>
// #include <cuda_runtime_api.h>
// #include <filesystem>
// #include <future>
// #include <iostream>
// #include <stdexcept>
// #include <string>
// #include <vector>

// namespace
// {
// size_t usedGpuMemoryBytes()
// {
// 	size_t free_bytes = 0;
// 	size_t total_bytes = 0;
// 	if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
// 	{
// 		throw std::runtime_error("cudaMemGetInfo failed.");
// 	}
// 	return total_bytes - free_bytes;
// }
//
// double megabytes(const size_t bytes)
// {
// 	return static_cast<double>(bytes) / (1024.0 * 1024.0);
// }
// }
//
// int main(const int argc, char** argv)
// {
// 	(void)argc;
// 	(void)argv;
// 	const std::filesystem::path output_dir = "segmentation_results";
//
// 	try
// 	{
// 		std::vector<cv::Mat> images;
// 		images.reserve(4);
// 		images.push_back(cv::imread("/media/avent/DATA/generated_data/train/2026.08.12-17:38/rgb/0006.png", cv::IMREAD_COLOR));
// 		images.push_back(cv::imread("/media/avent/DATA/generated_data/train/2026.08.12-17:38/rgb/3030.png", cv::IMREAD_COLOR));
// 		images.push_back(cv::imread("/media/avent/DATA/generated_data/train/2026.08.12-17:38/rgb/3058.png", cv::IMREAD_COLOR));
// 		images.push_back(cv::imread("/media/avent/DATA/generated_data/train/2026.08.12-17:38/rgb/3191.png", cv::IMREAD_COLOR));
//
// 		for (size_t image_index = 0; image_index < images.size(); ++image_index)
// 		{
// 			if (images[image_index].empty())
// 			{
// 				throw std::runtime_error("Failed to read hardcoded test image at index " + std::to_string(image_index));
// 			}
// 		}
//
// 		std::filesystem::create_directories(output_dir);
//
// 		constexpr const char* model_path =
// 		    "/home/avent/Desktop/Image-Segmentation/SemanticSegmentation/Segformer/outputs/segformer_singlebatch.plan";
// 		std::vector<std::unique_ptr<SegFormerSegmentor>> segmentors;
// 		segmentors.reserve(images.size());
// 		for (size_t image_index = 0; image_index < images.size(); ++image_index)
// 		{
// 			segmentors.push_back(std::make_unique<SegFormerSegmentor>(model_path));
// 			if (segmentors.back()->batchSize() != 1)
// 			{
// 				throw std::runtime_error("Concurrent inference requires a batch-1 engine.");
// 			}
// 		}
// 		cudaDeviceSynchronize();
// 		const size_t memory_after_contexts = usedGpuMemoryBytes();
//
// 		std::vector<std::future<SemanticResult>> futures;
// 		futures.reserve(images.size());
// 		const auto inference_start = std::chrono::steady_clock::now();
// 		for (size_t image_index = 0; image_index < images.size(); ++image_index)
// 		{
// 			futures.push_back(std::async(std::launch::async,
// 			                             [&segmentor = *segmentors[image_index], &image = images[image_index]] {
// 				                             return segmentor.segment(image);
// 			                             }));
// 		}
//
// 		std::vector<SemanticResult> results;
// 		results.reserve(images.size());
// 		for (auto& future : futures)
// 		{
// 			results.push_back(future.get());
// 		}
// 		// cudaDeviceSynchronize();
// 		const auto inference_end = std::chrono::steady_clock::now();
// 		const size_t memory_after_inference = usedGpuMemoryBytes();
// 		const double inference_milliseconds =
// 		    std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
// 		std::cout << "Concurrent inference time for " << results.size() << " images: "
// 		          << inference_milliseconds << " ms (" << inference_milliseconds / results.size()
// 		          << " ms/image wall-clock average)\n";
// 		std::cout << "GPU memory after 4 contexts: " << megabytes(memory_after_contexts) << " MiB\n";
// 		std::cout << "GPU memory after inference: " << megabytes(memory_after_inference) << " MiB\n";
// 		const double memory_change = static_cast<double>(memory_after_inference) -
// 		                             static_cast<double>(memory_after_contexts);
// 		std::cout << "GPU memory change during inference: " << memory_change / (1024.0 * 1024.0)
// 		          << " MiB\n";
// 		std::cout << "Segmented " << results.size() << " images concurrently with independent batch-1 contexts.\n";
// 		const std::vector<cv::Vec3b> color_map = makePascalLikeColorMap(256);
//
// 		for (size_t image_index = 0; image_index < results.size(); ++image_index)
// 		{
// 			const cv::Mat colorized = colorizeClassMap(results[image_index].class_map, color_map);
// 			const cv::Mat overlay = overlaySegmentation(images[image_index], results[image_index].class_map, color_map);
//
//             if (const std::string suffix = std::to_string(image_index); !cv::imwrite((output_dir / ("class_map_" + suffix + ".png")).string(), colorized) ||
//                                                                         !cv::imwrite(
//                                                                             (output_dir / ("overlay_" + suffix + ".png")).string(),
//                                                                             overlay))
// 			{
// 				throw std::runtime_error("Failed to write visualization for image " + suffix);
// 			}
// 		}
//
// 		std::cout << "Results written to " << output_dir << '\n';
// 	}
// 	catch (const std::exception& error)
// 	{
// 		std::cerr << "Batch segmentation failed: " << error.what() << '\n';
// 		return 1;
// 	}
//
// 	return 0;
// }