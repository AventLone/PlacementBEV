#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class DynamicVoxelGrid
{
public:
	struct ResolutionBand
	{
		// Radial distance bound in XY plane, meters.
		// Example: [0, 10] uses near-field high resolution.
		float max_radius_xy{0.0F};
		float voxel_size_xy{0.2F};
		float voxel_size_z{0.2F};
	};

	struct Config
	{
		std::vector<ResolutionBand> bands;
		float min_z{-3.0F};
		float max_z{3.0F};
		float center_x{0.0F};
		float center_y{0.0F};
	};

	struct VoxelIndex
	{
		int level{0};
		int ix{0};
		int iy{0};
		int iz{0};

		bool operator==(const VoxelIndex& other) const
		{
			return level == other.level && ix == other.ix && iy == other.iy && iz == other.iz;
		}
	};

	struct VoxelStats
	{
		int count{0};
		float sum_x{0.0F};
		float sum_y{0.0F};
		float sum_z{0.0F};
		float sum_intensity{0.0F};
		float min_z{std::numeric_limits<float>::max()};
		float max_z{-std::numeric_limits<float>::max()};
	};

	struct OccupiedVoxel
	{
		VoxelIndex index;
		VoxelStats stats;
		float center_x{0.0F};
		float center_y{0.0F};
		float center_z{0.0F};
		float size_xy{0.0F};
		float size_z{0.0F};
	};

	explicit DynamicVoxelGrid(Config config)
		: mConfig(std::move(config))
	{
		validateConfig();
	}

	void clear()
	{
		mSparseGrid.clear();
	}

	[[nodiscard]] size_t size() const
	{
		return mSparseGrid.size();
	}

	[[nodiscard]] const Config& config() const
	{
		return mConfig;
	}

	bool addPoint(const pcl::PointXYZI& point)
	{
		const std::optional<VoxelIndex> voxel_index = computeVoxelIndex(point.x, point.y, point.z);
		if (!voxel_index.has_value())
		{
			return false;
		}

		auto& [count, sum_x, sum_y, sum_z, sum_intensity, min_z, max_z] = mSparseGrid[*voxel_index];
		count += 1;
		sum_x += point.x;
		sum_y += point.y;
		sum_z += point.z;
		sum_intensity += point.intensity;
		min_z = std::min(min_z, point.z);
		max_z = std::max(max_z, point.z);
		return true;
	}

	void addPointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud)
	{
		for (const auto& point : cloud)
		{
			(void)addPoint(point);
		}
	}

	[[nodiscard]] std::vector<OccupiedVoxel> exportOccupiedVoxels() const
	{
		std::vector<OccupiedVoxel> output;
		output.reserve(mSparseGrid.size());

		for (const auto& [index, stats] : mSparseGrid)
		{
			const ResolutionBand& band = mConfig.bands[static_cast<size_t>(index.level)];

			OccupiedVoxel voxel;
			voxel.index = index;
			voxel.stats = stats;
			voxel.size_xy = band.voxel_size_xy;
			voxel.size_z = band.voxel_size_z;

			const float local_x = (static_cast<float>(index.ix) + 0.5F) * band.voxel_size_xy;
			const float local_y = (static_cast<float>(index.iy) + 0.5F) * band.voxel_size_xy;
			voxel.center_x = local_x + mConfig.center_x;
			voxel.center_y = local_y + mConfig.center_y;
			voxel.center_z = mConfig.min_z + (static_cast<float>(index.iz) + 0.5F) * band.voxel_size_z;
			output.push_back(voxel);
		}

		return output;
	}

private:
	struct VoxelIndexHash
	{
		size_t operator()(const VoxelIndex& index) const
		{
			const size_t h1 = std::hash<int>{}(index.level);
			const size_t h2 = std::hash<int>{}(index.ix);
			const size_t h3 = std::hash<int>{}(index.iy);
			const size_t h4 = std::hash<int>{}(index.iz);

			size_t hash = h1;
			hash ^= h2 + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
			hash ^= h3 + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
			hash ^= h4 + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
			return hash;
		}
	};

	void validateConfig() const
	{
		if (mConfig.bands.empty())
		{
			throw std::invalid_argument("DynamicVoxelGrid config must contain at least one resolution band.");
		}

		if (mConfig.max_z <= mConfig.min_z)
		{
			throw std::invalid_argument("DynamicVoxelGrid config is invalid: max_z must be greater than min_z.");
		}

		float previous_radius = 0.0F;
		for (const auto& band : mConfig.bands)
		{
			if (band.max_radius_xy <= previous_radius)
			{
				throw std::invalid_argument("Resolution bands must have strictly increasing max_radius_xy.");
			}
			if (band.voxel_size_xy <= 0.0F || band.voxel_size_z <= 0.0F)
			{
				throw std::invalid_argument("Resolution bands must have positive voxel_size_xy and voxel_size_z.");
			}
			previous_radius = band.max_radius_xy;
		}
	}

	[[nodiscard]] std::optional<int> selectBand(const float x, const float y) const
	{
		const float dx = x - mConfig.center_x;
		const float dy = y - mConfig.center_y;
		const float radius = std::sqrt(dx * dx + dy * dy);

		for (size_t i = 0; i < mConfig.bands.size(); ++i)
		{
			if (radius <= mConfig.bands[i].max_radius_xy)
			{
				return static_cast<int>(i);
			}
		}

		return std::nullopt;
	}

	[[nodiscard]] std::optional<VoxelIndex> computeVoxelIndex(const float x, const float y, const float z) const
	{
		if (z < mConfig.min_z || z >= mConfig.max_z)
		{
			return std::nullopt;
		}

		const std::optional<int> level = selectBand(x, y);
		if (!level.has_value())
		{
			return std::nullopt;
		}

		const ResolutionBand& band = mConfig.bands[static_cast<size_t>(*level)];
		const float local_x = x - mConfig.center_x;
		const float local_y = y - mConfig.center_y;

		const int ix = static_cast<int>(std::floor(local_x / band.voxel_size_xy));
		const int iy = static_cast<int>(std::floor(local_y / band.voxel_size_xy));
		const int iz = static_cast<int>(std::floor((z - mConfig.min_z) / band.voxel_size_z));

		return VoxelIndex{*level, ix, iy, iz};
	}

	Config mConfig;
	std::unordered_map<VoxelIndex, VoxelStats, VoxelIndexHash> mSparseGrid;
};