#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "helper_math.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define BANDWIDTH 2
#define NUM_DIMENSIONS 3
#define NUM_ITERATIONS 10
#define TILE_WIDTH 64
#define BLOCK_DIM TILE_WIDTH


__global__ void MeanShiftWithoutTiling(float* shiftedPoints, const float* __restrict__ originalPoints, const unsigned numPoints) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 newPosition = make_float3(0.0, 0.0, 0.0);
	float totalWeight = 0.0;

	if (idx < numPoints) {
		float x = shiftedPoints[idx];
		float y = shiftedPoints[idx + numPoints];
		float z = shiftedPoints[idx + 2 * numPoints];
		float3 shiftedPoint = make_float3(x, y, z);

		for (int i = 0; i < numPoints; i++) {
			x = originalPoints[i];
			y = originalPoints[i + numPoints];
			z = originalPoints[i + 2 * numPoints];
			float3 originalPoint = make_float3(x, y, z);
			float3 difference = shiftedPoint - originalPoint;
			float squaredDistance = dot(difference, difference);
			float weight = std::exp((-squaredDistance) / (2 * powf(BANDWIDTH, 2)));
			newPosition += originalPoint * weight;
			totalWeight += weight;
		}
		newPosition /= totalWeight;
		shiftedPoints[idx] = newPosition.x;
		shiftedPoints[idx + numPoints] = newPosition.y;
		shiftedPoints[idx + 2 * numPoints] = newPosition.z;
	}
}

__global__ void MeanShiftTiling(float* shiftedPoints, const float* __restrict__ originalPoints, const unsigned numPoints) {

	__shared__ float tile[TILE_WIDTH][3];

	int tx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tx;

	float3 newPosition = make_float3(0.0, 0.0, 0.0);
	float totalWeight = 0.0;
	
	// loading phase - each thread load something into shared memory
	for (int tile_i = 0; tile_i < (numPoints - 1) / TILE_WIDTH + 1; ++tile_i) {

		int tile_idx = tile_i * TILE_WIDTH + tx;

		if (tile_idx < numPoints) {
			tile[tx][0] = originalPoints[tile_idx];
			tile[tx][1] = originalPoints[tile_idx + numPoints];
			tile[tx][2] = originalPoints[tile_idx + 2 * numPoints];
		}
		else {
			tile[tx][0] = 0.0;
			tile[tx][1] = 0.0;
			tile[tx][2] = 0.0;
		}

		__syncthreads();
	}
	//end of loading into shared memory
	
	//computing phase
	// only the threads inside the bounds do some computation
	if (idx < numPoints) {
		float x = shiftedPoints[idx];
		float y = shiftedPoints[idx + numPoints];
		float z = shiftedPoints[idx + 2 * numPoints];
		float3 shiftedPoint = make_float3(x, y, z);

		for (int i = 0; i < TILE_WIDTH; i++) {
			if (tile[i][0] != 0.0 && tile[i][1] != 0.0 && tile[i][2] != 0.0) {
				float3 originalPoint = make_float3(tile[i][0], tile[i][1], tile[i][2]);
				float3 difference = shiftedPoint - originalPoint;
				float squaredDistance = dot(difference, difference);
				if (sqrt(squaredDistance) <= BANDWIDTH) {
					float weight = std::exp((-squaredDistance) / (2 * powf(BANDWIDTH, 2)));
					newPosition += originalPoint * weight;
					totalWeight += weight;
				}
			}
		}
	}
	__syncthreads();
	
	if (idx < numPoints) {
		newPosition /= totalWeight;
		shiftedPoints[idx] = newPosition.x;
		shiftedPoints[idx + numPoints] = newPosition.y;
		shiftedPoints[idx + 2 * numPoints] = newPosition.z;
	}

}

int main(void)
{
	/* inserico i punti dal csv su vettore points - inzio */
 	float time;
	//std::string fileName = "dataset/3D_data_100.csv";
	//std::string fileName = "dataset/3D_data_1000.csv";
	//std::string fileName = "dataset/3D_data_10000.csv";
	//std::string fileName = "dataset/3D_data_20000.csv";
	//std::string fileName = "dataset/3D_data_100000.csv";
	//std::string fileName = "dataset/3D_data_250000.csv";
	//std::string fileName = "dataset/3D_data_500000.csv";
	//std::string fileName = "dataset/3D_data_1000000.csv";
	std::string fileName = "dataset/3D_data_2000000.csv";
	
	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;
	std::vector<float> h_inputPoints;
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	std::ifstream data(fileName);
	std::string line;
	while (std::getline(data, line)) {
		std::vector<float> point;
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) {
			point.push_back(stod(cell));
		}
		x.push_back(point[0]);
		y.push_back(point[1]);
		z.push_back(point[2]);
	}
	h_inputPoints = x;
	h_inputPoints.insert(h_inputPoints.end(), y.begin(), y.end());
	h_inputPoints.insert(h_inputPoints.end(), z.begin(), z.end());
	/* inserico i punti dal csv su vettore points - fine */

	int numPoints = h_inputPoints.size() / NUM_DIMENSIONS;
	printf("Numero di punti %d\n", numPoints);


	dim3 gridDim = dim3(ceil((float)numPoints / BLOCK_DIM));
	dim3 blockDim = dim3(BLOCK_DIM);

	/* TEST CUDA CON TILING - inizio */
	 

	/* Copy host_vector to device_vector */
	thrust::device_vector<float> d_originalPoints2 = h_inputPoints;
	thrust::device_vector<float> d_shiftedPoints2 = h_inputPoints;

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		MeanShiftTiling<<< gridDim, blockDim >>> (thrust::raw_pointer_cast(&d_shiftedPoints2[0]), thrust::raw_pointer_cast(&d_originalPoints2[0]), numPoints);
		cudaDeviceSynchronize();
	}

	end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
	 

	printf("\nTempo esecuzione Mean Shift con tiling: %f\n", time);
	/* TEST CUDA CON TILING - fine */
	/*d_originalPoints.clear();
	d_originalPoints.shrink_to_fit();
	d_shiftedPoints.clear();
	d_shiftedPoints.shrink_to_fit();*/
	
	/* TEST CUDA SENZA TILING - inizio*/
	 
	
	/* Copy host_vector to device_vector */
	thrust::device_vector<float> d_originalPoints = h_inputPoints;
	thrust::device_vector<float> d_shiftedPoints = h_inputPoints;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		MeanShiftWithoutTiling <<<gridDim, blockDim>>> (thrust::raw_pointer_cast(&d_shiftedPoints[0]), thrust::raw_pointer_cast(&d_originalPoints[0]), numPoints);
		cudaDeviceSynchronize();
	}

	end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
	

	printf("\nTempo esecuzione Mean Shift senza tiling: %f\n", time);
	/* TEST CUDA SENZA TILING  - fine */


	/* CLUSTERING - inizio */
	/* Copy device_vector to host_vector */
	thrust::host_vector<float> h_ShiftedPoints = d_shiftedPoints;

	std::vector<float> clusterPoints;
	clusterPoints.resize(numPoints);

	start = std::chrono::high_resolution_clock::now();
	std::vector<float3> clusters;
	float clusterEps = 5;
	for (int i = 0; i < numPoints; i++) {
		float x = h_ShiftedPoints[i];
		float y = h_ShiftedPoints[i + numPoints];
		float z = h_ShiftedPoints[i + 2 * numPoints];
		float3 point = make_float3(x, y, z);
		auto iter = clusters.begin();
		auto iterEnd = clusters.end();
		while (iter != iterEnd) {
			float3 difference = point - *iter;
			float distance = sqrt(dot(difference, difference));
			if (distance <= clusterEps) {
				int clusterIndex = iter - clusters.begin();
				clusterPoints[i] = clusterIndex;
				break;
			}
			iter++;
		}
		if (iter == iterEnd) {
			clusters.push_back(point);
			int clusterIndex = clusters.size() - 1;
			clusterPoints[i] = clusterIndex;
		}
	}
	end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

	/* write to CSV */
	std::ofstream outputFile("output.csv");
	for (int i = 0; i < numPoints; i++) {
		outputFile << h_inputPoints[i] << ",";
		outputFile << h_inputPoints[i + numPoints] << ",";
		outputFile << h_inputPoints[i + 2 * numPoints] << ",";
		outputFile << clusterPoints[i] << "\n";
	}

	printf("\nClustering elapsed time: %f", time);
	printf("\nNum clusters: %lu", clusters.size());
	/* CLUSTERING - fine */

	return 0;
}

