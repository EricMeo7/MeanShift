#include "SpeedupTests.h"


void goWithDifferentNumberOfPoints(const std::string& inputPath, const std::string& outputPath){
    std::cout << "\nTesting with different number of points" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;

    std::vector<std::string> fileNames;
    for (const auto & entry : std::experimental::filesystem::directory_iterator(inputPath))
        fileNames.emplace_back(entry.path().filename().string());
    std::sort(fileNames.begin(), fileNames.end());
    for (const auto & fileName : fileNames) {
        calculateMeanShift(BANDWIDTH, inputPath +"/"+ fileName, outputPath + "/out_" + fileName, outputPath + "/ResultsSpeedUpTests");
        std::cout << "" << std::endl;
    }

    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\n\n" << std::endl;
}

void calculateMeanShift(float bandwidth, const std::string& inputFileName, const std::string& outputClustersFileName, const std::string& outputTimeFileName){
    std::vector<Point> points = readPointsFromCSV(inputFileName);
    int appo = omp_get_max_threads();
    for (int numThreads = 1; numThreads <= omp_get_max_threads(); numThreads++) {

        int numClusters = 0;

        // mean shift algorithm with "numThreads" threads
        std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
        std::vector<Point> shiftedPoints = meanShift(points, bandwidth, numThreads);
        std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();

        float totalTimeMeanShift = std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime).count();

        //  Cluster classification
        ClusterManager clusterManager = ClusterManager(points, shiftedPoints, CLUSTER_EPS);
        clusterManager.buildClusters();
        numClusters = clusterManager.getNumClusters();
        writeClustersToCSV(outputClustersFileName, clusterManager);


        std::cout << "Number of threads: " << numThreads << " || Time: " << totalTimeMeanShift
                  << " || Number of points: " << points.size() << " || Dimensions: " << points[0].getNumDimensions()
                  << " || Clusters: " << numClusters << " || Bandwidth: " << bandwidth << std::endl;
        writeTimeToCSV(outputTimeFileName, numThreads, totalTimeMeanShift , points.size(), points[0].getNumDimensions(), numClusters, bandwidth);
    }
}