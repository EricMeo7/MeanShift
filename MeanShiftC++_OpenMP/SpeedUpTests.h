//
// Created by corsinovi on 10/08/2023.
//

#ifndef UNTITLED_SPEEDUPTESTS_H
#define UNTITLED_SPEEDUPTESTS_H
#include <iostream>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <experimental/filesystem>

#include "Point.h"
#include "ClusterManager.h"
#include "Utils.h"
#include "MeanShift.h"


#define BANDWIDTH 2
#define CLUSTER_EPS 1.0
#define NUM_TEST_ITERATIONS 5


void goWithDifferentNumberOfPoints(const std::string& inputPath, const std::string& outputPath);

void calculateMeanShift(float bandwidth, const std::string& inputFileName, const std::string& outputClustersFileName, const std::string& outputTimeFileName);

#endif //UNTITLED_SPEEDUPTESTS_H
