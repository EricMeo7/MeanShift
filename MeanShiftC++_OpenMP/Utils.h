#ifndef UNTITLED_UTILS_H
#define UNTITLED_UTILS_H

#include "Point.h"
#include "ClusterManager.h"

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <experimental/filesystem>



std::vector<Point> readPointsFromCSV(const std::string& fileName);

void writeClustersToCSV(const std::string& fileName, const ClusterManager& clusterManager);

void writeTimeToCSV(const std::string& fileName, int numThreads, float time, int numPoints, int dimensions, int numClusters, float bandwidth);

float euclideanDistance(const Point &x, const Point &y);

#endif //UNTITLED_UTILS_H
