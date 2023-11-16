#ifndef UNTITLED_CLUSTERMANAGER_H
#define UNTITLED_CLUSTERMANAGER_H
#include <vector>

#include "Cluster.h"

class ClusterManager {

public:
    ClusterManager(std::vector<Point> &originalPoints, std::vector<Point> &shiftedPoints, float clusterEps);
    std::vector<Cluster> buildClusters();
    int getNumClusters();
    const std::vector<Cluster> &getClusters() const;

private:
    std::vector<Point> originalPoints;
    std::vector<Point> shiftedPoints;
    std::vector<Cluster> clusters;
    float clusterEps;
};

#endif //UNTITLED_CLUSTERMANAGER_H
