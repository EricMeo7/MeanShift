#include "MeanShift.h"
#include "Utils.h"


std::vector<Point> meanShift(const std::vector<Point> &points, float bandwidth, int numThreads) {
    std::vector<Point> shiftedPoints = points;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
#pragma omp parallel for default(none) shared(points, bandwidth, shiftedPoints) schedule(static) num_threads(numThreads)
        //#pragma omp parallel for default(none) shared(points, bandwidth, shiftedPoints) schedule(dynamic) num_threads(numThreads)
        for (int k = 0; k < points.size(); k++) {
            Point shiftedPoint = shiftPoint(shiftedPoints[k], points, bandwidth);
            shiftedPoints[k] = shiftedPoint;
        }
    }
    return shiftedPoints;
}

Point shiftPoint(const Point &point, const std::vector<Point> &allPoints, float bandwidth) {
    float totalWeight = 0.0;
    Point shiftedPoint(point.getNumDimensions());
    float distance;
    for (auto &p : allPoints){
        distance = euclideanDistance(point, p);
        if (distance <= bandwidth) {
            float weight = gaussianKernel(distance, bandwidth);
            shiftedPoint += p * weight;
            totalWeight += weight;
        }
    }
    shiftedPoint /= totalWeight;
    return shiftedPoint;
}

float gaussianKernel(float distance, float bandwidth){
    return std::exp(-(distance * distance) / (2 * powf(bandwidth, 2)));
}

