#ifndef UNTITLED_MAINSHIFT_H
#define UNTITLED_MAINSHIFT_H
#include "Point.h"
#include "Utils.h"

#define NUM_ITERATIONS 10

std::vector<Point> meanShift(const std::vector<Point> &points, float bandwidth, int numThreads=1);

Point shiftPoint(const Point &oldPoint, const std::vector<Point> &allPoints, float bandwidth);

float gaussianKernel(float distance, float bandwidth);
#endif //UNTITLED_MAINSHIFT_H
