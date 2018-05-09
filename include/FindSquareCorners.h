#ifndef FIND_SQUARE_CORNERS_H
#define FIND_SQUARE_CORNERS_H

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using std::vector;
using namespace std;
using namespace cv;

class FindSquareCorners
{
public:
	FindSquareCorners();
	~FindSquareCorners();

public:
	bool findSquareCorners(Mat &gray, vector<Point2f> &resultCornors, float scoreThreshold);
};

#endif