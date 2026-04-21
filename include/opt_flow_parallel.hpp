#ifndef OPT_FLOW_PARALLEL_HPP
#define OPT_FLOW_PARALLEL_HPP

#include <opencv2/opencv.hpp>

// Parallel Horn–Schunck (OpenMP)
void hornSchunckParallel(const cv::Mat &I1, const cv::Mat &I2,
                         cv::Mat &u, cv::Mat &v,
                         float alpha, int iterations);

#endif
