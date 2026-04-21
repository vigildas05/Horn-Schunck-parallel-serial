#ifndef OPT_FLOW_SERIAL_HPP
#define OPT_FLOW_SERIAL_HPP

#include <opencv2/opencv.hpp>

// Serial Horn–Schunck
void hornSchunckSerial(const cv::Mat &I1, const cv::Mat &I2,
                       cv::Mat &u, cv::Mat &v,
                       float alpha, int iterations);

#endif
