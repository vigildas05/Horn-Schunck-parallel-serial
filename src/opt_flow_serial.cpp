#include "opt_flow_serial.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cmath>

// Compute spatial and temporal derivatives (with slight smoothing for stability)
static void computeDerivativesHS(const cv::Mat &I1, const cv::Mat &I2,
                                    cv::Mat &Ix, cv::Mat &Iy, cv::Mat &It)
{
    cv::Mat I1s, I2s;
    cv::GaussianBlur(I1, I1s, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(I2, I2s, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REPLICATE);

    cv::Sobel(I1s, Ix, CV_32F, 1, 0, 3, 0.5, 0, cv::BORDER_REPLICATE);
    cv::Sobel(I1s, Iy, CV_32F, 0, 1, 3, 0.5, 0, cv::BORDER_REPLICATE);

    It = I2s - I1s;
}

static inline bool isFloat32Gray(const cv::Mat &m)
{
    return m.type() == CV_32F && m.channels() == 1;
}


void hornSchunckSerial(const cv::Mat &I1, const cv::Mat &I2,
                       cv::Mat &u, cv::Mat &v,
                       float alpha, int iterations)
{
    CV_Assert(!I1.empty() && !I2.empty());
    CV_Assert(I1.size() == I2.size());

    // Ensure float32 grayscale
    cv::Mat I1f, I2f;
    if (isFloat32Gray(I1))
        I1f = I1;
    else
        I1.convertTo(I1f, CV_32F);
    if (isFloat32Gray(I2))
        I2f = I2;
    else
        I2.convertTo(I2f, CV_32F);

    cv::Mat Ix, Iy, It;
    computeDerivativesHS(I1f, I2f, Ix, Iy, It);

    // Precompute denominator: alpha^2 + Ix^2 + Iy^2
    const float a2 = alpha * alpha;
    cv::Mat denom;
    {
        cv::Mat Ix2, Iy2;
        cv::multiply(Ix, Ix, Ix2);
        cv::multiply(Iy, Iy, Iy2);
        denom = Ix2 + Iy2;
        denom += a2;
    }

    // Initialize flows
    u = cv::Mat::zeros(I1f.size(), CV_32F);
    v = cv::Mat::zeros(I1f.size(), CV_32F);

    const int rows = I1f.rows;
    const int cols = I1f.cols;

    // In-place GaussSeidel updates
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (int y = 1; y < rows - 1; ++y)
        {
            for (int x = 1; x < cols - 1; ++x)
            {
                // 4-neighbor average (consistent with parallel)
                float u_avg = 0.25f * (u.at<float>(y - 1, x) + u.at<float>(y + 1, x) +
                                       u.at<float>(y, x - 1) + u.at<float>(y, x + 1));
                float v_avg = 0.25f * (v.at<float>(y - 1, x) + v.at<float>(y + 1, x) +
                                       v.at<float>(y, x - 1) + v.at<float>(y, x + 1));

                float ix = Ix.at<float>(y, x);
                float iy = Iy.at<float>(y, x);
                float it = It.at<float>(y, x);
                float den = denom.at<float>(y, x);

                float t = (ix * u_avg + iy * v_avg + it) / den;
                u.at<float>(y, x) = u_avg - ix * t;
                v.at<float>(y, x) = v_avg - iy * t;
            }
        }
        // borders remain unchanged
    }
}