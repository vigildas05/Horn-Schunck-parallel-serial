#include "opt_flow_serial.hpp"
#include "opt_flow_parallel.hpp"
#include <iostream>
#include <chrono>
using namespace cv;
using namespace std;

static void drawFlow(const Mat &u, const Mat &v, Mat &output)
{
    output = Mat::zeros(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; y += 5)
    {
        for (int x = 0; x < u.cols; x += 5)
        {
            Point2f pt(x, y);
            Point2f flow(u.at<float>(y, x), v.at<float>(y, x));
            line(output, pt, pt + flow, Scalar(0, 255, 0));
            circle(output, pt, 1, Scalar(0, 0, 255), -1);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: ./optical_flow <frame1> <frame2>" << endl;
        return -1;
    }

    Mat I1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat I2 = imread(argv[2], IMREAD_GRAYSCALE);
    if (I1.empty() || I2.empty()) {
        cerr << "Error: Cannot load images." << endl;
        return -1;
    }

    int K = 10000;

    I1.convertTo(I1, CV_32F, 1.0 / 255.0);
    I2.convertTo(I2, CV_32F, 1.0 / 255.0);

    Mat u_serial, v_serial, u_parallel, v_parallel;

    // Serial timing
    auto start_serial = chrono::high_resolution_clock::now();
    hornSchunckSerial(I1, I2, u_serial, v_serial, 0.5f, 1000);
    auto end_serial = chrono::high_resolution_clock::now();

    double serial_time = chrono::duration<double>(end_serial - start_serial).count();
    cout << "Serial Horn-Schunck time: " << serial_time << " seconds" << endl;

    // Parallel timing
    auto start_parallel = chrono::high_resolution_clock::now();
    hornSchunckParallel(I1, I2, u_parallel, v_parallel, 0.5f, 1000);
    auto end_parallel = chrono::high_resolution_clock::now();

    double parallel_time = chrono::duration<double>(end_parallel - start_parallel).count();
    cout << "Parallel Horn-Schunck time: " << parallel_time << " seconds" << endl;

    Mat flow_serial, flow_parallel;
    drawFlow(u_serial, v_serial, flow_serial);
    drawFlow(u_parallel, v_parallel, flow_parallel);

    imwrite("output/flow_serial.png", flow_serial);
    imwrite("output/flow_parallel.png", flow_parallel);

    imshow("Serial Flow", flow_serial);
    imshow("Parallel Flow", flow_parallel);
    waitKey(0);

    return 0;
}
