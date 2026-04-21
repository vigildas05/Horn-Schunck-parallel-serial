#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>

static void computeDerivativesHS(const cv::Mat &I1, const cv::Mat &I2,
                                 cv::Mat &Ix, cv::Mat &Iy, cv::Mat &It) {
    // Inputs are expected as CV_32F grayscale. Smooth slightly for stability.
    cv::Mat I1s, I2s;
    cv::GaussianBlur(I1, I1s, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(I2, I2s, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REPLICATE);

    cv::Sobel(I1s, Ix, CV_32F, 1, 0, 3, 0.5, 0, cv::BORDER_REPLICATE);
    cv::Sobel(I1s, Iy, CV_32F, 0, 1, 3, 0.5, 0, cv::BORDER_REPLICATE);

    // Temporal derivative (I2 - I1)
    It = I2s - I1s;
}

static inline bool isFloat32Gray(const cv::Mat &m) {
    return m.type() == CV_32F && m.channels() == 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc < 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " image1 image2 alpha iterations\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string img1_path = argv[1];
    std::string img2_path = argv[2];
    float alpha = std::stof(argv[3]);
    int iterations = std::stoi(argv[4]);

    cv::Mat I1f, I2f;         // full images (only on rank 0)
    int rows = 0, cols = 0;

    // Rank 0 reads images and computes derivatives
    cv::Mat Ix_full, Iy_full, It_full, denom_full;
    if (rank == 0) {
        cv::Mat I1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
        cv::Mat I2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
        if (I1.empty() || I2.empty()) {
            std::cerr << "Failed to load input images\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (I1.size() != I2.size()) {
            std::cerr << "Images must be the same size\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Convert to float
        I1.convertTo(I1f, CV_32F, 1.0/255.0);
        I2.convertTo(I2f, CV_32F, 1.0/255.0);

        rows = I1f.rows;
        cols = I1f.cols;

        // Compute derivatives on full image
        computeDerivativesHS(I1f, I2f, Ix_full, Iy_full, It_full);

        // denom = Ix^2 + Iy^2 + alpha^2
        cv::Mat Ix2, Iy2;
        cv::multiply(Ix_full, Ix_full, Ix2);
        cv::multiply(Iy_full, Iy_full, Iy2);
        denom_full = Ix2 + Iy2;
        denom_full += alpha * alpha;
    }

    // Broadcast image size to all ranks
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rows <= 0 || cols <= 0) {
        if (rank == 0) std::cerr << "Invalid image size\n";
        MPI_Finalize();
        return 1;
    }

    // Compute block sizes (rows) for each rank (handle non-divisible case)
    std::vector<int> counts_rows(comm_sz);
    std::vector<int> displs_rows(comm_sz);
    int base = rows / comm_sz;
    int rem = rows % comm_sz;
    int offset = 0;
    for (int r = 0; r < comm_sz; ++r) {
        counts_rows[r] = base + (r < rem ? 1 : 0);
        displs_rows[r] = offset;
        offset += counts_rows[r];
    }

    int local_rows = counts_rows[rank];
    int start_row = displs_rows[rank]; // global start row index for this rank

    // Prepare buffers for local chunks of derivatives and denom
    // Each block has local_rows * cols floats
    std::vector<float> Ix_local(local_rows * cols);
    std::vector<float> Iy_local(local_rows * cols);
    std::vector<float> It_local(local_rows * cols);
    std::vector<float> denom_local(local_rows * cols);

    // Prepare counts/displacements in terms of floats for Scatterv (counts * cols)
    std::vector<int> sendcounts(comm_sz), senddispls(comm_sz);
    for (int r = 0; r < comm_sz; ++r) {
        sendcounts[r] = counts_rows[r] * cols;            // number of floats
        senddispls[r] = displs_rows[r] * cols;            // float offset
    }

    // Scatter Ix, Iy, It, denom (root supplies contiguous float arrays)
    if (rank == 0) {
        // Ensure Mats are continuous
        if (!Ix_full.isContinuous() || !Iy_full.isContinuous() || !It_full.isContinuous() || !denom_full.isContinuous()) {
            Ix_full = Ix_full.clone();
            Iy_full = Iy_full.clone();
            It_full = It_full.clone();
            denom_full = denom_full.clone();
        }
    }

    MPI_Scatterv(rank==0 ? Ix_full.ptr<float>(0) : nullptr, sendcounts.data(), senddispls.data(), MPI_FLOAT,
                 Ix_local.data(), sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank==0 ? Iy_full.ptr<float>(0) : nullptr, sendcounts.data(), senddispls.data(), MPI_FLOAT,
                 Iy_local.data(), sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank==0 ? It_full.ptr<float>(0) : nullptr, sendcounts.data(), senddispls.data(), MPI_FLOAT,
                 It_local.data(), sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank==0 ? denom_full.ptr<float>(0) : nullptr, sendcounts.data(), senddispls.data(), MPI_FLOAT,
                 denom_local.data(), sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Convert the flat vectors into cv::Mat wrappers (no copy) for convenience
    cv::Mat Ix_mat(local_rows, cols, CV_32F, Ix_local.data());
    cv::Mat Iy_mat(local_rows, cols, CV_32F, Iy_local.data());
    cv::Mat It_mat(local_rows, cols, CV_32F, It_local.data());
    cv::Mat denom_mat(local_rows, cols, CV_32F, denom_local.data());

    // Initialize u and v (local part). We'll store them in contiguous vectors too.
    std::vector<float> u_local_buf(local_rows * cols, 0.0f);
    std::vector<float> v_local_buf(local_rows * cols, 0.0f);
    std::vector<float> u_next_buf(local_rows * cols, 0.0f);
    std::vector<float> v_next_buf(local_rows * cols, 0.0f);

    cv::Mat u_mat(local_rows, cols, CV_32F, u_local_buf.data());
    cv::Mat v_mat(local_rows, cols, CV_32F, v_local_buf.data());
    cv::Mat u_next_mat(local_rows, cols, CV_32F, u_next_buf.data());
    cv::Mat v_next_mat(local_rows, cols, CV_32F, v_next_buf.data());

    // Prepare halos: top and bottom rows for u and v (size cols)
    std::vector<float> u_top_halo(cols), u_bottom_halo(cols);
    std::vector<float> v_top_halo(cols), v_bottom_halo(cols);

    int prev = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next = (rank == comm_sz-1) ? MPI_PROC_NULL : rank + 1;

    // Start timing
    double start_time = MPI_Wtime();

    // Iterations
    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halo rows for u
        // 1) send first local row upwards (to prev) and receive last row of next into bottom_halo
        MPI_Sendrecv(u_mat.ptr<float>(0), cols, MPI_FLOAT, prev, 100,
                     u_bottom_halo.data(), cols, MPI_FLOAT, next, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 2) send last local row downwards (to next) and receive first row of prev into top_halo
        MPI_Sendrecv(u_mat.ptr<float>(local_rows-1), cols, MPI_FLOAT, next, 101,
                     u_top_halo.data(), cols, MPI_FLOAT, prev, 101,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Exchange halo rows for v (same tags but different buffer)
        MPI_Sendrecv(v_mat.ptr<float>(0), cols, MPI_FLOAT, prev, 200,
                     v_bottom_halo.data(), cols, MPI_FLOAT, next, 200,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(v_mat.ptr<float>(local_rows-1), cols, MPI_FLOAT, next, 201,
                     v_top_halo.data(), cols, MPI_FLOAT, prev, 201,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Now compute Jacobi update for local rows
        // For convenience, we treat global boundary rows specially:
        // global_y = start_row + y
        for (int y = 0; y < local_rows; ++y) {
            for (int x = 1; x < cols - 1; ++x) {
                int global_y = start_row + y;
                // If global border rows, keep as previous (like serial code copies borders)
                if (global_y == 0 || global_y == rows - 1 || x == 0 || x == cols - 1) {
                    // copy from u, v (no change)
                    u_next_mat.at<float>(y, x) = u_mat.at<float>(y, x);
                    v_next_mat.at<float>(y, x) = v_mat.at<float>(y, x);
                    continue;
                }

                // Compute u_avg and v_avg using 4-neighbor, taking care of halos
                float u_up, u_down, u_left, u_right;
                float v_up, v_down, v_left, v_right;

                // up
                if (y == 0) u_up = (prev == MPI_PROC_NULL) ? u_mat.at<float>(y, x) : u_top_halo[x];
                else u_up = u_mat.at<float>(y - 1, x);
                // down
                if (y == local_rows - 1) u_down = (next == MPI_PROC_NULL) ? u_mat.at<float>(y, x) : u_bottom_halo[x];
                else u_down = u_mat.at<float>(y + 1, x);
                // left, right (always local)
                u_left  = u_mat.at<float>(y, x - 1);
                u_right = u_mat.at<float>(y, x + 1);

                // same for v
                if (y == 0) v_up = (prev == MPI_PROC_NULL) ? v_mat.at<float>(y, x) : v_top_halo[x];
                else v_up = v_mat.at<float>(y - 1, x);
                if (y == local_rows - 1) v_down = (next == MPI_PROC_NULL) ? v_mat.at<float>(y, x) : v_bottom_halo[x];
                else v_down = v_mat.at<float>(y + 1, x);
                v_left  = v_mat.at<float>(y, x - 1);
                v_right = v_mat.at<float>(y, x + 1);

                float u_avg = 0.25f * (u_up + u_down + u_left + u_right);
                float v_avg = 0.25f * (v_up + v_down + v_left + v_right);

                // local derivatives
                float ix = Ix_mat.at<float>(y, x);
                float iy = Iy_mat.at<float>(y, x);
                float it = It_mat.at<float>(y, x);
                float den = denom_mat.at<float>(y, x);

                float t = (ix * u_avg + iy * v_avg + it) / den;
                u_next_mat.at<float>(y, x) = u_avg - ix * t;
                v_next_mat.at<float>(y, x) = v_avg - iy * t;
            }

            // Handle x=0 and x=cols-1 border columns by copying (like original)
            int x0 = 0, x1 = cols - 1;
            int global_y = start_row + y;
            u_next_mat.at<float>(y, x0) = u_mat.at<float>(y, x0);
            u_next_mat.at<float>(y, x1) = u_mat.at<float>(y, x1);
            v_next_mat.at<float>(y, x0) = v_mat.at<float>(y, x0);
            v_next_mat.at<float>(y, x1) = v_mat.at<float>(y, x1);

            // If global top/bottom border rows, copy entire row (to keep boundary stable)
            if (global_y == 0 || global_y == rows - 1) {
                for (int x = 0; x < cols; ++x) {
                    u_next_mat.at<float>(y, x) = u_mat.at<float>(y, x);
                    v_next_mat.at<float>(y, x) = v_mat.at<float>(y, x);
                }
            }
        } // end compute

        // swap buffers
        u_local_buf.swap(u_next_buf);
        v_local_buf.swap(v_next_buf);
        // update cv::Mat data pointers (no reallocation)
        u_mat = cv::Mat(local_rows, cols, CV_32F, u_local_buf.data());
        v_mat = cv::Mat(local_rows, cols, CV_32F, v_local_buf.data());
        u_next_mat = cv::Mat(local_rows, cols, CV_32F, u_next_buf.data());
        v_next_mat = cv::Mat(local_rows, cols, CV_32F, v_next_buf.data());
    } // end iterations

    // End timing
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Gather u and v back to root using Gatherv (counts in floats)
    std::vector<int> recvcounts(comm_sz), recvdispls(comm_sz);
    for (int r = 0; r < comm_sz; ++r) {
        recvcounts[r] = counts_rows[r] * cols;   // floats
        recvdispls[r] = displs_rows[r] * cols;   // float offset
    }

    std::vector<float> u_full, v_full;
    if (rank == 0) {
        u_full.resize(rows * cols);
        v_full.resize(rows * cols);
    }

    MPI_Gatherv(u_local_buf.data(), recvcounts[rank], MPI_FLOAT,
                rank==0 ? u_full.data() : nullptr, recvcounts.data(), recvdispls.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(v_local_buf.data(), recvcounts[rank], MPI_FLOAT,
                rank==0 ? v_full.data() : nullptr, recvcounts.data(), recvdispls.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Rank 0 can convert to cv::Mat and optionally save/visualize
    if (rank == 0) {
        cv::Mat u_out(rows, cols, CV_32F, u_full.data());
        cv::Mat v_out(rows, cols, CV_32F, v_full.data());

        // Example: compute flow visualization (angle->hue, magnitude->value)
        cv::Mat mag, ang;
        cv::cartToPolar(u_out, v_out, mag, ang, true); // ang in degrees
        cv::Mat hsv(rows, cols, CV_8UC3);
        cv::Mat hsv_split[3];
        // normalize mag to [0,255]
        cv::Mat mag_n;
        cv::normalize(mag, mag_n, 0, 255, cv::NORM_MINMAX);
        mag_n.convertTo(mag_n, CV_8U);
        ang.convertTo(ang, CV_8U, 0.5); // scale degrees to fit 0-180
        hsv_split[0] = ang;              // Hue
        hsv_split[1] = cv::Mat::ones(rows, cols, CV_8U) * 255; // Saturation
        hsv_split[2] = mag_n;            // Value
        cv::merge(hsv_split, 3, hsv);
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::imwrite("flow_vis.png", bgr);

        std::cout << "MPI Horn-Schunck time: " << elapsed_time << " seconds\n";
        std::cout << "Output flow visualization written to flow_vis.png\n";
    }

    MPI_Finalize();
    return 0;
}
