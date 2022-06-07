#ifndef Q_INTEGRATOR_H
#define Q_INTEGRATOR_H

#include "globals.h"

__global__ void matmul(const double* A, const unsigned int m, const unsigned int n, double* x, double* y) {
    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int tidx = threadIdx.x;
    const unsigned int bidx = blockIdx.x;

    for (unsigned int i = 0; i < n; ++i) {
        //y[tid] +=  A[tid+i*m]*x[i];
        x[i] = A[i*m];
    }
}
__global__ void copy_K_to_A(double* K, double* A, unsigned int ld) {
        const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
        const unsigned int tidx = threadIdx.x;
        const unsigned int bidx = blockIdx.x;

        unsigned int curr_a = tid+tid*ld;
        
        A[curr_a] = 0;
        A[curr_a+num_ch_nodes] = 0.5*K[3*tid+0];
        A[curr_a+2*num_ch_nodes] = 0.5*K[3*tid+1];
        A[curr_a+3*num_ch_nodes] = 0.5*K[3*tid+2];

        A[curr_a+num_ch_nodes*ld] = -0.5*K[3*tid+0];
        A[curr_a+num_ch_nodes*ld+num_ch_nodes] = 0;
        A[curr_a+num_ch_nodes*ld+2*num_ch_nodes] = -0.5*K[3*tid+2];
        A[curr_a+num_ch_nodes*ld+3*num_ch_nodes] = 0.5*K[3*tid+1];

        A[curr_a+2*num_ch_nodes*ld] = -0.5*K[3*tid+1];
        A[curr_a+2*num_ch_nodes*ld+num_ch_nodes] = 0.5*K[3*tid+2];
        A[curr_a+2*num_ch_nodes*ld+2*num_ch_nodes] = 0;
        A[curr_a+2*num_ch_nodes*ld+3*num_ch_nodes] = -0.5*K[3*tid+0];

        A[curr_a+3*num_ch_nodes*ld] = -0.5*K[3*tid+2];
        A[curr_a+3*num_ch_nodes*ld+num_ch_nodes] = -0.5*K[3*tid+1];
        A[curr_a+3*num_ch_nodes*ld+2*num_ch_nodes] = 0.5*K[3*tid+0];
        A[curr_a+3*num_ch_nodes*ld+3*num_ch_nodes] = 0;
    }   

template <unsigned int t_stateDim, unsigned int t_numNodes>
class qIntegrator : public odeBase<t_stateDim, t_numNodes> {
public:
    qIntegrator(integrationDirection t_direction) : odeBase<t_stateDim, t_numNodes>(t_direction) {
        
    };

    void getA() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;

        // std::array<Eigen::Vector3d, t_numNodes> K;
        // //Eigen::Matrix<double, probDim, probDim> A = Eigen::Matrix<double, probDim, probDim>::Zero();
        // Eigen::Matrix<double, t_stateDim, t_stateDim> A_at_chebyshev_point;

        // for(unsigned int i=0; i < t_numNodes; i++){

        //     //  Extract the curvature from the strain
        //     K[i] = this->Phi_array[this->direction][i]*this->qe;

        //     //  Compute the A matrix of Q' = 1/2 A(K) Q
        //     A_at_chebyshev_point <<      0, -K[i](0),  -K[i](1),  -K[i](2),
        //                             K[i](0),     0,   K[i](2),  -K[i](1),
        //                             K[i](1), -K[i](2),      0,   K[i](0),
        //                             K[i](2),  K[i](1),  -K[i](0),      0;

        //     A_at_chebyshev_point = 0.5*A_at_chebyshev_point;

        //     for (unsigned int row = 0; row < A_at_chebyshev_point.rows(); ++row) {
        //         for (unsigned int col = 0; col < A_at_chebyshev_point.cols(); ++col) {
        //             int row_index = row*num_ch_nodes+i;
        //             int col_index = col*num_ch_nodes+i;
        //             this->A(row_index, col_index) = A_at_chebyshev_point(row, col);
        //         }
        //     }

        // }
        const double alpha = 1;
        const double beta = 0;
        CUBLAS_CHECK(cublasDgemvStridedBatched(this->cublasH,
                                                CUBLAS_OP_N,
                                                na, na*ne,
                                                &alpha,
                                                this->d_phi_array, na,
                                                na*na*ne,
                                                this->d_qe, 1,
                                                0,
                                                &beta,
                                                this->d_K, 1,
                                                3,
                                                t_numNodes));

        // // for (unsigned int i = 0; i < t_numNodes; i++) {
        // //     CUBLAS_CHECK(cublasDgemv(this->cublasH, CUBLAS_OP_N,
        // //                              na, na*ne,
        // //                              &alpha,
        // //                              this->d_phi_array+i*na*na*ne, na,
        // //                              this->d_qe, 1,
        // //                              &beta,
        // //                              this->d_K+i*3, 1));
        // // }

        copy_K_to_A<<<1, t_numNodes>>>(this->d_K, this->d_A, probDim);
        // // CUDA_CHECK(
        // //     cudaMemcpyAsync(this->d_A, A.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice, this->stream)
        // // );

        // Eigen::Matrix<double, 3*t_numNodes, 1> k_error;
        // CUDA_CHECK(
        //     cudaMemcpyAsync(k_error.data(), this->d_K, sizeof(double) * 3*t_numNodes, cudaMemcpyDeviceToHost, this->stream)
        // );

        // for (unsigned int i = 0; i < t_numNodes; ++i) {
        //     std::cout << k_error.block(i*3, 0, 3, 1) - K[i] << "\n" << std::endl;
        // }

        // Eigen::Matrix<double, probDim, probDim> error;
        // CUDA_CHECK(
        //     cudaMemcpyAsync(error.data(), this->d_A, sizeof(double) * probDim*probDim, cudaMemcpyDeviceToHost, this->stream)
        // );

        // for (unsigned int i = 0; i < this->A.cols(); ++i) {
        //     std::cout << "Col: " << i << std::endl;
        //     for (unsigned int j = 0; j < this->A.rows(); ++j) {
        //         std::cout << "Row " << j << ": " << this->A(i, j) - error(i, j) << std::endl;
        //     }
        //     std::cout << "\n\n" << std::endl;
        // }
        //std::cout << this->A-error << std::endl;
    }

    void getb() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        this->b = Eigen::Vector<double, probDim>::Zero();
    }
};

#endif