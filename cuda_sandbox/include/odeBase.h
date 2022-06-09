#ifndef ODE_BASE_H
#define ODE_BASE_H

#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"
#include "lie_algebra_utilities.h"
#include "spectral_integration_library.h"
#include "globals.h"

template <unsigned int t_stateDim, unsigned int t_numNodes>
class odeBase {
public:
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> A;
    Eigen::Vector<double, t_stateDim> x0;
    Eigen::Vector<double, t_stateDim*t_numNodes> b;
    Eigen::Matrix<double, t_numNodes, t_numNodes> Dn;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> D;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> P;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> Dp;
    Eigen::VectorXd qe;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd Lambda;
    std::array<std::array<Eigen::MatrixXd, t_numNodes>, 2> Phi_array;
    integrationDirection direction;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnHandle_t cusolverH = NULL;

    double* d_b_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_x0 = nullptr;
    double* d_A_NN = nullptr;
    double* d_D_NN = nullptr;
    double* d_A = nullptr;
    double* d_b = nullptr;
    double* d_Dp = nullptr;
    double* d_P = nullptr;
    double* d_K = nullptr;
    double* d_phi_array = nullptr;
    double* d_qe = nullptr;
    double* d_AP = nullptr;
    double* d_Ipiv = nullptr;

    int info = 0;
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */
    
    odeBase(integrationDirection t_direction) {
        constexpr unsigned int probDim = t_numNodes*t_stateDim;
        constexpr unsigned int unknownDim = t_numNodes*(t_stateDim-1);

        stateDim = t_stateDim;
        numNodes = t_numNodes;

        direction = t_direction;

        Dn = getDn<t_numNodes>(direction);
        D = kroneckerProduct<probDim, probDim>(Eigen::Matrix<double, t_stateDim, t_stateDim>::Identity(), Dn);
        P = getP<t_stateDim, t_numNodes>();
        Dp = P.transpose() * D * P;

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        CUBLAS_CHECK(cublasCreate(&cublasH));
        CUBLAS_CHECK(cublasSetStream(cublasH, stream));

        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    };

    ~odeBase() {
        cudaFree(d_b_NN);
        cudaFree(d_D_IN);
        cudaFree(d_x0);
        cudaFree(d_A_NN);
        cudaFree(d_D_NN);
        cudaFree(d_info);
        cudaFree(d_work);
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_Dp);
        cudaFree(d_P);
        cudaFree(d_K);
        cudaFree(d_phi_array);
        cudaFree(d_qe);
        cudaFree(d_Ipiv);
    }

    void initMemory() {
        constexpr unsigned int probDim = t_numNodes*t_stateDim;
        constexpr unsigned int unknownDim = t_stateDim*(t_numNodes-1);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_NN), sizeof(double) * unknownDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_IN), sizeof(double) * unknownDim*t_stateDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x0), sizeof(double) * t_stateDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_NN), sizeof(double) * unknownDim*unknownDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D_NN), sizeof(double) * unknownDim*unknownDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * probDim*probDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * probDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Dp), sizeof(double) * probDim*probDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_P), sizeof(double) * probDim*probDim));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_K), sizeof(double) * 3*t_numNodes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_phi_array), sizeof(double) * na*na*ne*t_numNodes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qe), sizeof(double) * na*ne));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_AP), sizeof(double) * probDim*probDim));
    }

    void copyDataToDevice() {
        constexpr unsigned int probDim = t_numNodes*t_stateDim;
        constexpr unsigned int unknownDim = t_stateDim*(t_numNodes-1);
        CUDA_CHECK(cudaMemcpyAsync(d_x0, x0.data(), sizeof(double) * t_stateDim, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_info, &info, sizeof(int), cudaMemcpyHostToDevice, stream));
        //CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, b.data(), sizeof(double) * probDim, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_Dp, Dp.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_P, P.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice, stream));

        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, unknownDim, unknownDim, d_A_NN, unknownDim, &lwork));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
    }

    void copy_phi_qe() {

        CUDA_CHECK(cudaMemcpyAsync(d_qe, qe.data(), sizeof(double) * na*ne, cudaMemcpyHostToDevice, stream));

        Eigen::Matrix<double, na, na*ne*t_numNodes> phi_matrix;
        for (unsigned int i = 0; i < t_numNodes; ++i) {
            phi_matrix.block(0, i*na*ne, na, na*ne) = Phi_array[direction][i];
        }

        CUDA_CHECK(cudaMemcpyAsync(d_phi_array, phi_matrix.data(), sizeof(double) *na*t_numNodes*na*ne, cudaMemcpyHostToDevice, stream));
    }

    virtual void getA(){};
    virtual void getb(){};

private:
    unsigned int stateDim;
    unsigned int numNodes;
};

#endif