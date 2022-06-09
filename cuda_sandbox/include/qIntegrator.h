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

__global__ void computeK(const double* phi, const double* qe, double* K) {
    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int tidx = threadIdx.x;
    const unsigned int bidx = blockIdx.x;

    double value = 0;

    for (unsigned int i = 0; i < ne; ++i) {
        value += phi[bidx*3+i]*qe[tidx*na+i];
    }
    
    K[bidx*3+tidx] = value;
}

__global__ void copy_K_to_A(const double* K, double* A, unsigned int ld) {
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
class qIntegrator  {
public:
    qIntegrator(std::vector<double> t_x0, integrationDirection t_direction, std::array<double, num_ch_nodes*ne> t_phi, cusolverDnHandle_t &t_cusolverH) 
        : stateDim(t_stateDim) , numNodes(t_numNodes) {

        tictoc tictoc;
        tictoc.tic();
        constexpr unsigned int probDim = t_numNodes*t_stateDim;
        constexpr unsigned int unknownDim = t_stateDim*(t_numNodes-1);

        direction = t_direction;
        phi = t_phi;
        x0 = t_x0;

        Dn = getDn<t_numNodes>(direction);
        D = kroneckerProduct<probDim, probDim>(Eigen::Matrix<double, t_stateDim, t_stateDim>::Identity(), Dn);
        P = getP<t_stateDim, t_numNodes>();
        Dp = P.transpose() * D * P;

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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) * 3*t_numNodes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_phi), sizeof(double) * ne*t_numNodes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_qe), sizeof(double) * na*ne));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_AP), sizeof(double) * probDim*probDim));

        CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Dp, Dp.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_P, P.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice));
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(t_cusolverH, unknownDim, unknownDim, d_A_NN, unknownDim, &lwork));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
        CUDA_CHECK(cudaMemcpy(d_phi, phi.data(), sizeof(double) *t_numNodes*ne, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x0, x0.data(), sizeof(double) * t_stateDim, cudaMemcpyHostToDevice));

        getb();

        tictoc.toc("finish initialization constant members");
    
    };

    void getK(const double* phi, const double* qe, double* K) {
        computeK<<<t_numNodes, 3>>>(this->d_phi, this->d_qe, this->d_K);
    }

    void getA(const double* K) 
    {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;

        const double alpha = 1;
        const double beta = 0;
        // CUBLAS_CHECK(cublasDgemvStridedBatched(t_cublasH,
        //                                         CUBLAS_OP_N,
        //                                         na, na*ne,
        //                                         &alpha,
        //                                         this->d_phi_array, na,
        //                                         na*na*ne,
        //                                         this->d_qe, 1,
        //                                         0,
        //                                         &beta,
        //                                         this->d_K, 1,
        //                                         3,
        //                                         t_numNodes));

        copy_K_to_A<<<1, t_numNodes>>>(this->d_K, this->d_A, probDim);
    }

    void getb() 
    {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        std::array<double, probDim> b{0};
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), sizeof(double) * probDim, cudaMemcpyHostToDevice));
    }

    void copy_qe() {
        CUDA_CHECK(cudaMemcpy(d_qe, qe.data(), sizeof(double) * na*ne, cudaMemcpyHostToDevice));
    }

public:
    std::vector<double> x0;
    Eigen::Matrix<double, t_numNodes, t_numNodes> Dn;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> D;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> P;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> Dp;
    std::vector<double> qe;
    std::array<double, t_numNodes*ne> phi;
    integrationDirection direction;

    

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
    double* d_phi = nullptr;
    double* d_qe = nullptr;
    double* d_AP = nullptr;

    int info = 0;
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

private:
    const unsigned int stateDim;
    const unsigned int numNodes;
};

#endif