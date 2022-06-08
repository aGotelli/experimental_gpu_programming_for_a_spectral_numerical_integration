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
class qIntegrator  {
public:
    qIntegrator(integrationDirection t_direction, std::array<std::array<Eigen::MatrixXd, num_ch_nodes>, 2> phi, cusolverDnHandle_t &t_cusolverH) : stateDim(t_stateDim) , numNodes(t_numNodes)
    {
        tictoc tictoc;
        tictoc.tic();
        constexpr unsigned int probDim = t_numNodes*t_stateDim;
        constexpr unsigned int unknownDim = t_stateDim*(t_numNodes-1);


        direction = t_direction;

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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_phi_array), sizeof(double) * na*na*ne*t_numNodes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_qe), sizeof(double) * na*ne));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_AP), sizeof(double) * probDim*probDim));



        Phi_array = phi;

        CUDA_CHECK(cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Dp, Dp.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_P, P.data(), sizeof(double) * probDim*probDim, cudaMemcpyHostToDevice));
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(t_cusolverH, unknownDim, unknownDim, d_A_NN, unknownDim, &lwork));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

        tictoc.toc("finish initialization constant members");
    
    };

    void getA(cublasHandle_t &t_cublasH) 
    {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;

        const double alpha = 1;
        const double beta = 0;
        CUBLAS_CHECK(cublasDgemvStridedBatched(t_cublasH,
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


        copy_K_to_A<<<1, t_numNodes>>>(this->d_K, this->d_A, probDim);

    }

    void getb() 
    {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        b = Eigen::Vector<double, probDim>::Zero();
    }

    void copy_phi_qe() {

        CUDA_CHECK(cudaMemcpy(d_qe, qe.data(), sizeof(double) * na*ne, cudaMemcpyHostToDevice));
        
        Eigen::Matrix<double, na, na*ne*t_numNodes> phi_matrix;
        for (unsigned int i = 0; i < t_numNodes; ++i) {
            phi_matrix.block(0, i*na*ne, na, na*ne) = Phi_array[direction][i];
        }

        CUDA_CHECK(cudaMemcpy(d_phi_array, phi_matrix.data(), sizeof(double) *na*t_numNodes*na*ne, cudaMemcpyHostToDevice));
        
    }

     void copyDataToDevice(cusolverDnHandle_t &t_cusolverH) {
        constexpr unsigned int probDim = t_numNodes*t_stateDim;
        constexpr unsigned int unknownDim = t_stateDim*(t_numNodes-1);
        CUDA_CHECK(cudaMemcpy(d_x0, x0.data(), sizeof(double) * t_stateDim, cudaMemcpyHostToDevice)); //?
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), sizeof(double) * probDim, cudaMemcpyHostToDevice));
        
    }

public:
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> A;
    Eigen::Vector<double, t_stateDim> x0;
    Eigen::Vector<double, t_stateDim*t_numNodes> b;
    Eigen::Matrix<double, t_numNodes, t_numNodes> Dn;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> D;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> P;
    Eigen::Matrix<double, t_stateDim*t_numNodes, t_stateDim*t_numNodes> Dp;
    Eigen::VectorXd qe;
    std::array<std::array<Eigen::MatrixXd, t_numNodes>, 2> Phi_array;
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
    double* d_phi_array = nullptr;
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