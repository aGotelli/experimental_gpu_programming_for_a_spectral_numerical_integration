#ifndef SPECTRAL_INTEGRATION_LIBRARY_H
#define SPECTRAL_INTEGRATION_LIBRARY_H

#include <Eigen/Dense>
#include <fstream>
// #include "chebyshev_differentiation.h"
// #include "spectral_integration_utilities.h"


constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe
constexpr unsigned int num_ch_nodes = 11;

// /*!
//  * \brief writeToFile writes a Eigen matrix into file
//  * \param t_name    name of the file
//  * \param t_matrix  the Eigen matrix to write into the file
//  * \param t_relative_path_from_build the relative path from the build folder to the file location. Default is none so the file is written in the build directory)
//  * \param t_format  the specification for writing. (Default in column major allignment, with comma column separator and 8 digits precision)
//  */
// void writeToFile(std::string t_name,
//                  const Eigen::MatrixXd &t_matrix,
//                  std::string t_relative_path_from_build = "",
//                  const Eigen::IOFormat &t_format = Eigen::IOFormat(16, 0, ","))
// {
//     if(not t_relative_path_from_build.empty()){
//         //  Ensure relative path ends with a backslash only if a path is given
//         if(t_relative_path_from_build[t_relative_path_from_build.length()-1] != '/') { //  not t_relative_path_from_build.ends_with('/'))
//             t_relative_path_from_build.append("/");
//     }


//     //  Ensure it ends with .csv
//     if(t_name.find(".csv") == std::string::npos)
//         t_name.append(".csv");

//     //  The file will be created in the location given by the realtive path and with the given name
//     const auto file_name_and_location = t_relative_path_from_build + t_name;

//     //  Create file in given location with given name
//     std::ofstream file(file_name_and_location.c_str());

//     //  Put matrix in this file
//     file << t_matrix.format(t_format);

//     //  Close the file
//     file.close();
//  }


// Eigen::Matrix<double, 3, 3> getHat(const Eigen::Vector3d t_v) {
//     Eigen::Matrix<double, 3, 3> hatMatrix;

//     hatMatrix << 0, -t_v(2), t_v(1),
//                  t_v(2), 0, -t_v(0),
//                  -t_v(1), t_v(0), 0;

//     return hatMatrix;
// }

template <unsigned int t_stateDim>
const Eigen::MatrixXd solveLinSys(cusolverDnHandle_t cusolver_handle,
                                  cublasHandle_t cublas_handle,
                                  cudaStream_t stream,
                                  double* b_NN,
                                  double* D_IN,
                                  double* x0,
                                  double* A_NN,
                                  double* D_NN,
                                  double* work,
                                  int* info) {
    
    constexpr int probDim = t_stateDim * num_ch_nodes;
    constexpr int unknownDim = t_stateDim * (num_ch_nodes-1);
    const double alpha = -1;
    const double beta = 0;
    const double beta_pos = 1;

    CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, unknownDim, t_stateDim, &alpha, D_IN, unknownDim, x0, 1, &beta_pos, b_NN, 1));
    CUBLAS_CHECK(cublasDgeam(cublas_handle, 
                             CUBLAS_OP_N, 
                             CUBLAS_OP_N, 
                             unknownDim, 
                             unknownDim, 
                             &alpha, 
                             A_NN, 
                             unknownDim, 
                             &beta_pos,
                             D_NN,
                             unknownDim,
                             A_NN,
                             unknownDim));

    std::vector<int> Ipiv(unknownDim, 0);
    int *d_Ipiv = nullptr; /* pivoting sequence */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
    const int pivot_on = 1;

    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolver_handle, unknownDim, unknownDim, A_NN, unknownDim, work, d_Ipiv, info));
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N, unknownDim, 1, A_NN, unknownDim, d_Ipiv, b_NN, unknownDim, info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolver_handle, unknownDim, unknownDim, A_NN, unknownDim, work, NULL, info));
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N, unknownDim, 1, A_NN, unknownDim, NULL, b_NN, unknownDim, info));
    }

    Eigen::Matrix<double, unknownDim, 1> X_NN;
    Eigen::Matrix<double, t_stateDim, 1> initState;

    CUDA_CHECK(
        cudaMemcpyAsync(X_NN.data(), b_NN, sizeof(double) * unknownDim, cudaMemcpyDeviceToHost, stream)
    );

    CUDA_CHECK(
        cudaMemcpyAsync(initState.data(), x0, sizeof(double) * t_stateDim, cudaMemcpyDeviceToHost, stream)
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto P = getP<t_stateDim, num_ch_nodes>();

    Eigen::Matrix<double, probDim, 1> X_tilde;
    X_tilde << initState, X_NN;
    X_tilde = P*X_tilde;
    Eigen::Matrix<double, num_ch_nodes, t_stateDim> X_stack = Eigen::Map<Eigen::Matrix<double, num_ch_nodes, t_stateDim, Eigen::ColMajor>>(X_tilde.data());

    CUDA_CHECK(cudaFree(d_Ipiv));
    return X_stack;
}

template <unsigned int t_stateDim>
const Eigen::MatrixXd integrateODE(cusolverDnHandle_t cusolver_handle,
                                   cublasHandle_t cublas_handle,
                                   cudaStream_t stream,
                                   double* b_NN,
                                   double* D_IN,
                                   double* x0,
                                   double* A_NN,
                                   double* D_NN,
                                   double* work,
                                   int* info,
                                   double* A,
                                   double* b,
                                   double* D,
                                   double* P) {

    constexpr int probDim = t_stateDim * num_ch_nodes;
    constexpr int unknownDim = t_stateDim * (num_ch_nodes-1);

    const double alpha = 1;
    const double beta = 0;
    const double beta_pos = 1;

    CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, probDim, probDim, &alpha, P, probDim, b, 1, &beta, b, 1));
    CUBLAS_CHECK(cublasDgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        A, probDim,
        P, probDim,
        &beta,
        A, probDim
    ));

    CUBLAS_CHECK(cublasDgemm(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        P, probDim,
        A, probDim,
        &beta,
        A, probDim
    ));

    CUBLAS_CHECK(cublasDgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        D, probDim,
        P, probDim,
        &beta,
        D, probDim
    ));

    CUBLAS_CHECK(cublasDgemm(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        probDim, probDim, probDim,
        &alpha,
        P, probDim,
        D, probDim,
        &beta,
        D, probDim
    ));

    CUBLAS_CHECK(cublasDcopy(
        cublas_handle, unknownDim,
        b+t_stateDim, 1,
        b_NN, 1
    ));

    for (unsigned int i = 0; i < t_stateDim; ++i) {
        CUBLAS_CHECK(cublasDcopy(
            cublas_handle, unknownDim,
            D+t_stateDim+i*probDim, 1,
            D_IN+i*unknownDim, 1
        ));
    }

    for (unsigned int i = 0; i < unknownDim; ++i) {
        CUBLAS_CHECK(cublasDcopy(
            cublas_handle, unknownDim,
            D+t_stateDim+t_stateDim*probDim+i*probDim, 1,
            D_NN+i*unknownDim, 1
        ));
    }

    for (unsigned int i = 0; i < unknownDim; ++i) {
        CUBLAS_CHECK(cublasDcopy(
            cublas_handle, unknownDim,
            A+t_stateDim+t_stateDim*probDim+i*probDim, 1,
            A_NN+i*unknownDim, 1
        ));
    }

    return solveLinSys<t_stateDim>(cusolver_handle,
                                   cublas_handle,
                                   stream,
                                   b_NN,
                                   D_IN,
                                   x0,
                                   A_NN,
                                   D_NN,
                                   work,
                                   info);
}
//template <unsigned int t_state_dimension>
// static const Eigen::MatrixXd integrateODE(const Eigen::VectorXd &t_initial_state,
//                                           const Eigen::MatrixXd A,
//                                           const Eigen::VectorXd b,
//                                           const integrationDirection t_direction,
//                                           const std::string filename){
//     constexpr unsigned int t_state_dimension = 4;
//     constexpr unsigned int prob_dimension = t_state_dimension * number_of_chebyshev_nodes;
//     constexpr unsigned int unknow_state_dimension = t_state_dimension * (number_of_chebyshev_nodes - 1);

//     typedef Eigen::Matrix<double, number_of_chebyshev_nodes, t_state_dimension, Eigen::ColMajor> MatrixNchebNs;

//     typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
//     typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

//     typedef Eigen::Matrix<double, prob_dimension, t_state_dimension> MatrixNpNs;

//     typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
//     typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

//     typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

//     typedef Eigen::Matrix<double, number_of_chebyshev_nodes, t_state_dimension, Eigen::ColMajor> MatrixNchebNs;

//     const MatrixNpNp  P = getP<t_state_dimension, number_of_chebyshev_nodes>();

//     const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(t_direction);
//     const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(t_state_dimension, t_state_dimension), Dn);

//     const MatrixNpNp Ap = P.transpose() * A * P;
//     const MatrixNpNp Dp = P.transpose() * D * P;
//     const VectorNp bp   = P * b;

// //    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);
//     const MatrixNpNs D_IT = Dp.block(0, 0, prob_dimension, t_state_dimension);
//     const MatrixNpNs A_IT = Ap.block(0, 0, prob_dimension, t_state_dimension);
//     const VectorNp b_IT = ( D_IT - A_IT ) * t_initial_state;

//     const MatrixNuNu D_NN = Dp.block(t_state_dimension, t_state_dimension, unknow_state_dimension, unknow_state_dimension);
//     const MatrixNuNu A_NN = Ap.block(t_state_dimension, t_state_dimension, unknow_state_dimension, unknow_state_dimension);
//     const VectorNu ivp = b_IT.block(t_state_dimension, 0, unknow_state_dimension, 1);
//     const VectorNu b_NN   = bp.block(t_state_dimension, 0, unknow_state_dimension, 1);

//     const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

//     const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

//     const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

//     writeToFile(filename, X_stack);

//     return X_stack;
// }

#endif
