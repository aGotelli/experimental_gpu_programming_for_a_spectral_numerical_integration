#include <cstdio>
#include <cstdlib>
#include <vector>

#include <fstream>

//  CUDQ Basic Linear Algebra 
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utilities.h"  //  Utilities for error handling (must be after cublas or cusolver)


#include <Eigen/Dense>



/*
This function computes the following function:
F = alpha*op( A )*op( B ) + beta*C
It takes as parameters{
    Matrices: A, B, C
    Variables: alpha, beta
    Matrix operations on A and B: can be any of
        CUBLAS_OP_N = No operation
        CUBLAS_OP_T = Transpose
        CUBLAS_OP_C = Hessian?
*/


//  Matrices are stored in vector form by column wise

/*
Definition of matrices dimensions.

Here the leading dimension is the number of rows.
This is because we are using column major order.
If we were using row major order, the leading dimension would be the number of columns.

https://stackoverflow.com/questions/16376804/clarification-of-the-leading-dimension-in-cublas-when-transposing
*/
static Eigen::Matrix3d skew(const Eigen::Vector3d &t_v) {

    const int rows_A = A.rows();
    const int cols_A = A.cols();
    const int ld_A = rows_A;

    const int rows_B = B.rows();
    const int cols_B = B.cols();
    const int ld_B = rows_B;

    const int ld_C = rows_A;


    /*
    The data-type is double, but you can remove this typedef
    Here we declare the pointers

    */
    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;


    const auto size_of_double = sizeof(double);
    const auto size_of_A_in_bytes = size_of_double * A.size();
    const auto size_of_B_in_bytes = size_of_double * B.size();
    const auto size_of_C_in_bytes = size_of_double * rows_A * cols_B;


    //  Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), size_of_A_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), size_of_B_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C), size_of_C_in_bytes)
    );


    //  Copy the data
    CUDA_CHECK(
        cudaMemcpy(d_A, A.data(), size_of_A_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_B, B.data(), size_of_B_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_C, C.data(), size_of_C_in_bytes, cudaMemcpyHostToDevice)
    );


    //Variables: cublasH, 
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_A, rows_B, cols_B, &alpha, d_A, ld_A, d_B, ld_B, &beta, d_C, ld_C)
    );

    
}