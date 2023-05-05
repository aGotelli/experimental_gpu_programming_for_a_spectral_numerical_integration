#include <cstdio>
#include <cstdlib>
#include <vector>

//  CUDQ Basic Linear Algebra 
#include <cublas_v2.h>
#include <cuda_runtime.h>

//  A set of functions to check the result of operations
#include "utilities.h"

using data_type = double;



__global__ void block_copy(data_type* src, 
                           const int src_rows,
                           data_type* dst, 
                           const int dst_rows,
                           const int row_index,
                           const int col_index) 
{
    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int tidx = threadIdx.x;
    const unsigned int bidx = blockIdx.x;

    dst[tidx+bidx*dst_rows] = src[row_index+col_index*src_rows+tidx+bidx*src_rows];

}

int main(int argc, char *argv[]) 
{
    /* step 1: create cublas handle, bind a stream */
    //  The handle and the stream
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    CUBLAS_CHECK(
        cublasCreate(&cublasH)
    );

    CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
    );
    CUBLAS_CHECK(
        cublasSetStream(cublasH, stream)
    );


    /*  Now we define some dummy data that we can verify on paper if needed
     *
     *   A = | 1  6 11 16 21 |
             | 2  7 12 17 22 |
             | 3  8 13 18 23 |
             | 4  9 14 19 24 |
             | 5 10 15 20 25 |
     * 
     *   B = | 0 0 0 |
     *       | 0 0 0 |
     *       | 0 0 0 |
     */

    std::vector<data_type> A;
    const unsigned int rows_a = 5;
    const unsigned int cols_a = 5;
    for(int i=1; i<= rows_a*cols_a; i++)
        A.push_back(i);


    //  Keep it as square matrix
    const unsigned int rows_b = 2;
    const unsigned int cols_b = 3;
    std::vector<data_type> B(rows_b*cols_b, 0);   //  Matrix stored as vector

    //  Pointer to the devide data (the data stored on the GPU)
    data_type *d_A = nullptr;
    data_type *d_B = nullptr;

    //  Log the matrices
    printf("A\n");
    print_matrix(rows_a, cols_a, A.data(), rows_a);
    printf("=====\n");

    printf("B\n");
    print_matrix(rows_b, cols_b, B.data(), rows_b);
    printf("=====\n");

    

    /* step 2: copy data to device */
    //  Allocate the memory (This is usually done in initialisation because very expensive in time)
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size())
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size())
    );

    //  Actual copy of the data
    unsigned int size_of_A_in_bytes = sizeof(data_type) * A.size();
    CUDA_CHECK(
        cudaMemcpyAsync(d_A,                    //  local pointer on the GPU
                        A.data(),               //  Pointer to the data
                        size_of_A_in_bytes,     //  The amount of memory we need in bytes
                        cudaMemcpyHostToDevice, //  Copy from the Host (the CPU) to the device (the GPU)                      
                        stream)
    );
    CUDA_CHECK(
        cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream)
    );



    /*
    Now we have data on the GPU.

    We will perform some basic block operations to see other functionalities.

    Here we use a kernel function that extracts a submatrix of the same dimension on B from A.
    The copy starts at an address A(left_upper_corner_row_index, left_upper_corner_col_index) which
    indicates the upper left corner of the submatrix we want to extract.
    This submatrix is copied into the B matrix
    */
    const unsigned int left_upper_corner_row_index = 2;
    const unsigned int left_upper_corner_col_index = 2;
    block_copy<<<cols_b, rows_b>>>(d_A, rows_a, d_B, rows_b, left_upper_corner_row_index, left_upper_corner_col_index);

    //  We know have copied part of A into B. to check the result, we move back the matrices into the host memory.

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   B = | 1.0 2.0 3.0 4.0 |
     */

    printf("B\n");
    print_matrix(rows_b, cols_b, B.data(), rows_b);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;

}
