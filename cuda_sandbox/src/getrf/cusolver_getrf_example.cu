#include <cstdio>
#include <cstdlib>
#include <vector>

#include <utility>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

/* 
## CUDA APIs involved
- [cusolverDnDgetrf API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrf)
- [cusolverDnDgeqrs API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrs)
*/


#include <iostream>

#include "tictoc.h"




void generate_random_dense_matrix(int M, int N, double **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    double *A   = (double *)malloc(sizeof(double) * M * N);

    // For each column
    for (j = 0; j < N; j++)
    {
        // For each row
        for (i = 0; i < M; i++)
        {
            double dr = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
            //printf("A[j * M + i] = %f \n",A[j * M + i]);
        }
    }

    *outA = A;
}


int main(int argc, char *argv[]) 
{

    /* step 0 : declare handle and stream */
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    tictoc preprocessing, lu_factorization, inversion;




    int colsA = 80;
    int rowsA = colsA;
    int NN = colsA;
    int MM = rowsA;

    double *h_A = NULL; // dense matrix from CSR(A)
    double *h_b = NULL; // b = ones(m,1)

    // Generate inputs
    srand(9384);  
    generate_random_dense_matrix(MM, NN, &h_A);
    generate_random_dense_matrix(MM, NN, &h_b);


    //const int m = 9;
    const int m = colsA;
    const int lda = m;
    const int ldb = m;

    

    /*       
        For the factorization of LU take this example: 

        Given a matrix A like this we get the following LU factorizations
        depending if we are using the pivoting or not

     *       | 1 2 3  |
     *   A = | 4 5 6  |
     *       | 7 8 10 |
     *
     * without pivoting: A = L*U
     *       | 1 0 0 |      | 1  2  3 |
     *   L = | 4 1 0 |, U = | 0 -3 -6 |
     *       | 7 2 1 |      | 0  0  1 |
     *
     * with pivoting: P*A = L*U
     *       | 0 0 1 |
     *   P = | 1 0 0 |
     *       | 0 1 0 |
     *
     *       | 1       0     0 |      | 7  8       10     |
     *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
     *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
    
        and then we solve Ax = b 

        a good explaination is give n here (if you want to know)
        https://courses.engr.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html
     */



    /*
    Dummy copy of the values into a matrix in array form.

    It is easier to deal with array than pointers.

    It is just for a more natural use of cudaMemcpyAsync as you will have the matrices initialized as vectors and not pointers


    However, note that the matrices have to in the form of a vector.
    The stack of the matrix is column-wise.

    */



    std::vector<int> Ipiv(m, 0);
    int info = 0;

    double *d_A = nullptr; /* device copy of A */
    double *d_B = nullptr; /* device copy of B */
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    const int pivot_on = 0; // pivot off is not numerically stable
    /*
    For the pivoting we reeeeeeeally slow things down a lot.
    Hoever it is necessary if we have zeros at (0, 0).
    So we really need to check the matrices before insterting or not the pivoting.
    

    I think (and hope) that we do not need it.

    */


    /* step 1: create cusolver handle, bind a stream
    
        same reason as in matrix multiplication 
     */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));


    
    
    /* step 2: allocate and copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * m*m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

   
    preprocessing.tic();


     CUDA_CHECK(
        cudaMemcpyAsync(d_A, h_A, sizeof(double) * m*m, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_B, h_b, sizeof(double) * m, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
    cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice));

     /* step 3: query working space of getrf */
    //  cuda solver api need CUSOLVER_CHECK
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));

    //  common CUDA need CUDA_CHECK
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    preprocessing.toc("Preprocessing");

    /*
    From the output you will see how expensive is copy this stuff....
    It it like 1/3 of the total computation time...
    It's just crazy....

    Luckily we will do everything on device in the future.
    However, for the moment just initialize the matrices with the Eigen routine we have.
    The we will start porting the initialization on device

    The good thing of breaking down the work in simple tasks is that we can measure the 
    improvement you make on the computational time.

    */

    /* step 4: LU factorization 
    
    A good explaination is given here 
    https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrf

    
    */

    lu_factorization.tic();


    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL, d_info));
    }


    lu_factorization.toc("LU factorization");


    inversion.tic();

    /*
     * step 5: solve A*X = B


     If we take the exaple given before, 
     the resoult will be this for a b as below


     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                        d_A, lda, d_Ipiv, d_B, ldb, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                        d_A, lda, NULL, d_B, ldb, d_info));
    }


    inversion.toc("Finish of the inversion");


    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

