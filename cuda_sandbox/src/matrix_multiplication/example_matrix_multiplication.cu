#include <cstdio>
#include <cstdlib>
#include <vector>

#include <fstream>

//  CUDQ Basic Linear Algebra 
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utilities.h"  //  Utilities for error handling (must be after cublas or cusolver)


#include <Eigen/Dense>




void generate_random_dense_matrix(int M, int N, double **outA, int range=100)
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
            A[j * M + i] = (dr / rMax) * range;
            //printf("A[j * M + i] = %f \n",A[j * M + i]);
        }
    }

    *outA = A;
}


void LoadEigenMatrixFromFile(Eigen::MatrixXd &t_matrix, 
                             std::string t_file_name,
                             std::string t_file_path="src/matrix_multiplication/data/",
                             const Eigen::IOFormat &t_format=Eigen::IOFormat())
{
    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c 
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    std::vector<double> matrixEntries;

    std::string file_to_open = "../" + t_file_path + t_file_name;
 
    // in this object we store the data from the matrix
    std::ifstream matrixDataFile;
    matrixDataFile.open(file_to_open);

    if(not matrixDataFile.is_open()){
        std::cout << "Could not open the file" << std::endl;
        return;
    }
    
 
    // this variable is used to store the row of the matrix that contains commas 
    std::string matrixRowString;
 
    // this variable is used to store the matrix entry;
    std::string matrixEntry;
 
    // this variable is used to track the number of rows
    int matrixRowNumber = 0;
 
    // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    while (std::getline(matrixDataFile, matrixRowString)) 
    {
        //convert matrixRowString that is a string to a stream variable.
        std::stringstream matrixRowStringStream(matrixRowString); 

        // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        while (std::getline(matrixRowStringStream, matrixEntry, ',')) 
        {
            //here we convert the string to double and fill in the row vector storing all the matrix entries
            matrixEntries.push_back(stod(matrixEntry));   
        }
        //update the column numbers
        matrixRowNumber++; 
    }
 
    // here we convet the vector variable into the matrix and return the resulting object, 
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    t_matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
 
}

using data_type = double;

int main(int argc, char *argv[]) 
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;


/* step 1: create cublas handle, bind a stream 

    Explaination:

    The handler is an object which is used to manage the api in its threads and eventually thrown errors


    Then there are the streams.

    Streams define the flow of data when copying.
    Imagine: We have 100 units of data (whatever it is) to copy from one place to another.
    Memory is already allocated. 
    Normal way: copy data 1, then data 2, than data 3, ..... untill the end.

    With streams, we copy data in parallel. It boils down to this.
    Here you can find a more detailed and clear explaination (with figures)

    Look only at the first 6 slides

    https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf

*/
    //  cuda blas api need CUBLAS_CHECK
    CUBLAS_CHECK(
        cublasCreate(&cublasH)
    );
    
    //  common CUDA need CUDA_CHECK
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
    );
    CUBLAS_CHECK(
        cublasSetStream(cublasH, stream)
    );


/*  Now We need to initialize the matrices with some values
    Here I am just loading stuff from saved matrices in csv format
*/

    Eigen::MatrixXd A;
    LoadEigenMatrixFromFile(A, "A.csv");

    Eigen::MatrixXd P;
    LoadEigenMatrixFromFile(P, "P.csv");

    Eigen::MatrixXd Ap;
    LoadEigenMatrixFromFile(Ap, "Ap.csv");

    //  Matrices are stored in vector form by column wise


    /*

    Definition of matrices dimensions.

    Feel free to change these names, I do not like them neither...
    */
    const int rows_A = A.rows();
    const int cols_A = A.cols();
    const int rows_P = P.rows();
    const int cols_P = P.cols();
    const int lda = rows_A;
    const int ldb = rows_P;
    const int ldc = rows_P;



    Eigen::MatrixXd AP(rows_A, cols_A);
    Eigen::MatrixXd PAP(rows_P, cols_P);

    const data_type alpha = 1.0;
    const data_type beta = 0.0;


    /*
    The data-type is double, but you can remove this typedef

    Here we declare the pointers

    */
    data_type *d_A = nullptr;
    data_type *d_P = nullptr;
    data_type *d_AP = nullptr;
    data_type *d_PAP = nullptr;




    
/* step 2: copy data to device 

    When we load the matrices, they are on the RAM.
    We need to allocate memory on the GPU for them.

    Then we perform the copy. 
    Note that we could use synchronous copy but here I want to show you the Async one.
    The difference it's just the name cudaMemcpy and cudaMemcpyAsync. Only the last one
    takes the parameter stream.

*/
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size())
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_P), sizeof(data_type) * P.size())
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_AP), sizeof(data_type) * P.size())
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_PAP), sizeof(data_type) * P.size())
    );

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream)
    );
    CUDA_CHECK(
        cudaMemcpyAsync(d_P, P.data(), sizeof(data_type) * P.size(), cudaMemcpyHostToDevice, stream)
    );



    /* step 3: compute 
    
    Here we just do the two operations AP= A*P and PAP = transpose(P)*AP.

    We use the function cublasDgemm which is composed of:

    clublas : CUDA implementation of the blas library (Basic Linear Algebra Subprograms)
    D : we are using matrices of double (double floating point precision)
    gemm : there is not a clear definition. The way I see it is GEneral Matrix Multiplication


    Now this function takes multiple parameters, everything is explained here : https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm

    I give some more details: 
   
   
    The function cublasDgemm actually performs the following operations
    
    C := alpha*op( A )op( B ) + beta C

    op( A ) and op( B ) are the operations performed on the matrices A and B respectively.
    These are respectively set as second and third parameters of the function.
    For example CUBLAS_OP_N will leave the matrix as it is wile CUBLAS_OP_T will transpose the matrix.
    alpha and beta are scaling factors. alpha can be used to scale the multiplication A*B by a constant scalar value 
    while beta will create a recursive increment on C. In our case, beta will typically be zero.
    rows_A rows_P and cols_A express the number of rows and columns of the involved matrices. In this case, we have squared matrices
    thus they are all the same.

    */

    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_A, rows_P, cols_A, &alpha, d_A, lda, d_P, ldb, &beta, d_AP, ldc)
    );

    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rows_A, rows_P, cols_A, &alpha, d_P, lda, d_AP, ldb, &beta, d_PAP, ldc)
    );


    /* step 4: copy data to host 
    
    see that the function is the same as before, but now I am passing cudaMemcpyDeviceToHost to copy from the device to the host (CPU)


    This is necessary as I will need to print
    
    */
    CUDA_CHECK(
        cudaMemcpyAsync(PAP.data(), d_PAP, sizeof(data_type) * P.size(), cudaMemcpyDeviceToHost,stream)
    );


    /*
    Before using the stuff in the host, wait for all the streams to have copied the data.
    */
    CUDA_CHECK(cudaStreamSynchronize(stream));




    /*
        Here If you want you can compare the results
    */
    Eigen::MatrixXd error = Ap - PAP;
    std::cout<< "error = " << std::endl << error.norm() <<std::endl << std::endl;



    /*

    ALWAYS AT THE END   

    Free the memory
    */

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_AP));
    CUDA_CHECK(cudaFree(d_PAP));


    /*
    Destry cuda objects
    */
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    


    std::cout << "Program finished correctly" << std::endl;

    return EXIT_SUCCESS;

return 4;
}
