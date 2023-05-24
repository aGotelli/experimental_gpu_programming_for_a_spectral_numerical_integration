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

F = A.transpose() * B * C + D

*/




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


/* step 1: create cublas handle, bind a stream 

    Explaination:

    The handler is an object which is used to manage the api in its threads and eventually thrown errors


    Then there are the streams. But we don't need to know what they are.
    We do not use them for now.

    If you are interested:

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
    


/*  Now We need to initialize the matrices with some values
*/

    Eigen::MatrixXd A(6, 4);
    A << Eigen::MatrixXd::Identity(4, 4),
          0, 1, 0, 0,
          0, 0, 1, 0;
    std::cout << "A : \n" << A << "\n\n";
   

    Eigen::MatrixXd B(6, 3);
    double row_coeff;
    double col_coeff;
    const double scale = 10.0f;
    for(unsigned int row=0; row<B.rows(); row++){
        row_coeff = static_cast<double>(row)/scale;
        for(unsigned int col=0; col<B.cols(); col++){
            col_coeff = static_cast<double>(col + 1)/scale;

            B(row, col) = col_coeff + row_coeff;
        }
    }
    std::cout << "B : \n" << B << "\n\n";


    Eigen::MatrixXd C(3, 3);
    C << 1, 2, 3,
         1, 2, 3,
         1, 2, 3;
    std::cout << "C : \n" << C << "\n\n";


    Eigen::MatrixXd D(4, 3);
    Eigen::Vector4d base(1, 2, 3, 4);
    for(unsigned int col=0; col<D.cols(); col++){
        D.col(col) = base;
        base /= 2;
    }
    
    std::cout << "D : \n" << D << "\n\n";


    //  Compute the solution 
    const Eigen::MatrixXd BC_Eigen = B*C;
    const Eigen::MatrixXd ATBC_Eigen = A.transpose()*BC_Eigen;
    const Eigen::MatrixXd F_Eigen = ATBC_Eigen + D;

    std::cout << "BC_Eigen : \n" << BC_Eigen << "\n\n";
    std::cout << "ATBC_Eigen : \n" << ATBC_Eigen << "\n\n";
    std::cout << "F_Eigen : \n" << F_Eigen << "\n\n";


    Eigen::MatrixXd BC_CUDA = Eigen::MatrixXd::Zero(BC_Eigen.rows(), BC_Eigen.cols());
    Eigen::MatrixXd ATBC_CUDA = Eigen::MatrixXd::Zero(ATBC_Eigen.rows(), ATBC_Eigen.cols());
    Eigen::MatrixXd F_CUDA = Eigen::MatrixXd::Zero(F_Eigen.rows(), F_Eigen.cols());




    //  Matrices are stored in vector form by column wise


    /*

    Definition of matrices dimensions.

    Here the leqding dimension is the number of rows.
    This is because we are using column major order.
    If we were using row major order, the leading dimension would be the number of columns.

    https://stackoverflow.com/questions/16376804/clarification-of-the-leading-dimension-in-cublas-when-transposing
    */
    const int rows_A = A.rows();
    const int cols_A = A.cols();
    const int ld_A = rows_A;

    const int rows_B = B.rows();
    const int cols_B = B.cols();
    const int ld_B = rows_B;

    const int rows_C = C.rows();
    const int cols_C = C.cols();
    const int ld_C = rows_C;

    const int rows_D = D.rows();
    const int cols_D = D.cols();
    const int ld_D = rows_D;


    const int rows_BC = BC_Eigen.rows();
    const int cols_BC = BC_Eigen.cols();
    const int ld_BC = rows_BC;


    const int rows_F = F_Eigen.rows();
    const int cols_F = F_Eigen.cols();
    const int ld_F = rows_F;

    

    


    /*
    The data-type is double, but you can remove this typedef

    Here we declare the pointers

    */
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    double* d_D = nullptr;

    //  The pointer to the solution of B * C
    double* d_BC = nullptr;

    //  The pointer for the solution of the linear equation which is done directly in cuda
    double* d_F = nullptr;





    
/* step 2: copy data to device 

    When we load the matrices, they are on the RAM.
    We need to allocate memory on the GPU for them.

    Then we perform the copy. 
    Note that we could use synchronous copy but here I want to show you the Async one.
    The difference it's just the name cudaMemcpy and cudaMemcpyAsync. Only the last one
    takes the parameter stream.

*/
    const auto size_of_double = sizeof(double);
    const auto size_of_A_in_bytes = size_of_double * A.size();
    const auto size_of_B_in_bytes = size_of_double * B.size();
    const auto size_of_C_in_bytes = size_of_double * C.size();
    const auto size_of_D_in_bytes = size_of_double * D.size();

    const auto size_of_BC_in_bytes = size_of_double * BC_Eigen.size();
    // const auto size_of_ATBC_in_bytes = size_of_double * ATBC_Eigen.size();
    const auto size_of_F_in_bytes = size_of_double * F_Eigen.size();


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
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D), size_of_D_in_bytes)
    );

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_BC), size_of_BC_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_F), size_of_F_in_bytes)
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
    CUDA_CHECK(
        cudaMemcpy(d_F, D.data(), size_of_D_in_bytes, cudaMemcpyHostToDevice)
    );





    /* step 3: compute 
    
    Here we solve the system F = A.transpose() * B * C + D

    We use the function cublasDgemm which is composed of:

    cublas : CUDA implementation of the blas library (Basic Linear Algebra Subprograms)
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
    while beta will create a recursive increment on C. 

    */
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(       //  d_BC = d_B*d_C
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_B, rows_C, cols_B, &alpha, d_B, ld_B, d_C, ld_C, &beta, d_BC, ld_BC)
    );


    CUDA_CHECK(
        cudaMemcpy(BC_CUDA.data(), d_BC, size_of_BC_in_bytes, cudaMemcpyDeviceToHost)
    );
    std::cout << "BC_Eigen : \n" << BC_Eigen << "\n\n";
    std::cout << "BC_CUDA : \n" << BC_CUDA << "\n\n";


    beta = 1.0;
    //  d_F <- D
    //  d_F += d_A.transpose() * d_BC
    //  d_F = d_A.transpose() * d_BC + d_F
    auto rows_opA = cols_A;
    auto cols_opA = rows_A;
    CUBLAS_CHECK( //    Pay attention here the rows of A.transpose() are the cols of A and vice versa
        cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rows_opA, rows_BC, cols_opA, &alpha, d_A, ld_A, d_BC, ld_BC, &beta, d_F, ld_F)
    );


    CUDA_CHECK(
        cudaMemcpy(F_CUDA.data(), d_F, size_of_F_in_bytes, cudaMemcpyDeviceToHost)
    );
    std::cout << "F_Eigen : \n" << F_Eigen << "\n\n";
    std::cout << "F_CUDA : \n" << F_CUDA << "\n\n";

    


    std::cout << "OK completed \n\n\n\n\n";

    /*

    ALWAYS AT THE END   

    Free the memory
    */

    CUDA_CHECK(
        cudaFree(d_A)
    );
    CUDA_CHECK(
        cudaFree(d_B)
    );
    CUDA_CHECK(
        cudaFree(d_BC)
    );
    CUDA_CHECK(
        cudaFree(d_F)
    );


    /*
    Destry cuda objects
    */
    CUBLAS_CHECK(
        cublasDestroy(cublasH)
    );


    CUDA_CHECK(
        cudaDeviceReset()
    );

    


    std::cout << "Program finished correctly" << std::endl;

    return EXIT_SUCCESS;

}
