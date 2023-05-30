#include <cstdio>
#include <cstdlib>
#include <vector>

#include <fstream>
#include <cmath>

//  CUDQ Basic Linear Algebra 
#include <cublas_v2.h>
// // #include <cuda_runtime.h>

#include "spectral_integration_utilities.h"
#include "chebyshev_differentiation.h"
#include "lie_algebra_utilities.h"
#include "utilities.h"  //  Utilities for error handling (must be after cublas or cusolver)
#include "globals.h"
#include "getCusolverErrorString.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>


static const unsigned int number_of_Chebyshev_points = 16;

static const unsigned int quaternion_state_dimension = 4;
static const unsigned int position_dimension = 3;
static const unsigned int quaternion_problem_dimension = quaternion_state_dimension * (number_of_Chebyshev_points-1);

static const unsigned int lambda_dimension = 6;

static const unsigned int Qa_dimension = 9;


// static constexpr unsigned int ne = 3;
// static constexpr unsigned int na = 3;

Eigen::Matrix<double, ne*na, 1> qe;

//  Obtain the Chebyshev differentiation matrix
const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

//FORWARD INTEGRATION:
//  Extract D_NN from the differentiation matrix (for the spectral integration)
const Eigen::MatrixXd Dn_NN_F = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
//  Extract D_IN (for the propagation of initial conditions)
const Eigen::MatrixXd Dn_IN_F = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

//BACKWARD INTEGRATION:
//  Extract D_NN from the differentiation matrix (for the spectral integration)
const Eigen::MatrixXd Dn_NN_B = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);
//  Extract D_IN (for the propagation of initial conditions)
const Eigen::MatrixXd Dn_IN_B = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);



// CUDA specific variables
const auto size_of_double = sizeof(double);

cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;


// Used to build Q_stack
Eigen::MatrixXd computeCMatrix(const Eigen::VectorXd &t_qe, const Eigen::MatrixXd &D_NN)
{

    Eigen::MatrixXd C_NN = D_NN;

    //  Define the Chebyshev points on the unit circle
    const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();


    Eigen::Vector3d K;
    Eigen::MatrixXd Z_at_chebyshev_point(quaternion_state_dimension, quaternion_state_dimension);
    Eigen::MatrixXd A_at_chebyshev_point(quaternion_state_dimension, quaternion_state_dimension);
//    unsigned int left_corner_row;
//    unsigned int left_corner_col;
    for(unsigned int i=0; i<Chebyshev_points.size()-1; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(Chebyshev_points[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        Z_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_at_chebyshev_point = 0.5*Z_at_chebyshev_point;


        for (unsigned int row = 0; row < quaternion_state_dimension; ++row) {
            for (unsigned int col = 0; col < quaternion_state_dimension; ++col) {
                int row_index = row*(number_of_Chebyshev_points-1)+i;
                int col_index = col*(number_of_Chebyshev_points-1)+i;
                C_NN(row_index, col_index) = D_NN(row_index, col_index) - A_at_chebyshev_point(row, col);
            }
        }

    }

    return C_NN;
}

Eigen::VectorXd integrateQuaternions()
{
    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN_F);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN_F);

    Eigen::MatrixXd C_NN =  computeCMatrix(qe, D_NN);

    Eigen::MatrixXd q_init(4,1);
    q_init << 1, 0, 0, 0;

    // Giorgio: START - 
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(quaternion_problem_dimension,1);

    // HOLA! Here I don't know if the .rows() and .cols() is compatible with the VectorXd type. If it's not, change it 
    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;  

    const int rows_q_init = q_init.rows();
    const int cols_q_init = q_init.cols();
    const int ld_q_init = rows_q_init;

    const int rows_b = b.rows();
    const int cols_b = b.cols();
    const int ld_b = rows_b;

    // Here I need the rows and columns for res (they are already defined in the following by exploiting the res variable that I dont have yet!)
    // So, if it works, just delete the second variables declaration. By the way, being res=b-D_IN*q_init it has to have the same rows and columns of b

    const int rows_res = b.rows();
    const int cols_res = b.cols();
    const int ld_res = rows_res;


    // Create Pointers (I commented out the pointer for res in the following. If it's fine you should put all the pointers in the same place for clairty pourpouses)
    double* d_D_IN = nullptr;
    double* d_q_init = nullptr;
    double* d_b = nullptr;
    double* d_res = nullptr;

    // Compute the memory occupation (I commented out the memory occupation for res in the following.)
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_q_init_in_bytes = size_of_double * q_init.size();
    const auto size_of_b_in_bytes = size_of_double * b.size();
    const auto size_of_res_in_bytes = size_of_double * rows_res * cols_res;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_q_init), size_of_q_init_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_b), size_of_b_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes)
    );

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_q_init, q_init.data(), size_of_q_init_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_b, b.data(), size_of_b_in_bytes, cudaMemcpyHostToDevice)
    );


    // Here the idea is this one: 
    // So now we have to compute res = b - ivp with ivp = D_IN*q_init >> res = b - D_IN*q_init >> By using cublasDgemm
    // The problem is that cublasDgemm is recursive, that is C := alpha*op( A )op( B ) + beta C
    // So in order to use it we have to proceed this way: b = -D_IN*q_init + b and then res = b;
    
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;
    // The result of cublasDgemm is stored into the variable d_b
    // -->  b = -D_IN*q_init + b
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_q_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_q_init, ld_q_init, &beta_cublas, d_b, ld_b)
    );


    CUDA_CHECK(
    cudaMemcpy(b.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost)
    );

    //Definition of matrices dimensions.
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_Q_stack = rows_C_NN;
    const int cols_Q_stack = cols_b;
    const int ld_Q_stack = rows_Q_stack;

    int info = 0;
    int lwork = 0;



    // Create Pointers
    double* d_Q_stack = nullptr;
    double* d_C_NN = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    
    // Compute the memory occupation
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int))
    );
    

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice)
    );

    // --> CNN*Qstack = b
    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_Q_stack, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };


    //Has to be after cusolverDnDgetrf_bufferSize as lwork is only computed then.
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork)
    );


    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info)
    );

    // Solving the final system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_b, ld_b, d_info)
    );

    //What we want to calculate
    Eigen::MatrixXd Q_stack_CUDA(rows_Q_stack, cols_Q_stack);

    CUDA_CHECK(
        cudaMemcpy(Q_stack_CUDA.data(), d_b, size_of_b_in_bytes, cudaMemcpyDeviceToHost)
    );



    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_D_IN)
    );
    CUDA_CHECK(
        cudaFree(d_C_NN)
    );
    CUDA_CHECK(
        cudaFree(d_b)
    );
    CUDA_CHECK(
        cudaFree(d_info)
    );
    CUDA_CHECK(
        cudaFree(d_q_init)
    );
    CUDA_CHECK(
        cudaFree(d_Q_stack)
    );
    CUDA_CHECK(
        cudaFree(d_res)
    );
    CUDA_CHECK(
        cudaFree(d_work)
    );


    return Q_stack_CUDA;
}



// Used to build r_stack
Eigen::MatrixXd updatePositionb(Eigen::MatrixXd t_Q_stack) {

    Eigen::Matrix<double, number_of_Chebyshev_points-1, position_dimension> b;

    Eigen::Quaterniond q;

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {


        q = { t_Q_stack(i),
              t_Q_stack(i  +  (number_of_Chebyshev_points-1)),
              t_Q_stack(i + 2*(number_of_Chebyshev_points-1)),
              t_Q_stack(i + 3*(number_of_Chebyshev_points-1)) };


        b.block<1,3>(i, 0) = (q.toRotationMatrix()*Eigen::Vector3d(1, 0, 0)).transpose();

    }
    return b;
}

Eigen::MatrixXd integratePosition()
{   
    Eigen::Vector3d r_init;
    r_init << 0,
              0,
              0;

    //  This matrix remains constant so we can pre invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_F.inverse();

    Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);
    for(unsigned int i=0; i<ivp.rows(); i++)
        ivp.row(i) = Dn_IN_F(i, 0) * r_init.transpose();

    const auto Q_stack_CUDA = integrateQuaternions();
    
    Eigen::MatrixXd b_NN = updatePositionb(Q_stack_CUDA);

    Eigen::MatrixXd res = b_NN - ivp;

    // Define dimensions
    const int rows_Dn_NN_inv = Dn_NN_inv.rows();
    const int cols_Dn_NN_inv = Dn_NN_inv.cols();
    const int ld_Dn_NN_inv = rows_Dn_NN_inv;  

    const int rows_res = res.rows();
    const int cols_res = res.cols();
    const int ld_res = rows_res;

    const int rows_r_stack = rows_Dn_NN_inv;
    const int cols_r_stack = cols_res;
    const int ld_r_stack = rows_r_stack;

    // Create Pointers
    double* d_Dn_NN_inv = nullptr;
    double* d_res = nullptr;
    double* d_r_stack = nullptr;

    
    // Compute the memory occupation
    const auto size_of_Dn_NN_inv_in_bytes = size_of_double * Dn_NN_inv.size();
    const auto size_of_res_in_bytes = size_of_double * res.size();
    const auto size_of_r_stack_in_bytes = size_of_double * rows_r_stack * cols_r_stack;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes)
    );
        CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_r_stack), size_of_r_stack_in_bytes)
    );
    
    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_res, res.data(), size_of_res_in_bytes, cudaMemcpyHostToDevice)
    );

    // Compute r_stack = Dn_NN_inv*res
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_res, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_res, ld_res, &beta_cublas, d_r_stack, ld_r_stack)
    );
    // Variable to check the result
    Eigen::MatrixXd r_stack_CUDA(rows_r_stack, cols_r_stack);

    CUDA_CHECK(
        cudaMemcpy(r_stack_CUDA.data(), d_r_stack, size_of_r_stack_in_bytes, cudaMemcpyDeviceToHost));

    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_Dn_NN_inv)
    );
    CUDA_CHECK(
        cudaFree(d_r_stack)
    );
    CUDA_CHECK(
        cudaFree(d_res)
    );

    return r_stack_CUDA;
}



// Used to build Lambda_stack:
Eigen::MatrixXd updateCMatrix(const Eigen::VectorXd &t_qe, const Eigen::MatrixXd &D_NN)
{

    Eigen::MatrixXd C_NN = D_NN;

    //  Define the Chebyshev points on the unit circle
    const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();

    Eigen::Vector3d K;
    Eigen::MatrixXd A_at_chebyshev_point(lambda_dimension/2, lambda_dimension/2);

    for(unsigned int i=0; i<Chebyshev_points.size()-1; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(Chebyshev_points[i])*t_qe;

        //  Build Skew Symmetric K matrix (K_hat)
        Eigen::Matrix3d K_hat = skew(K);
        A_at_chebyshev_point = K_hat.transpose();

        for (unsigned int row = 0; row < lambda_dimension/2; ++row) {
            for (unsigned int col = 0; col < lambda_dimension/2; ++col) {
                int row_index = row*(number_of_Chebyshev_points-1)+i;
                int col_index = col*(number_of_Chebyshev_points-1)+i;
                C_NN(row_index, col_index) = D_NN(row_index, col_index) - A_at_chebyshev_point(row, col);
            }
        }

    }

    return C_NN;

}

Eigen::MatrixXd integrateInternalForces()
{   
    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B);

    Eigen::MatrixXd C_NN =  updateCMatrix(qe, D_NN);


    Eigen::VectorXd N_init(lambda_dimension/2);
    N_init << 1, 0, 0;

    Eigen::MatrixXd beta((lambda_dimension/2)*(number_of_Chebyshev_points-1) , 1);

    // Variables definition to include gravity (Nbar)
    const double g = 9.81; // m/s^2
    const double radius = 0.001; // m
    const double A = M_PI*radius*radius;
    const double rho = 7800; // kg/m^3

    Eigen::VectorXd Fg(lambda_dimension/2);
    Fg << 0, 0, A*g*rho;
    
    Eigen::VectorXd Q_stack = integrateQuaternions();
    Eigen::Matrix3d R;
    Eigen::VectorXd Nbar(lambda_dimension/2);
    Eigen::VectorXd Nbar_stack((lambda_dimension/2)*(number_of_Chebyshev_points-1));

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

            Eigen::Quaterniond Qbar(Q_stack.block<quaternion_state_dimension,1>(i*quaternion_state_dimension,0));
        
        R = Qbar.toRotationMatrix();
        Nbar = R.transpose()*Fg;

        Nbar_stack.block<lambda_dimension/2,1>(i*lambda_dimension,0) = Nbar;
    }

    std::cout << "Nbar_stack \n" << Nbar_stack << std::endl;

    //Definition of matrices dimensions.
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;

    const int rows_N_init = N_init.rows();
    const int cols_N_init = N_init.cols();
    const int ld_N_init = rows_N_init;

    const int rows_beta = beta.rows();
    const int cols_beta = beta.cols();
    const int ld_beta = rows_beta;

    const int rows_N_stack = rows_beta;
    const int cols_N_stack = cols_beta;
    const int ld_N_stack = rows_N_stack;
    
    int info = 0;
    int lwork = 0;

    // Create Pointers
    double* d_C_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_N_init = nullptr;
    double* d_beta = nullptr;
    double* d_res = nullptr;
    double* d_N_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_N_init_in_bytes = size_of_double * N_init.size();
    const auto size_of_beta_in_bytes = size_of_double * beta.size();
    const auto size_of_N_stack_in_bytes = size_of_double * rows_N_stack * cols_N_stack;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_N_init), size_of_N_init_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_beta), size_of_beta_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_N_stack), size_of_N_stack_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int))
    );

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_N_init, N_init.data(), size_of_N_init_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_beta, beta.data(), size_of_beta_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice)
    );

    // Computation of beta = -D_IN*N_init + beta (attention to this beta that is not the constant factor in the cublasDgemm function) and then res = beta;
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;

    // The result of cublasDgemm is stored into the variable d_beta
    // --> res = -D_IN*N_init + beta
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_N_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_N_init, ld_N_init, &beta_cublas, d_beta, ld_beta)
    );

    CUDA_CHECK(
        cudaMemcpy(beta.data(), d_beta, size_of_beta_in_bytes, cudaMemcpyDeviceToHost)
    );

    // Now, if im not mistken we should have res so let's compute N_stack
    // --> C_NN*N_stack = beta

    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, cols_C_NN, cols_C_NN, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };

    //Has to be after cusolverDnDgetrf_bufferSize as lwork is only computed then.
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork)
    );

    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, cols_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, cols_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta, ld_beta, d_info));

    //What we want to calculate
    Eigen::MatrixXd N_stack_CUDA(rows_N_stack, cols_N_stack);
    CUDA_CHECK(
        cudaMemcpy(N_stack_CUDA.data(), d_beta, size_of_beta_in_bytes, cudaMemcpyDeviceToHost));



        //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_beta)
    );
    CUDA_CHECK(
        cudaFree(d_C_NN)
    );
    CUDA_CHECK(
        cudaFree(d_D_IN)
    );
    CUDA_CHECK(
        cudaFree(d_info)
    );
    CUDA_CHECK(
        cudaFree(d_N_init)
    );
    CUDA_CHECK(
        cudaFree(d_N_stack)
    );
    CUDA_CHECK(
        cudaFree(d_work)
    );


    return N_stack_CUDA;
}

Eigen::MatrixXd updateCouplesb(Eigen::MatrixXd t_N_stack) {

    Eigen::MatrixXd beta((lambda_dimension/2)*(number_of_Chebyshev_points-1), 1); // Dimension: 45x1

    Eigen::VectorXd Gamma(lambda_dimension/2);
    Gamma << 1, 0, 0;

    //  TODO: Update it to work with any possible C_bar
    //  Building C_bar
    const Eigen::Vector3d C_bar = Eigen::Vector3d::Zero();

    Eigen::Vector3d N;

    Eigen::Vector3d b;


    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        N << t_N_stack(i),
             t_N_stack(i  +  (number_of_Chebyshev_points-1)),
             t_N_stack(i + 2*(number_of_Chebyshev_points-1));


        b = skew(Gamma).transpose()*N-C_bar;

        beta(i) = b(0);
        beta(i  +  (number_of_Chebyshev_points-1)) = b(1);
        beta(i + 2*(number_of_Chebyshev_points-1)) = b(2);

    }


    return beta;
}

Eigen::MatrixXd integrateInternalCouples()
{
    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN_B); // Dimension: 45x45
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN_B); // Dimension: 45x3

    Eigen::MatrixXd C_NN =  updateCMatrix(qe, D_NN);


    //  Building the b_NN vector
    const auto N_stack_CUDA = integrateInternalForces();
    
    //beta_NN((lambda_dimension/2)*(number_of_Chebyshev_points-1), 1);
    Eigen::MatrixXd beta_NN = updateCouplesb(N_stack_CUDA);


    Eigen::VectorXd C_init(lambda_dimension/2);
    C_init << 1, 0, 0;


    // Giorgio: START
    // Again, the system to solve is: C_NN*C_stack = res that is in the form Ax=b. I'll do everthing I did before:
    // Before to do that we have to compute res = beta_NN - D_IN*C_init. It can be done with cublasDgemm
    // Eigen::VectorXd ivp = D_IN*C_init;
    // const auto res = beta_NN - D_IN*C_init;
    

    //Definition of matrices dimensions.
    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_D_IN = D_IN.rows();
    const int cols_D_IN = D_IN.cols();
    const int ld_D_IN = rows_D_IN;

    const int rows_C_init = C_init.rows();
    const int cols_C_init = C_init.cols();
    const int ld_C_init = rows_C_init;

    const int rows_beta_NN = beta_NN.rows();
    const int cols_beta_NN = beta_NN.cols();
    const int ld_beta_NN = rows_beta_NN;

    const int rows_C_stack = rows_beta_NN;
    const int cols_C_stack = cols_beta_NN;
    const int ld_C_stack = rows_C_stack;
    
    int info = 0;
    int lwork = 0;


    // Create Pointers
    double* d_C_NN = nullptr;
    double* d_D_IN = nullptr;
    double* d_C_init = nullptr;
    double* d_beta_NN = nullptr;
    double* d_N_stack = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;

    // Compute the memory occupation
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_D_IN_in_bytes = size_of_double * D_IN.size();
    const auto size_of_C_init_in_bytes = size_of_double * C_init.size();
    const auto size_of_beta_NN_in_bytes = size_of_double * beta_NN.size();
    const auto size_of_N_stack_in_bytes = size_of_double * N_stack_CUDA.size();

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_D_IN), size_of_D_IN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_init), size_of_C_init_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_beta_NN), size_of_beta_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_N_stack), size_of_N_stack_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int))
    );

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_D_IN, D_IN.data(), size_of_D_IN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_C_init, C_init.data(), size_of_C_init_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_beta_NN, beta_NN.data(), size_of_beta_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice)
    );

    // Computation of beta_NN = -D_IN*C_init + beta_NN
    double alpha_cublas = -1.0;
    double beta_cublas = 1.0;
    // The result of cublasDgemm is stored into the variable d_beta
    // --> res = -D_IN*C_init + beta_NN
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_D_IN, cols_C_init, cols_D_IN, &alpha_cublas, d_D_IN, ld_D_IN, d_C_init, ld_C_init, &beta_cublas, d_beta_NN, ld_beta_NN)
    );

 // --> TESTED   

    // Now, if im not mistken we should have res so let's compute C_stack
    // --> C_NN*C_stack = beta_NN


    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };

    //Has to be after cusolverDnDgetrf_bufferSize as lwork is only computed then.
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), size_of_double * lwork)
    );

    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info)
    );

    // Solving the final system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_beta_NN, ld_beta_NN, d_info)
    );

    //What we want to calculate
    Eigen::MatrixXd C_stack_CUDA(N_stack_CUDA.rows(), N_stack_CUDA.cols());
    CUDA_CHECK(
        cudaMemcpy(C_stack_CUDA.data(), d_beta_NN, size_of_beta_NN_in_bytes, cudaMemcpyDeviceToHost)
    );


    //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_beta_NN)
    );
    CUDA_CHECK(
        cudaFree(d_C_init)
    );
    CUDA_CHECK(
        cudaFree(d_C_NN)
    );
    CUDA_CHECK(
        cudaFree(d_D_IN)
    );
    CUDA_CHECK(
        cudaFree(d_info)
    );
    CUDA_CHECK(
        cudaFree(d_N_stack)
    );
    CUDA_CHECK(
        cudaFree(d_work)
    );

    return C_stack_CUDA;
}

Eigen::MatrixXd buildLambda(Eigen::MatrixXd t_C_stack, Eigen::MatrixXd t_N_stack)
{
    Eigen::Vector3d C;
    Eigen::Vector3d N;

    Eigen::VectorXd lambda(lambda_dimension);

    Eigen::MatrixXd Lambda_stack(lambda_dimension*(number_of_Chebyshev_points-1), 1);

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        N << t_N_stack(i),
             t_N_stack(i  +  (number_of_Chebyshev_points-1)),
             t_N_stack(i + 2*(number_of_Chebyshev_points-1));

        C << t_C_stack(i),
             t_C_stack(i  +  (number_of_Chebyshev_points-1)),
             t_C_stack(i + 2*(number_of_Chebyshev_points-1));

        lambda << C, N;

        Lambda_stack.block<6,1>(i*lambda_dimension,0) = lambda;
    }

    return Lambda_stack;
}



// Used to build Qa_stack
Eigen::MatrixXd updateQad_vector_b(Eigen::MatrixXd t_Lambda_stack)
{
    //  Define the Chebyshev points on the unit circle
    const auto Chebyshev_points = ComputeChebyshevPoints<number_of_Chebyshev_points>();

    Eigen::MatrixXd B_NN(number_of_Chebyshev_points-1, Qa_dimension);

    Eigen::VectorXd b(Qa_dimension);

    Eigen::MatrixXd B(6, 3);
    B.block(0, 0, 3, 3).setIdentity();
    B.block(3, 0, 3, 3).setZero();

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {

        // NOTE: Lambda_stack is already built without the first cheb. pt. however we need to index the Chebyshev_points[1] as our first cheb. pt (PORCA PUTTANA)
        b =  -Phi<na,ne>(Chebyshev_points[i+1]).transpose()*B.transpose()*t_Lambda_stack.block<lambda_dimension,1>(lambda_dimension*i,0);

        B_NN.block<1,Qa_dimension>(i, 0) = b.transpose();
    }
    return B_NN;
}

Eigen::MatrixXd integrateGeneralisedForces(Eigen::MatrixXd t_Lambda_stack)
{

    Eigen::Vector3d Qa_init;
    Qa_init << 0,
               0,
               0;

    Eigen::MatrixXd B_NN(number_of_Chebyshev_points-1, Qa_dimension);

    // Dn_NN is constant so we can pre-invert
    Eigen::MatrixXd Dn_NN_inv = Dn_NN_B.inverse();

    B_NN = updateQad_vector_b(t_Lambda_stack);

    // Qa_stack = B_NN*Dn_NN_inv
    //Definition of matrices dimensions.
    const int rows_B_NN = B_NN.rows();
    const int cols_B_NN = B_NN.cols();
    const int ld_B_NN = rows_B_NN;

    const int rows_Dn_NN_inv = Dn_NN_inv.rows();
    const int cols_Dn_NN_inv = Dn_NN_inv.cols();
    const int ld_Dn_NN_inv = rows_Dn_NN_inv;


    const int rows_Qa_stack = rows_Dn_NN_inv;
    const int cols_Qa_stack = cols_B_NN;
    const int ld_Qa_stack = rows_Qa_stack;

    // Create Pointers
    double* d_B_NN = nullptr;
    double* d_Dn_NN_inv = nullptr;    
    double* d_Qa_stack = nullptr;

    // Compute the memory occupation
    const auto size_of_B_NN_in_bytes = size_of_double * B_NN.size();
    const auto size_of_Dn_NN_inv_in_bytes = size_of_double * Dn_NN_inv.size();
    const auto size_of_Qa_stack_in_bytes = size_of_double * rows_Qa_stack * cols_Qa_stack;

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B_NN), size_of_B_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Dn_NN_inv), size_of_Dn_NN_inv_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Qa_stack), size_of_Qa_stack_in_bytes)
    );

    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_B_NN, B_NN.data(), size_of_B_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_Dn_NN_inv, Dn_NN_inv.data(), size_of_Dn_NN_inv_in_bytes, cudaMemcpyHostToDevice)
    );

    // Compute Qa_stack = Dn_NN_inv*B_NN
    double alpha_cublas = 1.0;
    double beta_cublas = 0.0;
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rows_Dn_NN_inv, cols_B_NN, cols_Dn_NN_inv, &alpha_cublas, d_Dn_NN_inv, ld_Dn_NN_inv, d_B_NN, ld_B_NN, &beta_cublas, d_Qa_stack, ld_Qa_stack)
    );
    // Variable to check the result
    Eigen::MatrixXd Qa_stack_CUDA(rows_Qa_stack, cols_Qa_stack);

    CUDA_CHECK(
        cudaMemcpy(Qa_stack_CUDA.data(), d_Qa_stack, size_of_Qa_stack_in_bytes, cudaMemcpyDeviceToHost));

        //FREEING MEMORY
    CUDA_CHECK(
        cudaFree(d_B_NN)
    );
    CUDA_CHECK(
        cudaFree(d_Qa_stack)
    );
    CUDA_CHECK(
        cudaFree(d_Dn_NN_inv)
    );

    return Qa_stack_CUDA;
}






int main(int argc, char *argv[])
{
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

    CUSOLVER_CHECK(
        cusolverDnCreate(&cusolverH)
    );


    //  Here we give some value for the strain
//    qe.setZero();
    qe <<   0,
            0,
            0,
            1.2877691307032,
           -1.63807499160786,
            0.437406679142598,
            0,
            0,
            0;

    const auto Q_stack = integrateQuaternions();
    std::cout << "Q_stack : \n" << Q_stack << std::endl;
    
    const auto r_stack = integratePosition();
    std::cout << "r_stack : \n" << r_stack << std::endl;

    const auto N_stack = integrateInternalForces();
    std::cout << "N_stack : \n" << N_stack << "\n" << std::endl;

    const auto C_stack = integrateInternalCouples();
    std::cout << "C_stack : \n" << C_stack << "\n" << std::endl;

    const auto Lambda_stack = buildLambda(C_stack, N_stack);
    std::cout << "Lambda_stack : \n" << Lambda_stack << "\n" << std::endl;

    const auto Qa_stack = integrateGeneralisedForces(Lambda_stack);
    std::cout << "Qa_stack : \n" << Qa_stack << std::endl;

    /*
    Destry cuda objects
    */
    CUBLAS_CHECK(
        cublasDestroy(cublasH)
    );

    CUSOLVER_CHECK(
        cusolverDnDestroy(cusolverH)
    );

    CUDA_CHECK(
        cudaDeviceReset()
    );

    return 0;
}
