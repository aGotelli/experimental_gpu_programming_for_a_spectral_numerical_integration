#include <cstdio>
#include <cstdlib>
#include <vector>

#include <fstream>

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
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;


    CUBLAS_CHECK(
        cublasCreate(&cublasH)
    );
    CUSOLVER_CHECK(
        cusolverDnCreate(&cusolverH)
    );
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
    );
    CUSOLVER_CHECK(
        cusolverDnSetStream(cusolverH, stream)
    );


    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);

    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);


    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN);


    Eigen::MatrixXd C_NN =  computeCMatrix(qe, D_NN);

    Eigen::VectorXd q_init(4);
    q_init << 1, 0, 0, 0;


    Eigen::VectorXd ivp = D_IN*q_init;

    const auto b = Eigen::VectorXd::Zero(quaternion_problem_dimension);

    const Eigen::VectorXd res = b - ivp;

    //Definition of matrices dimensions.

    const int rows_C_NN = C_NN.rows();
    const int cols_C_NN = C_NN.cols();
    const int ld_C_NN = rows_C_NN;

    const int rows_res = res.rows();
    const int cols_res = res.cols();
    const int ld_res = rows_res;

    
    const int rows_Q_stack = rows_C_NN;
    const int cols_Q_stack = cols_res;
    const int ld_Q_stack = rows_Q_stack;

    //What we want to calculate
    Eigen::MatrixXd Qstack_CUDA = Eigen::MatrixXd::Zero(rows_Q_stack, cols_Q_stack);

    int info = 0;

    int lwork = 0;


    // Create Pointers
    double* d_Q_stack = nullptr;
    double* d_C_NN = nullptr;
    double* d_res = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    
    // Compute the memory occupation
    const auto size_of_double = sizeof(double);
    const auto size_of_Q_stack_in_bytes = size_of_double * rows_Q_stack * cols_Q_stack;
    const auto size_of_C_NN_in_bytes = size_of_double * C_NN.size();
    const auto size_of_res_in_bytes = size_of_double * res.size();

    // Allocate the memory
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_Q_stack), size_of_Q_stack_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_NN), size_of_C_NN_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_res), size_of_res_in_bytes)
    );
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int))
    );
    


    //  Copy the data: cudaMemcpy(destination, file_to_copy, size_of_the_file, std_cmd)
    CUDA_CHECK(
        cudaMemcpy(d_C_NN, C_NN.data(), size_of_C_NN_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_res, res.data(), size_of_res_in_bytes, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK(
        cudaMemcpy(d_info, &info, sizeof(int), cudaMemcpyHostToDevice)
    );


    // Allocates buffer size for the LU decomposition
    cusolverStatus_t status = cusolverDnDgetrf_bufferSize(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cusolver error: " << getCusolverErrorString(status) << std::endl;
        // Handle or debug the error appropriately
    };



    //Has to be after cusolverDnDgetrf_bufferSize as lwork is only computed then.
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork)
    );


    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, rows_C_NN, cols_C_NN, d_C_NN, ld_C_NN, d_work, NULL, d_info));

    // Solving the final system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_C_NN, 1, d_C_NN, ld_C_NN, NULL, d_res, ld_res, d_info));

    CUDA_CHECK(
        cudaMemcpy(Qstack_CUDA.data(), d_res, size_of_res_in_bytes, cudaMemcpyDeviceToHost));

    // Eigen::VectorXd Q_stack = C_NN.inverse() * res;

    //  move back Q_stack

    return Qstack_CUDA;


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
    const auto Q_stack = integrateQuaternions();
    Eigen::MatrixXd b_NN(number_of_Chebyshev_points-1, position_dimension);


    Eigen::Vector3d r_init;
    r_init << 0,
              0,
              0;


    //  Get the diffetentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //  Extract the submatrix responsible for the spectral integration
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);

    //  This matrix remains constant so we can pre invert
    const auto Dn_NN_inv = Dn_NN.inverse();

    //  Extract the submatrix responsible for propagating the initial conditions
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

    Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);
    for(unsigned int i=0; i<ivp.rows(); i++)
        ivp.row(i) = Dn_IN(i, 0) * r_init.transpose();

    Eigen::MatrixXd r_stack(number_of_Chebyshev_points-1, position_dimension);



    b_NN = updatePositionb(Q_stack);

    r_stack = Dn_NN_inv*(b_NN - ivp);


    return r_stack;
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
    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);

    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);


    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN);

    Eigen::MatrixXd C_NN =  updateCMatrix(qe, D_NN);

    //  Building N_bar
    //const Eigen::Vector3d N_bar = Eigen::Vector3d::Zero();

    Eigen::VectorXd N_init(lambda_dimension/2);
    N_init << 1, 0, 0;

    Eigen::VectorXd ivp = D_IN*N_init;

    //  TODO: Update it to work with any possible N_bar
    const auto beta = Eigen::VectorXd::Zero((lambda_dimension/2)*(number_of_Chebyshev_points-1));

    const auto res = beta - ivp;

    Eigen::VectorXd N_stack = C_NN.inverse() * res;

    return N_stack;
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
    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);

    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(1, 0);


    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_NN); // Dimension: 45x45
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::MatrixXd::Identity(lambda_dimension/2, lambda_dimension/2), Dn_IN); // Dimension: 45x3

    Eigen::MatrixXd C_NN =  updateCMatrix(qe, D_NN);


    //  Building the b_NN vector
    const auto N_stack = integrateInternalForces();
    Eigen::MatrixXd beta_NN((lambda_dimension/2)*(number_of_Chebyshev_points-1), 1);

    beta_NN = updateCouplesb(N_stack);


    Eigen::VectorXd C_init(lambda_dimension/2);
    C_init << 1, 0, 0;

    Eigen::VectorXd ivp = D_IN*C_init;

    const auto res = beta_NN - ivp;

    Eigen::VectorXd C_stack = C_NN.inverse() * res;

    return C_stack;
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

    //  Extract the submatrix responsible for propagating the initial conditions
//    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

//    Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);
//    for(unsigned int i=0; i<ivp.rows(); i++)
//        ivp.row(i) = Dn_IN(i, 0) * r_init.transpose();

    Eigen::MatrixXd B_NN(number_of_Chebyshev_points-1, Qa_dimension);


    //  Get the diffetentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //  Extract the submatrix responsible for the spectral integration
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(1, 1);

    //  This matrix remains constant so we can pre invert
    const auto Dn_NN_inv = Dn_NN.inverse();

    Eigen::MatrixXd Qa_stack(Qa_dimension*(number_of_Chebyshev_points-1), 1);


    B_NN = updateQad_vector_b(t_Lambda_stack);

    Qa_stack = Dn_NN_inv*(B_NN);


    return Qa_stack;
}




int main(int argc, char *argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
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

    // const auto r_stack = integratePosition();
    // std::cout << "r_stack : \n" << r_stack << std::endl;

    // const auto N_stack = integrateInternalForces();
    // std::cout << "N_stack : \n" << N_stack << "\n" << std::endl;

    // const auto C_stack = integrateInternalCouples();
    // std::cout << "C_stack : \n" << C_stack << "\n" << std::endl;

    // const auto Lambda_stack = buildLambda(C_stack, N_stack);
    // std::cout << "Lambda_stack : \n" << Lambda_stack << "\n" << std::endl;

    // const auto Qa_stack = integrateGeneralisedForces(Lambda_stack);
    // std::cout << "Qa_stack : \n" << Qa_stack << std::endl;




    return 0;
}
