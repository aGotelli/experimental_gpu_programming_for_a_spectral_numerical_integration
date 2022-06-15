#include <iostream>
#include <benchmark/benchmark.h>

#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"

#include <Eigen/Dense>


static constexpr unsigned int number_of_Chebyshev_points = 16;

static constexpr unsigned int state_dimension = 4;

static constexpr unsigned int ne = 3;
static constexpr unsigned int na = 3;


//  Quantities needed for the optimized spectral numerical integration
static constexpr unsigned int reduced_problem_dimension = state_dimension * (number_of_Chebyshev_points-1);
typedef Eigen::Matrix<double, reduced_problem_dimension, reduced_problem_dimension> MatrixNrNr;
typedef Eigen::Matrix<double, reduced_problem_dimension, 1> VectorNr;


//  Quantities needed for the classic numerical integration
constexpr unsigned int problem_dimension = state_dimension*number_of_Chebyshev_points;
constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_Chebyshev_points - 1);

typedef Eigen::Matrix<double, number_of_Chebyshev_points, state_dimension, Eigen::ColMajor> MatrixNchebNs;

typedef Eigen::Matrix<double, problem_dimension, problem_dimension> MatrixNpNp;
typedef Eigen::Matrix<double, problem_dimension, 1> VectorNp;

typedef Eigen::Matrix<double, problem_dimension, state_dimension> MatrixNpNs;

typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

typedef Eigen::Matrix<double, number_of_Chebyshev_points, number_of_Chebyshev_points> MatrixNchebNcheb;

typedef Eigen::Matrix<double, number_of_Chebyshev_points, state_dimension, Eigen::ColMajor> MatrixNchebNs;


static const auto x = ComputeChebyshevPoints<number_of_Chebyshev_points>();



Eigen::MatrixXd getQuaternionA(const Eigen::VectorXd &t_qe) {



    Eigen::Vector3d K;
    Eigen::Matrix<double, state_dimension, state_dimension> A_at_chebyshev_point;
    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, problem_dimension, problem_dimension> A =
            Eigen::Matrix<double, problem_dimension, problem_dimension>::Zero();

    for(unsigned int i=0; i < number_of_Chebyshev_points; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_at_chebyshev_point = 0.5*A_at_chebyshev_point;

        for (unsigned int row = 0; row < A_at_chebyshev_point.rows(); ++row) {
            for (unsigned int col = 0; col < A_at_chebyshev_point.cols(); ++col) {
                int row_index = row*number_of_Chebyshev_points+i;
                int col_index = col*number_of_Chebyshev_points+i;
                A(row_index, col_index) = A_at_chebyshev_point(row, col);
            }
        }
    }

    return A;
}

void updateA(const Eigen::Matrix<double, ne*ne, 1> &t_qe, MatrixNrNr &A_NN, const MatrixNrNr &D_NN)
{

    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<number_of_Chebyshev_points>();


    Eigen::Vector3d K;
    Eigen::Matrix<double, state_dimension, state_dimension> A_at_chebyshev_point;
    unsigned int left_corner_row;
    unsigned int left_corner_col;
    for(unsigned int i=0; i<x.size()-1; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;


        for (unsigned int row = 0; row < state_dimension; ++row) {
            for (unsigned int col = 0; col < state_dimension; ++col) {
                int row_index = row*(number_of_Chebyshev_points-1)+i;
                int col_index = col*(number_of_Chebyshev_points-1)+i;
                A_NN(row_index, col_index) = D_NN(row_index, col_index) - 0.5*A_at_chebyshev_point(row, col);
            }
        }

    }

}





Eigen::Matrix<double, reduced_problem_dimension, 1> RevisedNumericalIntegrationQuaternion(const Eigen::Matrix<double, ne*na, 1> &t_qe,
                                                                                          const Eigen::Vector4d t_q_init=Eigen::Vector4d(1, 0, 0, 0))
{
    //  Initialization of all the variables

    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

    const Eigen::MatrixXd D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);
    const MatrixNrNr D_NN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn_NN);

    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn_IN);

    //  Initialization of matrices for the problem
    MatrixNrNr A_NN = D_NN;
    VectorNr b_NN = VectorNr::Zero();


    //  Preallocation of memory
    Eigen::Matrix<double, reduced_problem_dimension, 1> ivp;
    Eigen::Matrix<double, reduced_problem_dimension, 1> Q_stack;


    //  Actual computations to be performed each time
    updateA(t_qe, A_NN, D_NN);
    ivp = D_IN*t_q_init;
    Q_stack = A_NN.inverse() * (b_NN - ivp);

    return Q_stack;

}


MatrixNchebNs OriginalNumericalIntegrationQuaternion(const Eigen::Matrix<double, ne*na, 1> &t_qe,
                                                     const Eigen::Vector4d t_q_init=Eigen::Vector4d(1, 0, 0, 0))
{

    //  Initialization of all the needed quantities
    const MatrixNpNp  P = getP<state_dimension, number_of_Chebyshev_points>();

    const MatrixNchebNcheb Dn = getDn<number_of_Chebyshev_points>();
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);


    MatrixNpNp A;
    const auto b = VectorNp::Zero();

    MatrixNpNp Ap;
    const MatrixNpNp Dp = P.transpose() * D * P;
    const VectorNp bp   = P * b;

    const MatrixNuNu D_NN = Dp.block(state_dimension, state_dimension, unknow_state_dimension, unknow_state_dimension);
    MatrixNuNu A_NN;
    VectorNu ivp;
    const VectorNu b_NN   = bp.block(state_dimension, 0, unknow_state_dimension, 1);

    const MatrixNpNs D_IT = Dp.block(0, 0, problem_dimension, state_dimension);

    //  Allocate all the memory
    MatrixNpNs A_IT;

    VectorNp b_IT;

    VectorNu Q_NN;

    VectorNp Q_tilde;

    MatrixNchebNs Q_stack;



    b_IT = ( D_IT - A_IT ) * t_q_init;

    A = getQuaternionA(t_qe);

    Ap = P.transpose() * A * P;

    A_IT = Ap.block(0, 0, problem_dimension, state_dimension);

    A_NN = Ap.block(state_dimension, state_dimension, unknow_state_dimension, unknow_state_dimension);
    ivp = b_IT.block(state_dimension, 0, unknow_state_dimension, 1);


    Q_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

    Q_tilde = P * (VectorNp() << t_q_init, Q_NN).finished();

    Q_stack = Eigen::Map<const MatrixNchebNs>(Q_tilde.data());

    return Q_stack;

}



int main(int argc, char *argv[])
{
    Eigen::Matrix<double, ne*na, 1> qe;
    //  Here we give some value for the strain
    qe <<   0,
            0,
            0,
            1.2877691307032,
           -1.63807499160786,
            0.437406679142598,
            0,
            0,
            0;

    std::cout << "Q stack old version : \n" << OriginalNumericalIntegrationQuaternion(qe)  << std::endl;
    std::cout << std::endl << std::endl << std::endl << std::endl << std::endl;
    std::cout << "Q stack new version : \n" << RevisedNumericalIntegrationQuaternion(qe) << std::endl;



    return 0;
}

