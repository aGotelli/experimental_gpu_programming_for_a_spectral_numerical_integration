#ifndef SPECTRAL_INTEGRATION_QUATERNION_H
#define SPECTRAL_INTEGRATION_QUATERNION_H

#include <iostream>

#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"
#include <benchmark/benchmark.h>

static constexpr unsigned int number_of_Chebyshev_points = 16;

static constexpr unsigned int state_dimension = 4;

constexpr unsigned int probDimension = state_dimension*number_of_Chebyshev_points;
constexpr unsigned int prob_dimension = state_dimension * number_of_Chebyshev_points;
constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_Chebyshev_points - 1);


static constexpr unsigned int ne = 3;
static constexpr unsigned int na = 3;

static const auto x = ComputeChebyshevPoints<number_of_Chebyshev_points>();


typedef Eigen::Matrix<double, number_of_Chebyshev_points, state_dimension, Eigen::ColMajor> MatrixNchebNs;

typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

typedef Eigen::Matrix<double, number_of_Chebyshev_points, number_of_Chebyshev_points> MatrixNchebNcheb;

typedef Eigen::Matrix<double, number_of_Chebyshev_points, state_dimension, Eigen::ColMajor> MatrixNchebNs;




Eigen::MatrixXd getQuaternionA(const Eigen::VectorXd &t_qe) {



    Eigen::Vector3d K;
    Eigen::Matrix<double, state_dimension, state_dimension> A_at_chebyshev_point;
    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, probDimension, probDimension> A =
            Eigen::Matrix<double, probDimension, probDimension>::Zero();

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


MatrixNchebNs integrateQold(const Eigen::VectorXd &t_qe, const Eigen::Vector4d &t_q_init=Eigen::Vector4d(1, 0, 0, 0))
{

    const MatrixNpNp  P = getP<state_dimension, number_of_Chebyshev_points>();

    const MatrixNchebNcheb Dn = getDn<number_of_Chebyshev_points>();
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);


    MatrixNpNp A;
    const auto b = Eigen::Matrix<double, prob_dimension, 1>::Zero();

    MatrixNpNp Ap;
    const MatrixNpNp Dp = P.transpose() * D * P;
    const VectorNp bp   = P * b;

    const MatrixNpNs D_IT = Dp.block(0, 0, prob_dimension, state_dimension);
    MatrixNpNs A_IT;
    const VectorNp b_IT = ( D_IT - A_IT ) * t_q_init;

    const MatrixNuNu D_NN = Dp.block(state_dimension, state_dimension, unknow_state_dimension, unknow_state_dimension);
    MatrixNuNu A_NN;
    VectorNu ivp;
    const VectorNu b_NN   = bp.block(state_dimension, 0, unknow_state_dimension, 1);

    VectorNu X_NN;

    VectorNp X_tilde;

    MatrixNchebNs X_stack;



    A = getQuaternionA(t_qe);

    Ap = P.transpose() * A * P;

    A_IT = Ap.block(0, 0, prob_dimension, state_dimension);

    A_NN = Ap.block(state_dimension, state_dimension, unknow_state_dimension, unknow_state_dimension);
    ivp = b_IT.block(state_dimension, 0, unknow_state_dimension, 1);


    X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);


    X_tilde = P * (VectorNp() << t_q_init, X_NN).finished();

    X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());


    return X_stack;
}

#endif // SPECTRAL_INTEGRATION_QUATERNION_H
