/*! \file main.cpp
    \brief The main file performing the spectral numerical integration.

    In this file, we perform the computation from the PDF.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>


#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"
#include "lie_algebra_utilities.h"
#include "spectral_integration_library.h"

#include "tictoc.h"

const auto chebyshev_points_top_down = ComputeChebyshevPoints<number_of_chebyshev_nodes>(TOP_TO_BOTTOM);
const auto chebyshev_points_bottom_up = ComputeChebyshevPoints<number_of_chebyshev_nodes>(BOTTOM_TO_TOP);
const std::array<std::array<double, number_of_chebyshev_nodes>, 2> chebyshev_points = {chebyshev_points_bottom_up, chebyshev_points_top_down};

const auto Phi_top_down = Phi<na, ne, number_of_chebyshev_nodes>(chebyshev_points[TOP_TO_BOTTOM]);
const auto Phi_bottom_up = Phi<na, ne, number_of_chebyshev_nodes>(chebyshev_points[BOTTOM_TO_TOP]);
const std::array<std::array<Eigen::MatrixXd, number_of_chebyshev_nodes>, 2> Phi_matrix = {Phi_bottom_up, Phi_top_down};

//density of iron [kg/m^3]
constexpr double rodDensity =  7874;

//1cm radius
const double rodCrossSec = M_PI * pow(0.01, 2);

//gravitational acceleration
constexpr double g = 9.8067;

template<unsigned int t_stateDimension>
Eigen::MatrixXd getQuaternionA(Eigen::VectorXd &t_qe) {
    constexpr integrationDirection direction = BOTTOM_TO_TOP;

    constexpr unsigned int probDimension = t_stateDimension*number_of_chebyshev_nodes;

    Eigen::Vector3d K;
    Eigen::Matrix<double, t_stateDimension, t_stateDimension> A_at_chebyshev_point;
    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, probDimension, probDimension> A =
            Eigen::Matrix<double, probDimension, probDimension>::Zero();

    for(unsigned int i=0; i < number_of_chebyshev_nodes; i++){

        //  Extract the curvature from the strain
        K = Phi_matrix[direction][i]*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_at_chebyshev_point = 0.5*A_at_chebyshev_point;

        for (unsigned int row = 0; row < A_at_chebyshev_point.rows(); ++row) {
            for (unsigned int col = 0; col < A_at_chebyshev_point.cols(); ++col) {
                int row_index = row*number_of_chebyshev_nodes+i;
                int col_index = col*number_of_chebyshev_nodes+i;
                A(row_index, col_index) = A_at_chebyshev_point(row, col);
            }
        }
    }

    return A;
}

template<unsigned int t_stateDimension>
Eigen::MatrixXd getPositionb(Eigen::MatrixXd t_Q) {
    constexpr unsigned int probDimension = t_stateDimension * number_of_chebyshev_nodes;

    Eigen::Matrix<double, probDimension, 1> b;
    Eigen::Quaterniond quaternion;

    for (unsigned int i = 0; i < number_of_chebyshev_nodes; ++i) {
        auto q = t_Q.row(i);
        quaternion = {q[0], q[1], q[2], q[3]};


        Eigen::Matrix<double, t_stateDimension, 1> b_at_ch_point = quaternion.toRotationMatrix()*Eigen::Vector3d(1, 0, 0);

        for (unsigned int j = 0; j < t_stateDimension; ++j) {
            b(i+j*number_of_chebyshev_nodes, 0) = b_at_ch_point(j);
        }
    }
    return b;
}

Eigen::MatrixXd getStressesA(Eigen::VectorXd &t_qe) {
    constexpr integrationDirection direction = TOP_TO_BOTTOM;
    constexpr unsigned int t_stateDimension = 6;
    constexpr unsigned int probDimension = t_stateDimension * number_of_chebyshev_nodes;

    Eigen::Vector3d K;
    Eigen::Matrix3d K_hat;
    const auto Gamma_hat = getHat(Eigen::Vector3d(1, 0, 0));
    Eigen::Matrix<double, t_stateDimension, t_stateDimension> ad_xi;

    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, probDimension, probDimension> A
            = Eigen::Matrix<double, probDimension, probDimension>::Zero();

    for(unsigned int i=0; i < number_of_chebyshev_nodes; i++){

        //  Extract the curvature from the strain
        K = Phi_matrix[direction][i]*t_qe;
        K_hat = getHat(K);

        ad_xi.block<3, 3>(0, 0) = K_hat;
        ad_xi.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
        ad_xi.block<3, 3>(3, 0) = Gamma_hat;
        ad_xi.block<3, 3>(3, 3) = K_hat;

        for (unsigned int row = 0; row < ad_xi.rows(); ++row) {
            for (unsigned int col = 0; col < ad_xi.cols(); ++col) {
                int row_index = row*number_of_chebyshev_nodes+i;
                int col_index = col*number_of_chebyshev_nodes+i;
                A(row_index, col_index) = ad_xi.transpose()(row, col);
            }
        }
    }

    return A;
}

template <unsigned int t_stateDimension>
Eigen::MatrixXd getStressesb(Eigen::MatrixXd t_Q) {
    constexpr unsigned int probDimension = t_stateDimension * number_of_chebyshev_nodes;

    Eigen::Matrix<double, probDimension, 1> b;
    Eigen::Quaterniond quaternion;

    const double rodSpecWeight = rodDensity*rodCrossSec;
    const double gravForces = rodSpecWeight*g;

    Eigen::Matrix<double, t_stateDimension, 1> F_ext = {0, 0, 0, 0, 0, gravForces};

    //going from the top down for stresses as opposed to bottom up
    for (int i = number_of_chebyshev_nodes - 1; i >= 0; i--) {
        auto q = t_Q.row(i);
        quaternion = {q[0], q[1], q[2], q[3]};

        Eigen::Matrix<double, t_stateDimension, 1> b_at_ch_point = Ad(quaternion.toRotationMatrix(), Eigen::Vector3d::Zero())*F_ext;

        for (unsigned int j = 0; j < t_stateDimension; ++j) {
            b(number_of_chebyshev_nodes-1-i+j*number_of_chebyshev_nodes, 0) = b_at_ch_point(j);
        }
    }

    return b;
}

Eigen::VectorXd getInitialStress(Eigen::Quaterniond t_q) {
    const Eigen::Matrix<double, 6, 6> Ad_at_tip = Ad(t_q.toRotationMatrix(), Eigen::Vector3d::Zero());

    Eigen::Matrix<double, 6, 1> F(0, 0, 0, 0, 0, -1);

    return Ad_at_tip.transpose()*F; //no stresses
}

template<unsigned int t_stateDimension>
Eigen::MatrixXd getQadb(Eigen::MatrixXd t_lambda) {
    constexpr integrationDirection direction = TOP_TO_BOTTOM;
    constexpr unsigned int QadProbDimension = t_stateDimension * number_of_chebyshev_nodes;

    //define B matrix for generalised forces
    Eigen::Matrix<double, 6, na> B;

    B << 1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         0, 0, 0,
         0, 0, 0,
         0, 0, 0;

    Eigen::Matrix<double, QadProbDimension, 1> b;

    for (unsigned int i = 0; i < number_of_chebyshev_nodes; ++i) {
        auto currLambda = t_lambda.row(i);

        Eigen::Matrix<double, t_stateDimension, 1> b_at_ch_point = Phi_matrix[direction][i].transpose()*B.transpose()*currLambda.transpose();

        for (unsigned int j = 0; j < t_stateDimension; ++j) {
            b(i+j*number_of_chebyshev_nodes, 0) = b_at_ch_point(j);
        }
    }

    return b;
}

int main()
{
    //  Const curvature strain field
    Eigen::VectorXd qe(ne*na);
    //  Here we give some value for the strain

    qe <<   0,
            0,
            0,
            1.28776905384098,
           -1.63807577049031,
            0.437404540900837,
            0,
            0,
            0;

    //Quaternion

    constexpr unsigned int qStateDimension = 4;
    constexpr unsigned int problemDimension = qStateDimension * number_of_chebyshev_nodes;
    const auto q_A = getQuaternionA<qStateDimension>(qe);
    const auto q_b = Eigen::Matrix<double, problemDimension, 1>::Zero();
    //  Define the initial state
    const Eigen::Vector4d initial_quaternion(1, 0, 0, 0); // Quaternion case
    const auto Q = integrateODE<qStateDimension>(initial_quaternion, q_A, q_b, BOTTOM_TO_TOP, "Q_stack");
    //Positions
    constexpr integrationDirection positionDirection = BOTTOM_TO_TOP;
    constexpr unsigned int positionStateDimension = 3;
    constexpr unsigned int positionProblemDimension = positionStateDimension * number_of_chebyshev_nodes;
    const auto position_A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();

    const auto position_b = getPositionb<positionStateDimension>(Q);

    const Eigen::Vector3d initial_position(0, 0, 0); // straight rod
    const auto r = integrateODE<positionStateDimension>(initial_position, position_A, position_b, positionDirection, "r_stack");

    //const auto r = integratePositions(initial_position, Q);

    //Stresses
    constexpr integrationDirection lambdaDirection = TOP_TO_BOTTOM;
    constexpr unsigned int lambdaStateDimension = 6;

    const auto stresses_A = getStressesA(qe);

    const auto stresses_b = getStressesb<lambdaStateDimension>(Q);

    const auto initial_stress = getInitialStress(Eigen::Quaterniond(Q.row(0)[0],
                                                                       Q.row(0)[1],
                                                                       Q.row(0)[2],
                                                                       Q.row(0)[3]));
    const auto lambda = integrateODE<lambdaStateDimension>(initial_stress, stresses_A, stresses_b, lambdaDirection, "lambda_stack");

    //General forces
    constexpr unsigned int QadStateDimension = 9;
    constexpr unsigned int QadProbDimension = QadStateDimension*number_of_chebyshev_nodes;
    constexpr integrationDirection forcesDirection = TOP_TO_BOTTOM;

    //  Declare the matrix for the system Ax = b
    const auto forces_A = Eigen::Matrix<double, QadProbDimension, QadProbDimension>::Zero();

    const auto forces_b = getQadb<QadStateDimension>(lambda);

    //refer to equation 2.18
    const Eigen::Matrix<double, 9, 1> initial_gen_forces(0, 0, 0, 0, 0, 0, 0, 0, 0);
    //const auto genForces = integrateGenForces(initial_gen_forces, lambda);
    const auto genForces = integrateODE<QadStateDimension>(initial_gen_forces, forces_A, forces_b, forcesDirection, "Qad_stack");

    std::cout << "Q_stack = \n" << Q << '\n' << std::endl;
    std::cout << "r_stack = \n" << r << '\n' << std::endl;
    std::cout << "Lambda_stack \n" << lambda << '\n' << std::endl;
    std::cout << "Gen_forces_stack \n" << genForces << '\n' << std::endl;

    return 0;

}

static void TestNumericalIntegration(benchmark::State& t_state)
{
    Eigen::VectorXd qe(ne*na);
    //  Here we give some value for the strain

    qe <<   0,
            0,
            0,
            1.28776905384098,
           -1.63807577049031,
            0.437404540900837,
            0,
            0,
            0;

    //Quaternion

    constexpr unsigned int qStateDimension = 4;
    constexpr unsigned int problemDimension = qStateDimension * number_of_chebyshev_nodes;
    const auto q_A = getQuaternionA<qStateDimension>(qe);
    //82185 ns

    const auto q_b = Eigen::Matrix<double, problemDimension, 1>::Zero();
    //  Define the initial state
    const Eigen::Vector4d initial_quaternion(1, 0, 0, 0); // Quaternion case
//    const auto Q = integrateODE<qStateDimension>(initial_quaternion, q_A, q_b, BOTTOM_TO_TOP, "Q_stack");
    //Positions
//    constexpr integrationDirection positionDirection = BOTTOM_TO_TOP;
//    constexpr unsigned int positionStateDimension = 3;
//    constexpr unsigned int positionProblemDimension = positionStateDimension * number_of_chebyshev_nodes;
//    const auto position_A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();

//    const auto position_b = getPositionb<positionStateDimension>(Q);

//    const Eigen::Vector3d initial_position(0, 0, 0); // straight rod
//    const auto r = integrateODE<positionStateDimension>(initial_position, position_A, position_b, positionDirection, "r_stack");

//    //const auto r = integratePositions(initial_position, Q);

//    //Stresses
//    constexpr integrationDirection lambdaDirection = TOP_TO_BOTTOM;
//    constexpr unsigned int lambdaStateDimension = 6;

//    const auto stresses_A = getStressesA(qe);

//    const auto stresses_b = getStressesb<lambdaStateDimension>(Q);

//    const auto initial_stress = getInitialStress(Eigen::Quaterniond(Q.row(0)[0],
//                                                                       Q.row(0)[1],
//                                                                       Q.row(0)[2],
//                                                                       Q.row(0)[3]));
//    const auto lambda = integrateODE<lambdaStateDimension>(initial_stress, stresses_A, stresses_b, lambdaDirection, "lambda_stack");

//    //General forces
//    constexpr unsigned int QadStateDimension = 9;
//    constexpr unsigned int QadProbDimension = QadStateDimension*number_of_chebyshev_nodes;
//    constexpr integrationDirection forcesDirection = TOP_TO_BOTTOM;

//    //  Declare the matrix for the system Ax = b
//    const auto forces_A = Eigen::Matrix<double, QadProbDimension, QadProbDimension>::Zero();

//    const auto forces_b = getQadb<QadStateDimension>(lambda);

//    //refer to equation 2.18
//    const Eigen::Matrix<double, 9, 1> initial_gen_forces(0, 0, 0, 0, 0, 0, 0, 0, 0);
//    //const auto genForces = integrateGenForces(initial_gen_forces, lambda);
//    const auto genForces = integrateODE<QadStateDimension>(initial_gen_forces, forces_A, forces_b, forcesDirection, "Qad_stack");

    while (t_state.KeepRunning()){
//        constexpr unsigned int qStateDimension = 4;
//        constexpr unsigned int problemDimension = qStateDimension * number_of_chebyshev_nodes;
//        const auto q_A = getQuaternionA<qStateDimension>(qe);
//        const auto q_b = Eigen::Matrix<double, problemDimension, 1>::Zero();
//        //  Define the initial state
//        const Eigen::Vector4d initial_quaternion(1, 0, 0, 0); // Quaternion case
        const auto Q = integrateODE<qStateDimension>(initial_quaternion, q_A, q_b, BOTTOM_TO_TOP, "Q_stack");
    }
}
BENCHMARK(TestNumericalIntegration);




//BENCHMARK_MAIN();
