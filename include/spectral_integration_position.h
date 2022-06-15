#ifndef SPECTRAL_INTEGRATION_POSITION_H
#define SPECTRAL_INTEGRATION_POSITION_H

#include "spectral_integration_quaternion.h"



VectorNp getPositionb(const MatrixNchebNs &t_Q_stack) {

    VectorNp b;
    Eigen::Quaterniond quaternion;

    for (unsigned int i = 0; i < number_of_Chebyshev_points; ++i) {
        auto q = t_Q_stack.row(i);
        quaternion = {q[0], q[1], q[2], q[3]};


        Eigen::Matrix<double, 3, 1> b_at_ch_point = quaternion.toRotationMatrix()*Eigen::Vector3d(1, 0, 0);

        for (unsigned int j = 0; j < 3; ++j) {
            b(i+j*number_of_Chebyshev_points, 0) = b_at_ch_point(j);
        }
    }
    return b;
}



void positionOld(const MatrixNchebNs &t_Q_stack)
{

    constexpr unsigned int positionProblemDimension = 3 * number_of_Chebyshev_points;
    const auto position_A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();

    const auto b = getPositionb(t_Q_stack);

    const Eigen::Vector3d initial_position(0, 0, 0);


    constexpr unsigned int prob_dimension = 3 * number_of_Chebyshev_points;
        constexpr unsigned int unknow_state_dimension = 3 * (number_of_Chebyshev_points - 1);

        typedef Eigen::Matrix<double, number_of_Chebyshev_points, 3, Eigen::ColMajor> MatrixNchebNs;

        typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
        typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

        typedef Eigen::Matrix<double, prob_dimension, 3> MatrixNpNs;

        typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
        typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

        typedef Eigen::Matrix<double, number_of_Chebyshev_points, number_of_Chebyshev_points> MatrixNchebNcheb;

        typedef Eigen::Matrix<double, number_of_Chebyshev_points, 3, Eigen::ColMajor> MatrixNchebNs;

        const MatrixNpNp  P = getP<3, number_of_Chebyshev_points>();

        const MatrixNchebNcheb Dn = getDn<number_of_Chebyshev_points>();
        const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(3, 3), Dn);

        const auto A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();

        const MatrixNpNp Ap = P.transpose() * A * P;
        const MatrixNpNp Dp = P.transpose() * D * P;
        const VectorNp bp   = P * b;

    //    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);
        const MatrixNpNs D_IT = Dp.block(0, 0, prob_dimension, 3);
        const MatrixNpNs A_IT = Ap.block(0, 0, prob_dimension, 3);
        const VectorNp b_IT = ( D_IT - A_IT ) * initial_position;

        const MatrixNuNu D_NN = Dp.block(3, 3, unknow_state_dimension, unknow_state_dimension);
        const MatrixNuNu A_NN = Ap.block(3, 3, unknow_state_dimension, unknow_state_dimension);
        const VectorNu ivp = b_IT.block(3, 0, unknow_state_dimension, 1);
        const VectorNu b_NN   = bp.block(3, 0, unknow_state_dimension, 1);

        const VectorNp b_NN_tilde = P * (VectorNp() << Eigen::Vector3d::Zero(), b_NN).finished();

        const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);


        const VectorNp X_tilde = P * (VectorNp() << initial_position, X_NN).finished();

        const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

        std::cout << "X_stack : \n" << X_stack << std::endl;
}


#endif // SPECTRAL_INTEGRATION_POSITION_H
