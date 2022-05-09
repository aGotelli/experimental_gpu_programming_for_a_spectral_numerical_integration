#ifndef SPECTRAL_INTEGRATION_LIBRARY_H
#define SPECTRAL_INTEGRATION_LIBRARY_H

#include <Eigen/Dense>
#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"


constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe
constexpr unsigned int number_of_chebyshev_nodes = 11;

/*!
 * \brief writeToFile writes a Eigen matrix into file
 * \param t_name    name of the file
 * \param t_matrix  the Eigen matrix to write into the file
 * \param t_relative_path_from_build the relative path from the build folder to the file location. Default is none so the file is written in the build directory)
 * \param t_format  the specification for writing. (Default in column major allignment, with comma column separator and 8 digits precision)
 */
void writeToFile(std::string t_name,
                 const Eigen::MatrixXd &t_matrix,
                 std::string t_relative_path_from_build="",
                 const Eigen::IOFormat &t_format=Eigen::IOFormat(16, 0, ","));

/*!
 * \brief getA Compute the matrix A for the system x' = Ax + b
 * \param t_qe The current generalized strains coordinates
 * \tparam t_state_dimension The dimension of the state x
 * \tparam t_number_of_chebyshev_nodes The number of Chebyshev nodes (which also account for the first one)
 * \tparam t_na The number of allowed strain coordinates
 * \tparam t_ne The number of modes per strain coordinate
 * \return
 */
template<unsigned int t_state_dimension>
Eigen::MatrixXd getA(const std::array<Eigen::Matrix<double, t_state_dimension, t_state_dimension>, number_of_chebyshev_nodes> t_A_stack)
{

    static constexpr unsigned int problem_dimension = t_state_dimension * number_of_chebyshev_nodes;

    Eigen::Matrix<double, problem_dimension, problem_dimension> A = Eigen::Matrix<double, problem_dimension, problem_dimension>::Zero();

    Eigen::VectorXi block_indexes = Eigen::VectorXi::LinSpaced(t_state_dimension, 0, number_of_chebyshev_nodes*(t_state_dimension-1));

    for(unsigned int chebyshev_point=0; chebyshev_point<number_of_chebyshev_nodes; chebyshev_point++){
        A(block_indexes, block_indexes) = t_A_stack[chebyshev_point] ;
        
        block_indexes += Eigen::VectorXi::Constant(t_state_dimension, 1, 1);
    }

    return A;
}                 

template<unsigned int t_state_dimension>
Eigen::VectorXd getb(const std::array<Eigen::Matrix<double, t_state_dimension, 1>, number_of_chebyshev_nodes> b_stack) {
    Eigen::Matrix<double, t_state_dimension*number_of_chebyshev_nodes, 1> b;

    for (unsigned int j = 0; j < t_state_dimension; ++j) {
        for (unsigned int i = 0; i < number_of_chebyshev_nodes; ++i) {
            b(i+j*number_of_chebyshev_nodes, 0) = b_stack[i][j];
        }
    }

    return b;
}

Eigen::Matrix<double, 3, 3> getHat(const Eigen::Vector3d t_v) {
    Eigen::Matrix<double, 3, 3> hatMatrix;

    hatMatrix << 0, -t_v(2), t_v(1),
                 t_v(2), 0, -t_v(0),
                 -t_v(1), t_v(0), 0;

    return hatMatrix;
}

static const Eigen::MatrixXd integrateQuaternion(const Eigen::Vector4d &t_initial_state, const Eigen::VectorXd &t_qe){
    
    integrationDirection direction = BOTTOM_TO_TOP;
    constexpr unsigned int state_dimension =4;
    constexpr unsigned int prob_dimension = state_dimension * number_of_chebyshev_nodes;
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

    const auto x = ComputeChebyshevPoints<number_of_chebyshev_nodes>(direction);

    Eigen::Vector3d K;
    Eigen::Matrix<double, state_dimension, state_dimension> A_at_chebyshev_point;
    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, prob_dimension, prob_dimension> A = Eigen::Matrix<double, prob_dimension, prob_dimension>::Zero();

    //  Define a vector containing the indexes of the top left corners of the blocks composing the matrix A
    Eigen::VectorXi block_indexes = Eigen::VectorXi::LinSpaced(state_dimension, 0, number_of_chebyshev_nodes*(state_dimension-1));

    for(unsigned int i=0; i < x.size(); i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_at_chebyshev_point = 0.5*A_at_chebyshev_point;

        A(block_indexes, block_indexes) = A_at_chebyshev_point;
        //  Get the next set of indexes for the coefficients of the matrix A at the next chebyshev point
        block_indexes += Eigen::VectorXi::Constant(state_dimension, 1, 1);
    }

    const VectorNp b = Eigen::Matrix<double, prob_dimension, 1>::Zero();

    const MatrixNpNp Ap = P.transpose() * A * P;
    const MatrixNpNp Dp = P.transpose() * D * P;
    const VectorNp bp   = P * b;

    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);
    const MatrixNpNs A_IT = Ap.block<prob_dimension, state_dimension>(0, 0);
    const VectorNp b_IT = ( D_IT - A_IT ) * t_initial_state;

    const MatrixNuNu D_NN = Dp.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const MatrixNuNu A_NN = Ap.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const VectorNu ivp = b_IT.block<unknow_state_dimension, 1>(state_dimension, 0);
    const VectorNu b_NN   = bp.block<unknow_state_dimension, 1>(state_dimension, 0);

    const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    writeToFile("Q_stack", X_stack);

    return X_stack;
}

Eigen::MatrixXd integratePositions(const Eigen::Vector3d &t_initial_state,Eigen::MatrixXd t_Q_stack) {
    integrationDirection direction = BOTTOM_TO_TOP;

    constexpr unsigned int state_dimension = 3;
    constexpr unsigned int prob_dimension = number_of_chebyshev_nodes * state_dimension;

    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    const Eigen::Matrix<double, prob_dimension, prob_dimension> A = Eigen::Matrix<double, prob_dimension, prob_dimension>::Zero();

    Eigen::Matrix<double, prob_dimension, 1> b;
    Eigen::Quaterniond quaternion;

    for (unsigned int i = 0; i < number_of_chebyshev_nodes; ++i) {
        auto q = t_Q_stack.row(i);
        quaternion = {q[0], q[1], q[2], q[3]};


        Eigen::Matrix<double, state_dimension, 1> b_at_ch_point = quaternion.toRotationMatrix()*Eigen::Vector3d(1, 0, 0);

        for (unsigned int j = 0; j < state_dimension; ++j) {
            b(i+j*number_of_chebyshev_nodes, 0) = b_at_ch_point(j);
        }
    }

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

    const MatrixNpNp Ap = P.transpose() * A * P;
    const MatrixNpNp Dp = P.transpose() * D * P;    //  Can be moved in setup
    const VectorNp bp   = P * b;

    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);    //  Can be moved in setup
    const MatrixNpNs A_IT = Ap.block<prob_dimension, state_dimension>(0, 0);
    const VectorNp b_IT = ( D_IT - A_IT ) * t_initial_state;

    const MatrixNuNu D_NN = Dp.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const MatrixNuNu A_NN = Ap.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const VectorNu ivp = b_IT.block<unknow_state_dimension, 1>(state_dimension, 0);
    const VectorNu b_NN   = bp.block<unknow_state_dimension, 1>(state_dimension, 0);

    const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    writeToFile("R_stack", X_stack);

    return X_stack;
}

Eigen::MatrixXd integrateStresses(const Eigen::Matrix<double, 6, 1> t_initial_state, const Eigen::VectorXd &t_qe) {

    integrationDirection direction = TOP_TO_BOTTOM;
    //  Dimension of the state
    constexpr unsigned int state_dimension = 6;
    //  Problem size is the total number of elements
    constexpr unsigned int prob_dimension = state_dimension * number_of_chebyshev_nodes;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<number_of_chebyshev_nodes>(direction);

    Eigen::Vector3d K;
    Eigen::Matrix3d K_hat;
    const auto Gamma_hat = getHat(Eigen::Vector3d(1, 0, 0));
    Eigen::Matrix<double, state_dimension, state_dimension> ad_xi;

    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, prob_dimension, prob_dimension> A = Eigen::Matrix<double, prob_dimension, prob_dimension>::Zero();

    //  Define a vector containing the indexes of the top left corners of the blocks composing the matrix A
    Eigen::VectorXi block_indexes = Eigen::VectorXi::LinSpaced(state_dimension, 0, number_of_chebyshev_nodes*(state_dimension-1));

    for(unsigned int i=0; i<x.size(); i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;
        K_hat = getHat(K);

        ad_xi.block<3, 3>(0, 0) = K_hat;
        ad_xi.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
        ad_xi.block<3, 3>(3, 0) = Gamma_hat;
        ad_xi.block<3, 3>(3, 3) = K_hat;

        A(block_indexes, block_indexes) = ad_xi.transpose();
        //  Get the next set of indexes for the coefficients of the matrix A at the next chebyshev point
        block_indexes += Eigen::VectorXi::Constant(state_dimension, 1, 1);
    }

    const VectorNp b = Eigen::Matrix<double, prob_dimension, 1>::Zero();

    const MatrixNpNp Ap = P.transpose() * A * P;
    const MatrixNpNp Dp = P.transpose() * D * P;    //  Can be moved in setup
    const VectorNp bp   = P * b;


    //  Compute the ivp
    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);    //  Can be moved in setup
    const MatrixNpNs A_IT = Ap.block<prob_dimension, state_dimension>(0, 0);
    const VectorNp b_IT = ( D_IT - A_IT ) * t_initial_state;


    //  Obtain the section related to the unknows of the problem
    const MatrixNuNu D_NN = Dp.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const MatrixNuNu A_NN = Ap.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const VectorNu ivp = b_IT.block<unknow_state_dimension, 1>(state_dimension, 0);
    const VectorNu b_NN   = bp.block<unknow_state_dimension, 1>(state_dimension, 0);

    //  Finally compute the states at the unknows Chebyshev points
    const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

    //  Then we write the element row-wise
    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    writeToFile("Lambda_stack", X_stack);

    return X_stack;
}

Eigen::MatrixXd integrateGenForces(const Eigen::VectorXd &t_initial_state, const Eigen::MatrixXd t_Lambda_stack) {
    integrationDirection direction = TOP_TO_BOTTOM;
    std::array<Eigen::Matrix<double, 9, 1>, number_of_chebyshev_nodes> b_stack;

    const auto x = ComputeChebyshevPoints<number_of_chebyshev_nodes>(direction);

    constexpr unsigned int state_dimension = 9;
    constexpr unsigned int prob_dimension = number_of_chebyshev_nodes * state_dimension;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    //  Declare the matrix for the system Ax = b
    const Eigen::Matrix<double, prob_dimension, prob_dimension> A = Eigen::Matrix<double, prob_dimension, prob_dimension>::Zero();

    //define B matrix for generalised forces
    Eigen::Matrix<double, 6, na> B;

    B << 1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         0, 0, 0,
         0, 0, 0,
         0, 0, 0;

    Eigen::Matrix<double, prob_dimension, 1> b;
    Eigen::Quaterniond quaternion;

    for (unsigned int i = 0; i < number_of_chebyshev_nodes; ++i) {
        auto lambda = t_Lambda_stack.row(i);

        Eigen::Matrix<double, state_dimension, 1> b_at_ch_point = Phi<na, ne>(x[i]).transpose()*B.transpose()*lambda.transpose();

        for (unsigned int j = 0; j < state_dimension; ++j) {
            b(i+j*number_of_chebyshev_nodes, 0) = b_at_ch_point(j);
        }
    }

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

    //  Apply transformation of initial condition onto ODE's matrices
    const MatrixNpNp Ap = P.transpose() * A * P;
    const MatrixNpNp Dp = P.transpose() * D * P;    //  Can be moved in setup
    const VectorNp bp   = P * b;

    //  Compute the ivp
    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);    //  Can be moved in setup
    const MatrixNpNs A_IT = Ap.block<prob_dimension, state_dimension>(0, 0);
    const VectorNp b_IT = ( D_IT - A_IT ) * t_initial_state;

    //  Obtain the section related to the unknows of the problem
    const MatrixNuNu D_NN = Dp.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const MatrixNuNu A_NN = Ap.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const VectorNu ivp = b_IT.block<unknow_state_dimension, 1>(state_dimension, 0);
    const VectorNu b_NN   = bp.block<unknow_state_dimension, 1>(state_dimension, 0);

    //  Finally compute the states at the unknows Chebyshev points
    const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

    //  Then we write the element row-wise
    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    writeToFile("R_stack", X_stack);

    return X_stack;
}

#endif