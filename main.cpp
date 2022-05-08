/*! \file main.cpp
    \brief The main file performing the spectral numerical integration.

    In this file, we perform the computation from the PDF.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <cmath>



#include <Eigen/Dense>


#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"
#include "lie_algebra_utilities.h"

#include "tictoc.h"

//  Number od admitted strain fields and number of mode per strain field
constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe

//  Number of Chebyshev nodes
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

Eigen::Matrix<double, 3, 3> getHat(const Eigen::Vector3d t_v) {
    Eigen::Matrix<double, 3, 3> hatMatrix;

    hatMatrix << 0, -t_v(2), t_v(1),
                 t_v(2), 0, -t_v(0),
                 -t_v(1), t_v(0), 0;

    return hatMatrix;
}

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
    //  Definition of the problem dymension
    static constexpr unsigned int problem_dimension = t_state_dimension * number_of_chebyshev_nodes;

    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, problem_dimension, problem_dimension> A = Eigen::Matrix<double, problem_dimension, problem_dimension>::Zero();

    //  Define a vector containing the indexes of the top left corners of the blocks composing the matrix A
    Eigen::VectorXi block_indexes = Eigen::VectorXi::LinSpaced(t_state_dimension, 0, number_of_chebyshev_nodes*(t_state_dimension-1));

    //  Populate this matrix with all the elements in the right order

    for(unsigned int chebyshev_point=0; chebyshev_point<number_of_chebyshev_nodes; chebyshev_point++){
        A(block_indexes, block_indexes) = t_A_stack[chebyshev_point] ;
        //  Get the current set of indexes for the coefficients of the matrix A at the current chebyshev point
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

Eigen::MatrixXd integrateQuaternion(const Eigen::Vector4d &t_initial_state, const Eigen::VectorXd &t_qe) {

/*  The state dimension and the number of nodes are known. The state dimension will not change
 *  (a quaternion will remain a quaternion, a twist will remain a twist ecc..) and the number of
 *  nodes are not changed at runtime and neither between runs. Typically, once a satisfactory number
 *  of nodes has been found, it is kept for the whole lifetime of the project.
 *
 *  Thus, we can enforce this concept with constexpr classifier so that the values are known at
 *  compile time and we can use templated function for our matrices and vector, which are a bit faster.
 */
    integrationDirection direction = BOTTOM_TO_TOP;
    //  Dimension of the state
    constexpr unsigned int state_dimension = 4;
    //  Problem size is the total number of elements
    constexpr unsigned int prob_dimension = state_dimension * number_of_chebyshev_nodes;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

/*  These are a set of type definition in order to have a more neat algorithm in
 *  terms of matrix and vector dymensions
 */
    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

/*  In this first part, we precompute the matrices P and Dn. In fact, once we know the state dimension and the
 *  number of nodes, these matrices are constant and thus me can compute them beforehand
 *
 *  P.S. As we know the number of nodes, the matrix Phi can be computed at every chebyshev point to be stacked
 *  into a vector. As a result, instead of computing the matrices, we directly access their value in the vector position.
 *  This can actually be something more than 200 times faster.
 *
 */

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

/*  In this part we compute the matrix A and the vector b.
 *  In this case, the elements of the matrix A depends on the strain. As we change the strains during
 *  simulation, we have to compute the components of A at runtime.
 *
 *  Similarly, the vector b contains the values not related to the derivating variable, but that depends on other parameters.
 *  For example when computing r' = R(Q)*Γ it does not depend on r but only on Q and Γ.
 *
 *  We can interpret everything before this point as the setup and everything after as the actual run-time operations.
 *
 *  In the following there are some matrices and operation that could be moved in the setup. However I choose to left them
 *  there for clarity, but feel free to move where you thing it's better.
 *
 *  The following is the translation into C++ of the equations presented in the PDF
 *
 */
    //tictoc.tic();
    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<number_of_chebyshev_nodes>(direction);

    std::array<Eigen::Matrix<double, state_dimension, state_dimension>, number_of_chebyshev_nodes> A_stack;

    Eigen::Vector3d K;
    Eigen::Matrix<double, state_dimension, state_dimension> A_at_chebyshev_point;
    for(unsigned int i=0; i < x.size(); i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_stack[i] = 0.5*A_at_chebyshev_point;
    }

    const MatrixNpNp A = getA<state_dimension>(A_stack);
    //tictoc.toc("Time to compute A : ");


    const VectorNp b = Eigen::Matrix<double, prob_dimension, 1>::Zero();


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

    //  We now stack together the initial state on top of the other we just compute
    //  Then we need to map back to a more readable stack of states
    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

    //  Then we write the element row-wise
    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    writeToFile("Q_stack", X_stack);

    return X_stack;
}

Eigen::MatrixXd integratePositions(const Eigen::Vector3d &t_initial_state,Eigen::MatrixXd t_Q_stack) {
    integrationDirection direction = BOTTOM_TO_TOP;

    std::array<Eigen::Matrix<double, 3, 1>, number_of_chebyshev_nodes> b_stack;
    Eigen::Quaterniond quaternion;

    for (unsigned int i = 0; i < number_of_chebyshev_nodes; ++i) {
        auto q = t_Q_stack.row(i);
        quaternion = {q[0], q[1], q[2], q[3]};


        b_stack[i] = quaternion.toRotationMatrix()*Eigen::Vector3d(1, 0, 0);
    }

    //dimension = number of chebyshev nodes * 3 because it's a rotation matrix
    constexpr unsigned int state_dimension = 3;
    constexpr unsigned int prob_dimension = number_of_chebyshev_nodes * state_dimension;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    /*  These are a set of type definition in order to have a more neat algorithm in
     *  terms of matrix and vector dymensions
     */
    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    //  Declare the matrix for the system Ax = b
    const Eigen::Matrix<double, prob_dimension, prob_dimension> A = Eigen::Matrix<double, prob_dimension, prob_dimension>::Zero();

    VectorNp b = getb<state_dimension>(b_stack);

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

/*  In this part we compute the matrix A and the vector b.
 *  In this case, the elements of the matrix A depends on the strain. As we change the strains during
 *  simulation, we have to compute the components of A at runtime.
 *
 *  Similarly, the vector b contains the values not related to the derivating variable, but that depends on other parameters.
 *  For example when computing r' = R(Q)*Γ it does not depend on r but only on Q and Γ.
 *
 *  We can interpret everything before this point as the setup and everything after as the actual run-time operations.
 *
 *  In the following there are some matrices and operation that could be moved in the setup. However I choose to left them
 *  there for clarity, but feel free to move where you thing it's better.
 *
 *  The following is the translation into C++ of the equations presented in the PDF
 *
 */


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

    //  We now stack together the initial state on top of the other we just compute
    //  Then we need to map back to a more readable stack of states
    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

    //  Then we write the element row-wise
    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    writeToFile("R_stack", X_stack);

    return X_stack;
}

Eigen::MatrixXd integrateStresses(const Eigen::Matrix<double, 6, 1> t_initial_state, const Eigen::VectorXd &t_qe) {
/*  The state dimension and the number of nodes are known. The state dimension will not change
 *  (a quaternion will remain a quaternion, a twist will remain a twist ecc..) and the number of
 *  nodes are not changed at runtime and neither between runs. Typically, once a satisfactory number
 *  of nodes has been found, it is kept for the whole lifetime of the project.
 *
 *  Thus, we can enforce this concept with constexpr classifier so that the values are known at
 *  compile time and we can use templated function for our matrices and vector, which are a bit faster.
 */
    integrationDirection direction = TOP_TO_BOTTOM;
    //  Dimension of the state
    constexpr unsigned int state_dimension = 6;
    //  Problem size is the total number of elements
    constexpr unsigned int prob_dimension = state_dimension * number_of_chebyshev_nodes;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

/*  These are a set of type definition in order to have a more neat algorithm in
 *  terms of matrix and vector dymensions
 */
    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

/*  In this first part, we precompute the matrices P and Dn. In fact, once we know the state dimension and the
 *  number of nodes, these matrices are constant and thus me can compute them beforehand
 *
 *  P.S. As we know the number of nodes, the matrix Phi can be computed at every chebyshev point to be stacked
 *  into a vector. As a result, instead of computing the matrices, we directly access their value in the vector position.
 *  This can actually be something more than 200 times faster.
 *
 */

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

/*  In this part we compute the matrix A and the vector b.
 *  In this case, the elements of the matrix A depends on the strain. As we change the strains during
 *  simulation, we have to compute the components of A at runtime.
 *
 *  Similarly, the vector b contains the values not related to the derivating variable, but that depends on other parameters.
 *  For example when computing r' = R(Q)*Γ it does not depend on r but only on Q and Γ.
 *
 *  We can interpret everything before this point as the setup and everything after as the actual run-time operations.
 *
 *  In the following there are some matrices and operation that could be moved in the setup. However I choose to left them
 *  there for clarity, but feel free to move where you thing it's better.
 *
 *  The following is the translation into C++ of the equations presented in the PDF
 *
 */
    //tictoc.tic();
    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<number_of_chebyshev_nodes>(direction);

    std::array<Eigen::Matrix<double, state_dimension, state_dimension>, number_of_chebyshev_nodes> A_stack;

    Eigen::Vector3d K;
    Eigen::Matrix3d K_hat;
    const auto Gamma_hat = getHat(Eigen::Vector3d(1, 0, 0));
    Eigen::Matrix<double, state_dimension, state_dimension> ad_xi;

    for(unsigned int i=0; i<x.size(); i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;
        K_hat = getHat(K);

        ad_xi.block<3, 3>(0, 0) = K_hat;
        ad_xi.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
        ad_xi.block<3, 3>(3, 0) = Gamma_hat;
        ad_xi.block<3, 3>(3, 3) = K_hat;

        A_stack[i] = ad_xi.transpose();
    }

    //const MatrixNpNp A = getA<state_dimension, number_of_chebyshev_nodes, na, ne>(t_qe);
    //tictoc.toc("Time to compute A : ");
    const MatrixNpNp A = getA<state_dimension>(A_stack);


    const VectorNp b = Eigen::Matrix<double, prob_dimension, 1>::Zero();


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

    //  We now stack together the initial state on top of the other we just compute
    //  Then we need to map back to a more readable stack of states
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

    //define B matrix for generalised forces
    Eigen::Matrix<double, 6, na> B;

    B << 1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         0, 0, 0,
         0, 0, 0,
         0, 0, 0;

    for (unsigned int i = 0; i < number_of_chebyshev_nodes; ++i) {
        auto lambda = t_Lambda_stack.row(i);

        b_stack[i] = Phi<na, ne>(x[i]).transpose()*B.transpose()*lambda.transpose();
    }

    constexpr unsigned int state_dimension = 9;
    constexpr unsigned int prob_dimension = number_of_chebyshev_nodes * state_dimension;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    /*  These are a set of type definition in order to have a more neat algorithm in
     *  terms of matrix and vector dymensions
     */
    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNchebNs;

    //  Declare the matrix for the system Ax = b
    const Eigen::Matrix<double, prob_dimension, prob_dimension> A = Eigen::Matrix<double, prob_dimension, prob_dimension>::Zero();

    VectorNp b = getb<state_dimension>(b_stack);

    const MatrixNpNp  P = getP<state_dimension, number_of_chebyshev_nodes>();

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(direction);
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

/*  In this part we compute the matrix A and the vector b.
 *  In this case, the elements of the matrix A depends on the strain. As we change the strains during
 *  simulation, we have to compute the components of A at runtime.
 *
 *  Similarly, the vector b contains the values not related to the derivating variable, but that depends on other parameters.
 *  For example when computing r' = R(Q)*Γ it does not depend on r but only on Q and Γ.
 *
 *  We can interpret everything before this point as the setup and everything after as the actual run-time operations.
 *
 *  In the following there are some matrices and operation that could be moved in the setup. However I choose to left them
 *  there for clarity, but feel free to move where you thing it's better.
 *
 *  The following is the translation into C++ of the equations presented in the PDF
 *
 */


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

    //  We now stack together the initial state on top of the other we just compute
    //  Then we need to map back to a more readable stack of states
    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();

    //  Then we write the element row-wise
    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    writeToFile("R_stack", X_stack);

    return X_stack;
}

int main()
{
    tictoc tictoc;

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

    //  Define the initial state
    const Eigen::Vector4d initial_quaternion(1, 0, 0, 0); // Quaternion case
    const auto Q = integrateQuaternion(initial_quaternion, qe);

    const Eigen::Vector3d initial_position(0, 0, 0); // straight rod
    const auto r = integratePositions(initial_position, Q);


    const Eigen::Quaterniond q_at_tip= {Q.row(0)[0],
                                        Q.row(0)[1],
                                        Q.row(0)[2],
                                        Q.row(0)[3]};
    const Eigen::Matrix<double, 6, 6> Ad_at_tip = Ad(q_at_tip.toRotationMatrix(), Eigen::Vector3d::Zero());
    Eigen::Matrix<double, 6, 1> F(0, 0, 0, 0, 0, -1);

    const Eigen::Matrix<double, 6, 1> initial_stress = Ad_at_tip.transpose()*F; //no stresses
    std::cout << "Initial stress: \n" << initial_stress.transpose() << std::endl;
    const auto lambda = integrateStresses(initial_stress, qe);

    /* TODO:
     * - put definition of B inside integrateGenForces function
     * - we can calculate the initial forces inside integrateGenForces function
     * - define stack of Phi<na, ne>(chebychev_point) globally in order to avoid calculating it over an dover
     */

    //refer to equation 2.18
    const Eigen::Matrix<double, 9, 1> initial_gen_forces(0, 0, 0, 0, 0, 0, 0, 0, 0);
    const auto genForces = integrateGenForces(initial_gen_forces, lambda);

    std::cout << "Q_stack = \n" << Q << '\n' << std::endl;
    std::cout << "r_stack = \n" << r << '\n' << std::endl;
    std::cout << "Lambda_stack \n" << lambda << '\n' << std::endl;
    std::cout << "Gen_forces_stack \n" << genForces << '\n' << std::endl;

    return 0;


    /*
     * Workflow
     *
     * 1) Read through, test some values qe Cheb points
     *
     * 2) Move the computations in main into a function IntegrateQuaternion (it returns quaternion stack)
     *
     * 3) Extend to positions (stack of r)
     *
     */


}












void writeToFile(std::string t_name,
                 const Eigen::MatrixXd &t_matrix,
                 std::string t_relative_path_from_build,
                 const Eigen::IOFormat &t_format)
{
    if(not t_relative_path_from_build.empty()){
        //  Ensure relative path ends with a backslash only if a path is given
        if(not t_relative_path_from_build.ends_with('/'))
            t_relative_path_from_build.append("/");
    }


    //  Ensure it ends with .csv
    if(t_name.find(".csv") == std::string::npos)
        t_name.append(".csv");

    //  The file will be created in the location given by the realtive path and with the given name
    const auto file_name_and_location = t_relative_path_from_build + t_name;

    //  Create file in given location with given name
    std::ofstream file(file_name_and_location.c_str());

    //  Put matrix in this file
    file << t_matrix.format(t_format);

    //  Close the file
    file.close();
 }
