#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>



#include <Eigen/Dense>


#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"




template<unsigned int t_ny, unsigned int t_N, unsigned int t_ns, unsigned int t_L, unsigned int na, unsigned int ne>
Eigen::MatrixXd getA(Eigen::VectorXd &t_qe)
{
    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<t_N+1>();



    std::vector<Eigen::MatrixXd> A_stack;

    for(unsigned int i=0; i<=t_N+1; i++){
        //  Extract the curvature from the strain
        const Eigen::Vector3d K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        Eigen::MatrixXd A_local(4,4);
        A_local <<   0, -K(0),  -K(1),  -K(2),
                  K(0),     0,   K(2),  -K(1),
                  K(1), -K(2),      0,   K(0),
                  K(2),  K(1),  -K(0),      0;

        A_stack.push_back(0.5*A_local);
    }

    //  Declare the matrix for the system Ax = b
    Eigen::MatrixXd A(t_ns, t_ns);

    //  Define a vector of index from 1 to t_ny
    Eigen::VectorXi idxY(t_ny);
    unsigned int i=0;
    for(auto& index : idxY)
        index = ++i;

    //  Now the index for the first element of each vector
    //  needs to skyp number of chebyshev points
    Eigen::VectorXi idxX0 = idxY * t_N;

    Eigen::VectorXi idxYN = (idxY - Eigen::VectorXi::Ones(t_ny)) * t_N;

    //  Report to C++ indexing
    idxY -= Eigen::VectorXi::Ones(t_ny);
    idxX0 -= Eigen::VectorXi::Ones(t_ny);
    idxYN -= Eigen::VectorXi::Ones(t_ny);


    //  Populate this matrix with all the elements in the right order
    Eigen::VectorXi index(t_ny);
    for(unsigned int in=0; in<t_N; in++){
        Eigen::VectorXi tmp(t_ny);
        tmp.setConstant(t_ny, 1, in+1);
        index = tmp + idxYN;
        A(index, index) = A_stack[in] ;
    }

    return A;
}

void writeToFile(const std::string &t_name, const Eigen::MatrixXd t_matrix, const Eigen::IOFormat &t_format=Eigen::IOFormat())
{
    std::ofstream file(t_name.c_str());
    file << t_matrix.format(t_format);
 }

int main()
{
    //  Define the initial state
    const Eigen::Vector4d Y0(1, 0, 0, 0);

/*  The state dimension and the number of nodes are known. The state dimension will not change
 *  (a quaternion will remain a quaternion, a twist will remain a twist ecc..) and the number of
 *  nodes are not changed at runtime and neither between runs. Typically, once a satisfactory number
 *  of nodes has been found, it is kept for the whole lifetime of the project.
 *
 *  Thus, we can enforce this concept with constexpr classifier so that the values are known at
 *  compile time and we can use templated function for our matrices and vector, which are a bit faster.
 */
    //  Dimension of the state
    constexpr unsigned int state_dimension = Y0.rows();
    //  Number of Chebyshev nodes
    constexpr unsigned int number_of_chebyshev_nodes = 29;
    //  Problem size is the total number of elements
    constexpr unsigned int prob_dimension = state_dimension * number_of_chebyshev_nodes;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    //  Consider unit length rod
    constexpr unsigned int L = 1;

    //  Number od admitted strain fields and number of mode per strain field
    constexpr unsigned int na = 3;
    constexpr unsigned int ne = 3;


    //  Const curvature strain field
    Eigen::VectorXd qe(ne*na);
    //  Here we give some value for the strain
    qe <<   0,
            0,
            0,
            0,
            0,
            0,
         -1.5,
          0.0000,
          0.0000;

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
    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>();
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
    const MatrixNpNp A = getA<state_dimension, number_of_chebyshev_nodes, prob_dimension, L, na, ne>(qe);
    const VectorNp b = Eigen::Matrix<double, prob_dimension, 1>::Zero();


    //  Apply transformation of initial condition onto ODE's matrices
    const MatrixNpNp Ap = P.transpose() * A * P;
    const MatrixNpNp Dp = P.transpose() * D * P;    //  Can be moved in setup
    const VectorNp bp   = P.transpose() * b;




    //  Compute the ivp
    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);    //  Can be moved in setup
    const MatrixNpNs A_IT = Ap.block<prob_dimension, state_dimension>(0, 0);
    const VectorNp ivp = ( D_IT - A_IT ) * Y0;


    //  Obtain the section related to the unknows of the problem
    const MatrixNuNu D_NN = Dp.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const MatrixNuNu A_NN = Ap.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const VectorNu ivp_NN = ivp.block<unknow_state_dimension, 1>(state_dimension, 0);
    const VectorNu b_NN   = bp.block<unknow_state_dimension, 1>(state_dimension, 0);

    //  Finally compute the states at the unknows Chebyshev points
    const VectorNu Yn = (D_NN - A_NN).inverse() * (b_NN - ivp_NN);

    //  We now stack together the initial state on top of the other we just compute
    //  Then we need to map back to a more readable stack of states
    const VectorNp Y = P * (VectorNp() << Y0, Yn).finished();

    //  Then we write the element row-wise
    const MatrixNchebNs Y_stack = Eigen::Map<const MatrixNchebNs>(Y.data());
    std::cout<< "Y_stack = " << std::endl << Y_stack <<std::endl << std::endl;





    return 0;
}
