#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>



#include <Eigen/Dense>


#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"

#include "tictoc.h"



/*!
 * \brief getA Compute the matrix A for the system x' = Ax + b
 * \param t_qe The current generalized strains coordinates
 * \tparam t_state_dimension The dimension of the state x
 * \tparam t_number_of_chebyshev_nodes The number of Chebyshev nodes (which also account for the first one)
 * \tparam t_na The number of allowed strain coordinates
 * \tparam t_ne The number of modes per strain coordinate
 * \return
 */
template<unsigned int t_state_dimension, unsigned int t_number_of_chebyshev_nodes, unsigned int t_na, unsigned int t_ne>
Eigen::MatrixXd getA(Eigen::VectorXd &t_qe)
{
    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<t_number_of_chebyshev_nodes+1>();



    std::vector<Eigen::MatrixXd> A_stack;

    for(unsigned int i=0; i<=t_number_of_chebyshev_nodes+1; i++){
        //  Extract the curvature from the strain
        const Eigen::Vector3d K = Phi<t_na, t_ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        Eigen::MatrixXd A_local(4,4);
        A_local <<   0, -K(0),  -K(1),  -K(2),
                  K(0),     0,   K(2),  -K(1),
                  K(1), -K(2),      0,   K(0),
                  K(2),  K(1),  -K(0),      0;

        A_stack.push_back(0.5*A_local);
    }


    //  Define a vector of index from 1 to t_ny
    Eigen::VectorXi idxY(t_state_dimension);
    unsigned int i=0;
    for(auto& index : idxY)
        index = ++i;

    //  Now the index for the first element of each vector
    //  needs to skyp number of chebyshev points
    Eigen::VectorXi idxX0 = idxY * t_number_of_chebyshev_nodes;

    Eigen::VectorXi idxYN = (idxY - Eigen::VectorXi::Ones(t_state_dimension)) * t_number_of_chebyshev_nodes;

    //  Report to C++ indexing
    idxY -= Eigen::VectorXi::Ones(t_state_dimension);
    idxX0 -= Eigen::VectorXi::Ones(t_state_dimension);
    idxYN -= Eigen::VectorXi::Ones(t_state_dimension);

    const unsigned int problem_dimension = t_state_dimension * t_number_of_chebyshev_nodes;

    //  Declare the matrix for the system Ax = b
    Eigen::MatrixXd A(problem_dimension, problem_dimension);


    //  Populate this matrix with all the elements in the right order
    Eigen::VectorXi index(t_state_dimension);
    for(unsigned int in=0; in<t_number_of_chebyshev_nodes; in++){
        Eigen::VectorXi tmp(t_state_dimension);
        tmp.setConstant(t_state_dimension, 1, in+1);
        index = tmp + idxYN;
        A(index, index) = A_stack[in] ;
    }

    return A;
}


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
                 const Eigen::IOFormat &t_format=Eigen::IOFormat(8, 0, ","))
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

int main()
{
    tictoc tictoc;

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
    const MatrixNpNp A = getA<state_dimension, number_of_chebyshev_nodes, na, ne>(qe);
    const VectorNp b = Eigen::Matrix<double, prob_dimension, 1>::Zero();


    //  Apply transformation of initial condition onto ODE's matrices
    tictoc.tic();
    const MatrixNpNp Ap = P.transpose() * A * P;
    tictoc.toc("Time to compute P' A P : ");
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
