/*! \file spectral_integration_utilities.h
    \brief This file contains some functions needed in the numerical integration

    In this file, we put functions needed to perform the numerical integration as the
    permutation matrix and the Phi matrix.
*/
#ifndef SPECTRAL_INTEGRATION_UTILITIES_H
#define SPECTRAL_INTEGRATION_UTILITIES_H

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include <boost/math/special_functions/legendre.hpp>
#include <array>
#include "chebyshev_differentiation.h"

template<unsigned int t_numRows, unsigned int t_numCols>
Eigen::MatrixXd kroneckerProduct(const Eigen::MatrixXd A, const Eigen::MatrixXd B) {
    const unsigned int Ar = A.rows(),
                       Ac = A.cols(),
                       Br = B.rows(),
                       Bc = B.cols();

    Eigen::Matrix<double, t_numRows, t_numCols> AB;
    
    for (unsigned int i=0; i < Ar; ++i) {
        for (unsigned int j=0; j < Ac; ++j) {
            AB.block(i*Br,j*Bc,Br,Bc) = A(i,j)*B;
        }
    }
    
    return AB;
}

/*!
 * \brief Phi Compute the basis matrix Phi for a given X
 * \param t_X The coordinate in the rod normalized domain. Must be in [0, 1]
 * \tparam t_na The number of allowed strain coordinates
 * \tparam t_ne The number of modes per strain coordinate
 * \return The basis matrix Phi for a given X
 */
// template<unsigned int t_na, unsigned int t_ne, unsigned int numNodes>
// static const std::array<Eigen::MatrixXd, numNodes> Phi(const std::array<double, numNodes> t_X, const double &t_begin=0, const double &t_end=1)
// {
//     std::array<Eigen::MatrixXd, numNodes> Phi;

//     for (unsigned int i = 0; i < numNodes; ++i) {
//         //  The coordinate must be transposed into the Chebyshev domain [-1, 1];
//         double x = ( 2 * t_X[i] - ( t_end + t_begin) ) / ( t_end - t_begin );

//         //  Compute the values of the polynomial for every element of the strain field
//         Eigen::Matrix<double, t_ne, 1> Phi_i;
//         for(unsigned int i=0; i<t_ne; i++)
//             Phi_i[i] = boost::math::legendre_p(i, x);


//         //  Define the matrix of bases
//         Phi[i] = kroneckerProduct<t_na, t_na*t_ne>(Eigen::Matrix<double, t_na, t_na>::Identity(), Phi_i.transpose());
//     }
    
//     return Phi;
// }

template<unsigned int t_na, unsigned int t_ne, unsigned int numNodes>
static const std::array<double, numNodes*t_ne> Phi(const std::array<double, numNodes> t_X, const double &t_begin=0, const double &t_end=1)
{
    std::array<double, numNodes*t_ne> Phi;

    for (unsigned int i = 0; i < numNodes; ++i) {
        //  The coordinate must be transposed into the Chebyshev domain [-1, 1];
        double x = ( 2 * t_X[i] - ( t_end + t_begin) ) / ( t_end - t_begin );

        //  Compute the values of the polynomial for every element of the strain field
        for(unsigned int j=0; j<t_ne; j++)
            Phi[i*t_na+j] = boost::math::legendre_p(j, x);
    }
    
    return Phi;
}


/*!
 * \brief getP Compute the Projection matrix
 * \tparam t_state_dimension Indicate the dymension of the state
 * \tparam t_number_of_chebyshev_nodes Indicate the number of Chebyshev points
 * \return the Projection matrix
 */
template<unsigned int t_state_dimension, unsigned int t_number_of_chebyshev_nodes>
static Eigen::MatrixXd getP()
{

    //  Define a vector of index from 1 to t_ny
    Eigen::VectorXi idxY(t_state_dimension);
    unsigned int i=0;
    for(auto& index : idxY)
        index = ++i;

    //  Now the index for the first element of each vector
    //  needs to skyp number of chebyshev points
    Eigen::VectorXi idxX0 = idxY * t_number_of_chebyshev_nodes;

    //  Report them back to C++ correct indexing
    idxY -= Eigen::VectorXi::Ones(t_state_dimension);
    idxX0 -= Eigen::VectorXi::Ones(t_state_dimension);

    const unsigned int problem_dimension = t_state_dimension * t_number_of_chebyshev_nodes;


    //  Initialization as Identity matrix
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(problem_dimension, problem_dimension);

    //  Start applying mapping principles
    P(idxY,idxY) << Eigen::MatrixXd::Zero(idxY.rows(),
                                         idxY.rows());
    P(idxX0,idxX0) = Eigen::MatrixXd::Zero(idxX0.rows(),
                                           idxX0.rows());
    P(idxY,idxX0) = Eigen::MatrixXd::Identity(t_state_dimension, t_state_dimension);
    P(idxX0,idxY) = Eigen::MatrixXd::Identity(t_state_dimension, t_state_dimension);

    return P;
}





#endif // SPECTRAL_INTEGRATION_UTILITIES_H
