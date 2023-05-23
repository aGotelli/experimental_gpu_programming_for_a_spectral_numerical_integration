/*! \file spectral_integration_utilities.h
    \brief This file contains some functions needed in the numerical integration

    In this file, we put functions needed to perform the numerical integration as the
    permutation matrix and the Phi matrix.
*/
#ifndef SPECTRAL_INTEGRATION_UTILITIES_H
#define SPECTRAL_INTEGRATION_UTILITIES_H

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include <boost/math/special_functions/chebyshev.hpp>
#include <boost/math/special_functions/legendre.hpp>

static Eigen::Matrix3d skew(const Eigen::Vector3d &t_v) {

    Eigen::Matrix3d v_hat;
    v_hat <<  0   ,  -t_v(2),   t_v(1),
            t_v(2),     0   ,  -t_v(0),
           -t_v(1),   t_v(0),     0   ;

    return v_hat;
}


static Eigen::MatrixXd ad(const Eigen::VectorXd &t_strain){
    //  Decompose the strain
    const Eigen::Vector3d k = t_strain.block<3,1>(0,0);
    const Eigen::Vector3d gamma = t_strain.block<3,1>(3,0);

    Eigen::MatrixXd ad(6,6);
    ad << skew(k)    , Eigen::Matrix3d::Zero(),
          skew(gamma),          skew(k) ;

    return ad;
}




/*!
 * \brief Phi Compute the basis matrix Phi for a given X
 * \param t_X The coordinate in the rod normalized domain. Must be in [0, 1]
 * \tparam t_na The number of allowed strain coordinates
 * \tparam t_ne The number of modes per strain coordinate
 * \return The basis matrix Phi for a given X
 */
template<unsigned int t_na, unsigned int t_ne>
static const Eigen::MatrixXd Phi(const double t_X, const double &t_begin=0, const double &t_end=1)
{

    //  The coordinate must be transposed into the Chebyshev domain [-1, 1];
    double x = ( 2 * t_X - ( t_end + t_begin) ) / ( t_end - t_begin );

    //  Compute the values of the polynomial for every element of the strain field
    Eigen::Matrix<double, t_ne, 1> Phi_i;
    for(unsigned int i=0; i<t_ne; i++)
        Phi_i[i] = boost::math::legendre_p(i, x);


    //  Define the matrix of bases
    Eigen::MatrixXd Phi = Eigen::KroneckerProduct<Eigen::MatrixXd,Eigen::MatrixXd>(Eigen::Matrix<double, t_na, t_na>::Identity(), Phi_i.transpose());


    return Phi;
}

#endif // SPECTRAL_INTEGRATION_UTILITIES_H
