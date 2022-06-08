#ifndef QUATERNIONINTEGRATOR_H
#define QUATERNIONINTEGRATOR_H

#include <iostream>

#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"

template<unsigned int t_Chebyshev_points_number>
class QuaternionIntegrator
{
public:
    QuaternionIntegrator(const int t_a);



private:

    static constexpr unsigned int m_state_dimension { 2 };

    Eigen::MatrixXd m_D_NN;
    Eigen::MatrixXd m_D_IN;

};

#endif // QUATERNIONINTEGRATOR_H
