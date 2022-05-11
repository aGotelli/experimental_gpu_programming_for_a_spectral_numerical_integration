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


#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"
#include "lie_algebra_utilities.h"
#include "spectral_integration_library.h"

#include "tictoc.h"


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
