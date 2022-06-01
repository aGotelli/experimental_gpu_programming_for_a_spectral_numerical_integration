#include <iostream>
#include <cuda_runtime.h>
#include <cublas.h>
#include <utility>

#include <cusolverDn.h>

#include "cusolver_utils.h"

#include "tictoc.h" //  Mesuring runtime

#include <Eigen/Dense>

#include <fstream>
#include <vector>

#include <iostream>

/*! \file main.cpp
    \brief The main file performing the spectral numerical integration.

    In this file, we perform the computation from the PDF.
*/
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>

#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"
#include "lie_algebra_utilities.h"
#include "spectral_integration_library.h"
#include "odeBase.h"
#include "qIntegrator.h"
#include "rIntegrator.h"
#include "lambdaIntegrator.h"
#include "qadIntegrator.h"


const auto chebyshev_points_top_down = ComputeChebyshevPoints<num_ch_nodes>(TOP_TO_BOTTOM);
const auto chebyshev_points_bottom_up = ComputeChebyshevPoints<num_ch_nodes>(BOTTOM_TO_TOP);
const std::array<std::array<double, num_ch_nodes>, 2> chebyshev_points = {chebyshev_points_bottom_up, chebyshev_points_top_down};

const auto Phi_top_down = Phi<na, ne, num_ch_nodes>(chebyshev_points[TOP_TO_BOTTOM]);
const auto Phi_bottom_up = Phi<na, ne, num_ch_nodes>(chebyshev_points[BOTTOM_TO_TOP]);
const std::array<std::array<Eigen::MatrixXd, num_ch_nodes>, 2> Phi_matrix = {Phi_bottom_up, Phi_top_down};

Eigen::VectorXd getInitLambda(Eigen::Quaterniond t_q) {
    const Eigen::Matrix<double, 6, 6> Ad_at_tip = Ad(t_q.toRotationMatrix(), Eigen::Vector3d::Zero());

    Eigen::Matrix<double, 6, 1> F(0, 0, 0, 0, 0, -1);

    return Ad_at_tip.transpose()*F; //no stresses
}

template <unsigned int t_stateDim>
void initIntegrator(std::shared_ptr<odeBase<t_stateDim, num_ch_nodes>> base,
                    Eigen::VectorXd qe,
                    std::array<std::array<Eigen::MatrixXd, num_ch_nodes>, 2> phi,
                    Eigen::VectorXd x0,
                    Eigen::MatrixXd Q = Eigen::Matrix<double, 4, num_ch_nodes>::Zero(),
                    Eigen::MatrixXd Lambda = Eigen::Matrix<double, 6, num_ch_nodes>::Zero()) {
    base->qe = qe;
    base->Phi_array = phi;
    base->x0 = x0;
    base->Q = Q;
    base->Lambda = Lambda;
    
    base->getA();
    base->getb();
    base->initMemory();
}

int main(int argc, char *argv[]) {
    Eigen::VectorXd qe(9);
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

    //Quaternions
    constexpr int qStateDim = 4;
    const Eigen::Vector4d initQuaternion(1, 0, 0, 0);

    std::shared_ptr<odeBase<qStateDim, num_ch_nodes>> qint_ptr(new qIntegrator<qStateDim, num_ch_nodes>(BOTTOM_TO_TOP));
    initIntegrator<qStateDim>(qint_ptr, qe, Phi_matrix, initQuaternion);

    const auto Q_stack = integrateODE<qStateDim>(qint_ptr);

    //POSITIONS

    constexpr unsigned int rStateDim = 3;
    const Eigen::Vector3d initPos(0, 0, 0); //straight rod
    std::shared_ptr<odeBase<rStateDim, num_ch_nodes>> rint_ptr(new rIntegrator<rStateDim, num_ch_nodes>(BOTTOM_TO_TOP));
    initIntegrator<rStateDim>(rint_ptr, qe, Phi_matrix, initPos, Q_stack);

    // rint_ptr->x0 = initPos;
    // rint_ptr->Q = Q_stack;
    // rint_ptr->getA();
    // rint_ptr->getb();
    // rint_ptr->initMemory();
    const auto r_stack = integrateODE<rStateDim>(rint_ptr);

    //STRESSES

    constexpr unsigned int lambdaStateDim = 6;
    const auto initLambda = getInitLambda(Eigen::Quaterniond(Q_stack.row(0)[0],
                                                             Q_stack.row(0)[1],
                                                             Q_stack.row(0)[2],
                                                             Q_stack.row(0)[3]));
    std::shared_ptr<odeBase<lambdaStateDim, num_ch_nodes>> lambdaint_ptr(new lambdaIntegrator<lambdaStateDim, num_ch_nodes>(TOP_TO_BOTTOM));
    initIntegrator<lambdaStateDim>(lambdaint_ptr, qe, Phi_matrix, initLambda, Q_stack);

    const auto lambda_stack = integrateODE<lambdaStateDim>(lambdaint_ptr);

    //GENERALISED FORCES

    constexpr unsigned int qadStateDim = 9;
    const auto initQad = Eigen::Vector<double, qadStateDim>::Zero();
    std::shared_ptr<odeBase<qadStateDim, num_ch_nodes>> qadint_ptr(new qadIntegrator<qadStateDim, num_ch_nodes>(TOP_TO_BOTTOM));
    initIntegrator<qadStateDim>(qadint_ptr, qe, Phi_matrix, initQad, Q_stack, lambda_stack);

    const auto qad_stack = integrateODE<qadStateDim>(qadint_ptr);

    //PRINT RESULTS

    std::cout << "Q_stack: \n" << Q_stack << "\n" << std::endl;
    std::cout << "r_stack: \n" << r_stack << "\n" << std::endl;
    std::cout << "lambda_stack: \n" << lambda_stack << "\n" << std::endl;
    std::cout << "qad_stack: \n" << qad_stack << "\n" << std::endl;

    return 0;
}
