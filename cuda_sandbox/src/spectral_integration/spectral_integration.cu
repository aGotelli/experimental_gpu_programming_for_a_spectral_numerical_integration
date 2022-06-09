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

#include <benchmark/benchmark.h>


const auto chebyshev_points_top_down = ComputeChebyshevPoints<num_ch_nodes>(TOP_TO_BOTTOM);
const auto chebyshev_points_bottom_up = ComputeChebyshevPoints<num_ch_nodes>(BOTTOM_TO_TOP);
const std::array<std::array<double, num_ch_nodes>, 2> chebyshev_points = {chebyshev_points_bottom_up, chebyshev_points_top_down};

const auto Phi_top_down = Phi<na, ne, num_ch_nodes>(chebyshev_points[TOP_TO_BOTTOM]);
const auto Phi_bottom_up = Phi<na, ne, num_ch_nodes>(chebyshev_points[BOTTOM_TO_TOP]);
//const std::array<std::array<Eigen::MatrixXd, num_ch_nodes>, 2> Phi_matrix = {Phi_bottom_up, Phi_top_down};

Eigen::VectorXd getInitLambda(Eigen::Quaterniond t_q) {
    const Eigen::Matrix<double, 6, 6> Ad_at_tip = Ad(t_q.toRotationMatrix(), Eigen::Vector3d::Zero());

    Eigen::Matrix<double, 6, 1> F(0, 0, 0, 0, 0, -1);

    return Ad_at_tip.transpose()*F; //no stresses
}

template <unsigned int t_stateDim>
Eigen::MatrixXd getResult(const double* p, qIntegrator<t_stateDim, num_ch_nodes>* base) {
    constexpr unsigned int probDim = num_ch_nodes*t_stateDim;
    constexpr unsigned int unknownDim = t_stateDim*(num_ch_nodes-1);

    Eigen::Matrix<double, unknownDim, 1> X_NN;
    Eigen::Vector<double, t_stateDim> x0(base->x0.data());

    CUDA_CHECK( 
        cudaMemcpy(X_NN.data(), p, sizeof(double) * unknownDim, cudaMemcpyDeviceToHost)
    );

    Eigen::Matrix<double, probDim, 1> X_tilde;
    X_tilde << x0, X_NN;
    X_tilde = base->P*X_tilde;
    Eigen::Matrix<double, num_ch_nodes, t_stateDim> X_stack = Eigen::Map<Eigen::Matrix<double, num_ch_nodes, t_stateDim  >>(X_tilde.data());

    return X_stack;
}

template <unsigned int t_stateDim>
void initIntegrator(qIntegrator<t_stateDim, num_ch_nodes>* base,
                    std::vector<double> qe,
                    cublasHandle_t &t_cublasH)  {
    base->qe = qe;
    base->copy_qe();
    //base->getb();
    base->getK(base->d_phi, base->d_qe, base->d_K);
    base->getA(base->d_K);
}

static void TestNumericalIntegration(benchmark::State& t_state)
{
    cublasHandle_t cublasH = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    //  Here we give some value for the strain
    std::vector<double> qe {
        0,
        0,
        0,
        1.28776905384098,
        -1.63807577049031,
        0.437404540900837,
        0,
        0,
        0
    };

    //Quaternions
    constexpr int qStateDim = 4;
    //const Eigen::Vector4d initQuaternion(1, 0, 0, 0);
    std::vector<double> initQuaternion{1, 0, 0, 0};

    qIntegrator<qStateDim, num_ch_nodes>* qint_ptr = new qIntegrator<qStateDim, num_ch_nodes>(initQuaternion, BOTTOM_TO_TOP, Phi_bottom_up, cusolverH);

    //  Compute Phi as std::array (bye bye Eigen)

    //  Compute K in parallel (memory already allocated)

    //  Construct A in parallel (kernels) as a memeber function

    //  Move copies in the constructor

    //  Solve the system
    qint_ptr->qe = qe;
    qint_ptr->copy_qe();
    qint_ptr->getK(qint_ptr->d_phi, qint_ptr->d_qe, qint_ptr->d_K);
    qint_ptr->getA(qint_ptr->d_K);
    // const auto Q_stack = integrateODE<qStateDim>(qint_ptr, cublasH, cusolverH);
    
    while (t_state.KeepRunning()){
        // qint_ptr->qe = qe;
        // qint_ptr->copy_qe();
        //qint_ptr->getK(qint_ptr->d_phi, qint_ptr->d_qe, qint_ptr->d_K);
        // qint_ptr->getA(qint_ptr->d_K);
        const auto Q_stack = integrateODE<qStateDim>(qint_ptr, cublasH, cusolverH);
    }
}
BENCHMARK(TestNumericalIntegration);




BENCHMARK_MAIN();




// int main(int argc, char *argv[]) {

//     cublasHandle_t cublasH = nullptr;
//     cusolverDnHandle_t cusolverH = nullptr;

//     CUBLAS_CHECK(cublasCreate(&cublasH));
//     CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

//     //  Here we give some value for the strain
//     std::vector<double> qe {
//         0,
//         0,
//         0,
//         1.28776905384098,
//         -1.63807577049031,
//         0.437404540900837,
//         0,
//         0,
//         0
//     };

//     //Quaternions
//     constexpr int qStateDim = 4;
//     //const Eigen::Vector4d initQuaternion(1, 0, 0, 0);
//     std::vector<double> initQuaternion{1, 0, 0, 0};

//     qIntegrator<qStateDim, num_ch_nodes>* qint_ptr = new qIntegrator<qStateDim, num_ch_nodes>(initQuaternion, BOTTOM_TO_TOP, Phi_bottom_up, cusolverH);

//     //  Compute Phi as std::array (bye bye Eigen)

//     //  Compute K in parallel (memory already allocated)

//     //  Construct A in parallel (kernels) as a memeber function

//     //  Move copies in the constructor

//     //  Solve the system
//     initIntegrator<qStateDim>(qint_ptr, qe, cublasH);
//     const auto Q_result = integrateODE<qStateDim>(qint_ptr, cublasH, cusolverH);

//     const auto Q_stack = getResult<qStateDim>(Q_result, qint_ptr);

//     std::cout << "Q_stack: \n" << Q_stack << "\n" << std::endl;


//     return 0;
// }
