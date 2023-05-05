#ifndef LAMBDA_INTEGRATOR_H
#define LAMBDA_INTEGRATOR_H

#include "lie_algebra_utilities.h"

// __global__ void getHat(const double* x, double* X_hat) {

// }

// __global__ void build_ad_xi(const double* K_hat, const double* Gamma_hat, double* ad_xi) {
//     const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
//     const unsigned int tidx = threadIdx.x;
//     const unsigned int bidx = blockIdx.x;

//     ad_xi[tid+(tid/3)*6] = K_hat[tid];
//     ad_xi[tid+3+(tid/3)*6] = Gamma_hat[tid];
//     ad_xi[tid+3*6+(tid/3)*6] = 0;
//     ad_xi[tid+3+3*6+(tid/3)*6] = K_hat[tid];
// }

template <unsigned int t_stateDim, unsigned int t_numNodes>
class lambdaIntegrator : public odeBase<t_stateDim, t_numNodes> {
public:
    lambdaIntegrator(integrationDirection t_direction) : odeBase<t_stateDim, t_numNodes>(t_direction) {

    };

    void getA() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        //same as t_stateDim but needs to be redefined because of bug with Eigen::Matrix::block
        constexpr unsigned int stateDim = 6;

        Eigen::Vector3d K;
        Eigen::Matrix3d K_hat;
        const auto Gamma_hat = skew(Eigen::Vector3d(1, 0, 0));
        Eigen::Matrix<double, stateDim, stateDim> ad_xi;

        for(unsigned int i=0; i < t_numNodes; i++){

            //  Extract the curvature from the strain
            K = this->Phi_array[this->direction][i]*this->qe;
            K_hat = skew(K);

            ad_xi.block<3, 3>(0, 0) = K_hat;
            ad_xi.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
            ad_xi.block<3, 3>(3, 0) = Gamma_hat;
            ad_xi.block<3, 3>(3, 3) = K_hat;

            for (unsigned int row = 0; row < ad_xi.rows(); ++row) {
                for (unsigned int col = 0; col < ad_xi.cols(); ++col) {
                    int row_index = row*t_numNodes+i;
                    int col_index = col*t_numNodes+i;
                    this->A(row_index, col_index) = ad_xi.transpose()(row, col);
                }
            }
        }
    }

    void getb() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;

        //density of iron [kg/m^3]
        constexpr double rodDensity =  7874;

        //1cm radius
        const double rodCrossSec = M_PI * pow(0.01, 2);

        //gravitational acceleration
        constexpr double g = 9.8067;

        const double rodSpecWeight = rodDensity*rodCrossSec;
        const double gravForces = rodSpecWeight*g;

        Eigen::Quaterniond quaternion;
        Eigen::Matrix<double, t_stateDim, 1> F_ext = {0, 0, 0, 0, 0, gravForces};

        //going from the top down for stresses as opposed to bottom up
        for (int i = t_numNodes - 1; i >= 0; i--) {
            auto q = this->Q.row(i);
            quaternion = {q[0], q[1], q[2], q[3]};

            Eigen::Matrix<double, t_stateDim, 1> b_at_ch_point = Ad(quaternion.toRotationMatrix(), Eigen::Vector3d::Zero())*F_ext;

            for (unsigned int j = 0; j < t_stateDim; ++j) {
                this->b(t_numNodes-1-i+j*t_numNodes, 0) = b_at_ch_point(j);
            }
        }
    }
};

#endif