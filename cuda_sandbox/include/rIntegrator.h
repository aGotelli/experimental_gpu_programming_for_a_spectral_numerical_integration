#ifndef R_INTEGRATOR_H
#define R_INTEGRATOR_H

template <unsigned int t_stateDim, unsigned int t_numNodes>
class rIntegrator : public odeBase<t_stateDim, t_numNodes> {
public:
    rIntegrator(integrationDirection t_direction) : odeBase<t_stateDim, t_numNodes>(t_direction) {

    };

    void getA() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        this->A = Eigen::Matrix<double, probDim, probDim>::Zero();
    }

    void getb() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        
        Eigen::Quaterniond quaternion;

        for (unsigned int i = 0; i < t_numNodes; ++i) {
            auto q = this->Q.row(i);
            quaternion = {q[0], q[1], q[2], q[3]};


            Eigen::Matrix<double, t_stateDim, 1> b_at_ch_point = quaternion.toRotationMatrix()*Eigen::Vector3d(1, 0, 0);

            for (unsigned int j = 0; j < t_stateDim; ++j) {
                this->b(i+j*t_numNodes, 0) = b_at_ch_point(j);
            }
        }
    }
};

#endif