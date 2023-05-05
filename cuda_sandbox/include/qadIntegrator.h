#ifndef QAD_INTEGRATOR_H
#define QAD_INTEGRATOR_H

template <unsigned int t_stateDim, unsigned int t_numNodes>
class qadIntegrator : public odeBase<t_stateDim, t_numNodes> {
public:
    qadIntegrator(integrationDirection t_direction) : odeBase<t_stateDim, t_numNodes>(t_direction) {

    };

    void getA() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        this->A = Eigen::Matrix<double, probDim, probDim>::Zero();
    }

    void getb() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;

        //define B matrix for generalised forces
        Eigen::Matrix<double, 6, na> B;

        B << 1, 0, 0,
            0, 1, 0,
            0, 0, 1,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0;

        Eigen::Matrix<double, probDim, 1> b;

        for (unsigned int i = 0; i < t_numNodes; ++i) {
            auto currLambda = this->Lambda.row(i);

            Eigen::Matrix<double, t_stateDim, 1> b_at_ch_point = 
                this->Phi_array[this->direction][i].transpose()*B.transpose()*currLambda.transpose();

            for (unsigned int j = 0; j < t_stateDim; ++j) {
                this->b(i+j*t_numNodes, 0) = b_at_ch_point(j);
            }
        }
    }
};

#endif