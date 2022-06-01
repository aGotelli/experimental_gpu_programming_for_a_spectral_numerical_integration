#ifndef Q_INTEGRATOR_H
#define Q_INTEGRATOR_H

template <unsigned int t_stateDim, unsigned int t_numNodes>
class qIntegrator : public odeBase<t_stateDim, t_numNodes> {
public:
    qIntegrator(integrationDirection t_direction) : odeBase<t_stateDim, t_numNodes>(t_direction) {

    };

    void getA() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;

        Eigen::Vector3d K;
        Eigen::Matrix<double, t_stateDim, t_stateDim> A_at_chebyshev_point;

        for(unsigned int i=0; i < t_numNodes; i++){

            //  Extract the curvature from the strain
            K = this->Phi_array[this->direction][i]*this->qe;

            //  Compute the A matrix of Q' = 1/2 A(K) Q
            A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                    K(0),     0,   K(2),  -K(1),
                                    K(1), -K(2),      0,   K(0),
                                    K(2),  K(1),  -K(0),      0;

            A_at_chebyshev_point = 0.5*A_at_chebyshev_point;

            for (unsigned int row = 0; row < A_at_chebyshev_point.rows(); ++row) {
                for (unsigned int col = 0; col < A_at_chebyshev_point.cols(); ++col) {
                    int row_index = row*num_ch_nodes+i;
                    int col_index = col*num_ch_nodes+i;
                    this->A(row_index, col_index) = A_at_chebyshev_point(row, col);
                }
            }
        }
    }

    void getb() override {
        constexpr unsigned int probDim = t_stateDim*t_numNodes;
        this->b = Eigen::Vector<double, probDim>::Zero();
    }
};

#endif