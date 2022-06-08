#include "quaternionintegrator.h"

template<unsigned int t_Chebyshev_points_number>
QuaternionIntegrator<t_Chebyshev_points_number>::QuaternionIntegrator(const int t_a)
{
    int b = t_a;

    const Eigen::MatrixXd Dn = getDn<t_Chebyshev_points_number>();
    const auto Dn_NN = Dn.block<t_Chebyshev_points_number-1, t_Chebyshev_points_number-1>(0, 0);
    const auto Dn_IN = Dn.block<t_Chebyshev_points_number-1, 1>(0, t_Chebyshev_points_number-1);

    m_D_NN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(m_state_dimension, m_state_dimension), Dn_NN);

    m_D_IN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(m_state_dimension, m_state_dimension), Dn_IN);

    std::cout << "Dn : \n " << Dn << std::endl << std::endl;
    std::cout << "Dn_NN : \n " << Dn_NN << std::endl << std::endl;
    std::cout << "Dn_IN : \n " << Dn_IN << std::endl << std::endl;
    std::cout << "m_D_NN : \n " << m_D_NN << std::endl << std::endl;
    std::cout << "m_D_IN : \n " << m_D_IN << std::endl << std::endl;
}
