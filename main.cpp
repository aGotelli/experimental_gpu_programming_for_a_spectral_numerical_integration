#include "chebyshev_differentiation.h"
#include "utilities.h"

static const unsigned int number_of_Chebyshev_points = 16;

static const unsigned int quaternion_state_dimension = 4;
static const unsigned int position_dimension = 3;


static const unsigned int quaternion_problem_dimension = quaternion_state_dimension * (number_of_Chebyshev_points-1);

static constexpr unsigned int ne = 3;
static constexpr unsigned int na = 3;

static const auto x = ComputeChebyshevPoints<number_of_Chebyshev_points>();

Eigen::Matrix<double, ne*na, 1> qe;



void updateA(const Eigen::VectorXd &t_qe, Eigen::MatrixXd &A_NN, const Eigen::MatrixXd &D_NN)
{

    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<number_of_Chebyshev_points>();


    Eigen::Vector3d K;
    Eigen::MatrixXd A_at_chebyshev_point(quaternion_state_dimension, quaternion_state_dimension);
    unsigned int left_corner_row;
    unsigned int left_corner_col;
    for(unsigned int i=0; i<x.size()-1; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;


        for (unsigned int row = 0; row < quaternion_state_dimension; ++row) {
            for (unsigned int col = 0; col < quaternion_state_dimension; ++col) {
                int row_index = row*(number_of_Chebyshev_points-1)+i;
                int col_index = col*(number_of_Chebyshev_points-1)+i;
                A_NN(row_index, col_index) = D_NN(row_index, col_index) - 0.5*A_at_chebyshev_point(row, col);
            }
        }

    }

}


Eigen::VectorXd integrateQuaternions()
{
    //  Obtain the Chebyshev differentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //  Extract D_NN from the differentiation matrix (for the spectral integration)
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);

    //  Extract D_IN (for the propagation of initial conditions)
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);


    //  Now stack the matrices in the diagonal of bigger ones (as meny times as the state dimension)
    const Eigen::MatrixXd D_NN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_NN);
    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(quaternion_state_dimension, quaternion_state_dimension), Dn_IN);


    Eigen::MatrixXd A_NN = D_NN;
    updateA(qe, A_NN, D_NN);

    Eigen::VectorXd q_init(4);
    q_init << 1, 0, 0, 0;


    Eigen::VectorXd ivp = D_IN*q_init;

    const auto b = Eigen::VectorXd::Zero(quaternion_problem_dimension);

    const auto res = b - ivp;

    Eigen::VectorXd Q_stack = A_NN.inverse() * res;

    //  move back Q_stack

    return Q_stack;


}


Eigen::MatrixXd updatePositionb(Eigen::MatrixXd t_Q_stack) {

    Eigen::Matrix<double, number_of_Chebyshev_points-1, position_dimension> b;

    Eigen::Quaterniond q;

    for (unsigned int i = 0; i < number_of_Chebyshev_points-1; ++i) {


        q = { t_Q_stack(i),
              t_Q_stack(i  +  (number_of_Chebyshev_points-1)),
              t_Q_stack(i + 2*(number_of_Chebyshev_points-1)),
              t_Q_stack(i + 3*(number_of_Chebyshev_points-1)) };


        b.block<1,3>(i, 0) = (q.toRotationMatrix()*Eigen::Vector3d(1, 0, 0)).transpose();

    }
    return b;
}




Eigen::MatrixXd integratePosition()
{
    const auto Q_stack = integrateQuaternions();
    Eigen::MatrixXd b_NN(number_of_Chebyshev_points-1, position_dimension);


    Eigen::Vector3d r_init;
    r_init << 0,
              0,
              0;


    //  Get the diffetentiation matrix
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();

    //  Extract the submatrix responsible for the spectral integration
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);

    //  This matrix remains constant so we can pre invert
    const auto Dn_NN_inv = Dn_NN.inverse();

    //  Extract the submatrix responsible for propagating the initial conditions
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

    Eigen::MatrixXd ivp(number_of_Chebyshev_points-1, position_dimension);
    for(unsigned int i=0; i<ivp.rows(); i++)
        ivp.row(i) = Dn_IN(i, 0) * r_init.transpose();

    Eigen::MatrixXd r_stack(number_of_Chebyshev_points-1, position_dimension);



    b_NN = updatePositionb(Q_stack);

    r_stack = Dn_NN_inv*(b_NN - ivp);


    return r_stack;
}




int main(int argc, char *argv[])
{


    //  Here we give some value for the strain
//    qe.setZero();
    qe <<   0,
            0,
            0,
            1.2877691307032,
           -1.63807499160786,
            0.437406679142598,
            0,
            0,
            0;

    const auto Q_stack = integrateQuaternions();
    std::cout << "Q_stack : \n" << Q_stack << std::endl;


    const auto r_stack = integratePosition();
    std::cout << "r_stack : \n" << r_stack << std::endl;


//    const auto Lambda_stack = integrateLambda();
//    std::cout << "Q_stack : \n" << Q_stack << std::endl;


//    const auto Qa_stack = integrateGeneralisedForces();
//    std::cout << "r_stack : \n" << r_stack << std::endl;




    return 0;
}
