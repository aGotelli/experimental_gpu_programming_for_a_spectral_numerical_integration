#include "chebyshev_differentiation.h"
#include "utilities.h"

static constexpr unsigned int number_of_Chebyshev_points = 16;

static constexpr unsigned int state_dimension = 4;

static constexpr unsigned int problem_dimension = state_dimension * (number_of_Chebyshev_points-1);
typedef Eigen::Matrix<double, problem_dimension, problem_dimension> MatrixNN;
typedef Eigen::Matrix<double, problem_dimension, 1> VectorNd;

static constexpr unsigned int ne = 3;
static constexpr unsigned int na = 3;

static const auto x = ComputeChebyshevPoints<number_of_Chebyshev_points>();

Eigen::Matrix<double, ne*na, 1> qe;


Eigen::MatrixXd getQuaternionA(Eigen::VectorXd &t_qe) {

    constexpr unsigned int probDimension = state_dimension*number_of_Chebyshev_points;

    Eigen::Vector3d K;
    Eigen::Matrix<double, state_dimension, state_dimension> A_at_chebyshev_point;
    //  Declare the matrix for the system Ax = b
    Eigen::Matrix<double, probDimension, probDimension> A =
            Eigen::Matrix<double, probDimension, probDimension>::Zero();

    for(unsigned int i=0; i < number_of_Chebyshev_points; i++){

        //  Extract the curvature from the strain
        K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        A_at_chebyshev_point <<      0, -K(0),  -K(1),  -K(2),
                                  K(0),     0,   K(2),  -K(1),
                                  K(1), -K(2),      0,   K(0),
                                  K(2),  K(1),  -K(0),      0;

        A_at_chebyshev_point = 0.5*A_at_chebyshev_point;

        for (unsigned int row = 0; row < A_at_chebyshev_point.rows(); ++row) {
            for (unsigned int col = 0; col < A_at_chebyshev_point.cols(); ++col) {
                int row_index = row*number_of_Chebyshev_points+i;
                int col_index = col*number_of_Chebyshev_points+i;
                A(row_index, col_index) = A_at_chebyshev_point(row, col);
            }
        }
    }

    return A;
}

void updateA(const Eigen::Matrix<double, ne*ne, 1> &t_qe, MatrixNN &A_NN, const MatrixNN &D_NN)
{

    //  Define the Chebyshev points on the unit circle
    const auto x = ComputeChebyshevPoints<number_of_Chebyshev_points>();


    Eigen::Vector3d K;
    Eigen::Matrix<double, state_dimension, state_dimension> A_at_chebyshev_point;
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


        for (unsigned int row = 0; row < state_dimension; ++row) {
            for (unsigned int col = 0; col < state_dimension; ++col) {
                int row_index = row*(number_of_Chebyshev_points-1)+i;
                int col_index = col*(number_of_Chebyshev_points-1)+i;
                A_NN(row_index, col_index) = D_NN(row_index, col_index) - 0.5*A_at_chebyshev_point(row, col);
            }
        }

    }

}


Eigen::VectorXd integrateQuaternions()
{
    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

    const Eigen::MatrixXd D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);
    const MatrixNN D_NN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn_NN);

    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn_IN);


    MatrixNN A_NN = D_NN;
    updateA(qe, A_NN, D_NN);

    Eigen::VectorXd q_init(4);
    q_init << 1, 0, 0, 0;

    Eigen::VectorXd ivp = D_IN*q_init;

    const auto b = VectorNd::Zero();

    Eigen::VectorXd Q_stack = A_NN.inverse() * (b - ivp);

    return Q_stack;


}


Eigen::Matrix<double, number_of_Chebyshev_points-1, 3> updatePositionb(Eigen::MatrixXd t_Q_stack) {

    Eigen::Matrix<double, number_of_Chebyshev_points-1, 3> b;

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
    Eigen::Matrix<double, number_of_Chebyshev_points-1, 3> b_NN;


    Eigen::Vector3d r_init;
    r_init << 0,
              0,
              0;


    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    const auto Dn_NN_inv = Dn_NN.inverse();
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

    Eigen::Matrix<double, number_of_Chebyshev_points-1, 3> ivp;
    for(unsigned int i=0; i<ivp.rows(); i++)
        ivp.row(i) = Dn_IN(i, 0) * r_init.transpose();

    Eigen::Matrix<double, number_of_Chebyshev_points-1, 3> r_stack;



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

    return 0;
}
