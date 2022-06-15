#include "




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

template<unsigned int t_stateDimension>
Eigen::MatrixXd getPositionb(Eigen::MatrixXd t_Q) {
    constexpr unsigned int probDimension = t_stateDimension * number_of_Chebyshev_points;

    Eigen::Matrix<double, probDimension, 1> b;
    Eigen::Quaterniond quaternion;

    for (unsigned int i = 0; i < number_of_Chebyshev_points; ++i) {
        auto q = t_Q.row(i);
        quaternion = {q[0], q[1], q[2], q[3]};


        Eigen::Matrix<double, t_stateDimension, 1> b_at_ch_point = quaternion.toRotationMatrix()*Eigen::Vector3d(1, 0, 0);

        for (unsigned int j = 0; j < t_stateDimension; ++j) {
            b(i+j*number_of_Chebyshev_points, 0) = b_at_ch_point(j);
        }
    }
    return b;
}

void positionOld()
{
    const auto Q = integrateQold();

    constexpr unsigned int positionStateDimension = 3;
    constexpr unsigned int positionProblemDimension = positionStateDimension * number_of_Chebyshev_points;
    const auto position_A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();

    const auto b = getPositionb<positionStateDimension>(Q);

    const Eigen::Vector3d initial_position(0, 0, 0);


    constexpr unsigned int prob_dimension = 3 * number_of_Chebyshev_points;
    constexpr unsigned int unknow_state_dimension = 3 * (number_of_Chebyshev_points - 1);

    typedef Eigen::Matrix<double, number_of_Chebyshev_points, 3, Eigen::ColMajor> MatrixNchebNs;

    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, 3> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_Chebyshev_points, number_of_Chebyshev_points> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_Chebyshev_points, 3, Eigen::ColMajor> MatrixNchebNs;

    const MatrixNpNp  P = getP<3, number_of_Chebyshev_points>();

    const MatrixNchebNcheb Dn = getDn<number_of_Chebyshev_points>();
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(3, 3), Dn);

    const auto A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();

    const MatrixNpNp Ap = P.transpose() * A * P;
    const MatrixNpNp Dp = P.transpose() * D * P;
    const VectorNp bp   = P * b;

//    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);
    const MatrixNpNs D_IT = Dp.block(0, 0, prob_dimension, 3);
    const MatrixNpNs A_IT = Ap.block(0, 0, prob_dimension, 3);
    const VectorNp b_IT = ( D_IT - A_IT ) * initial_position;

    const MatrixNuNu D_NN = Dp.block(3, 3, unknow_state_dimension, unknow_state_dimension);
    const MatrixNuNu A_NN = Ap.block(3, 3, unknow_state_dimension, unknow_state_dimension);
    const VectorNu ivp = b_IT.block(3, 0, unknow_state_dimension, 1);
    const VectorNu b_NN   = bp.block(3, 0, unknow_state_dimension, 1);

    const VectorNp b_NN_tilde = P * (VectorNp() << Eigen::Vector3d::Zero(), b_NN).finished();

    const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);


    const VectorNp X_tilde = P * (VectorNp() << initial_position, X_NN).finished();

    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    std::cout << "X_stack : \n" << X_stack << std::endl;
}

void positionNew()
{
    const auto Q_stack = integrateQ();
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


    //while (t_state.KeepRunning()){

        b_NN = updatePositionb(Q_stack);

        r_stack = Dn_NN_inv*(b_NN - ivp);


        std::cout << "r_stack : \n" << r_stack << std::endl;
    //}
}




static void RevisedNumericalIntegrationPosition(benchmark::State& t_state)
{

    const auto Q_stack = integrateQ();
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


    while (t_state.KeepRunning()){

        b_NN = updatePositionb(Q_stack);

        r_stack = Dn_NN_inv*(b_NN - ivp);


        //std::cout << "r_stack : \n" << r_stack << std::endl;
    }
}
BENCHMARK(RevisedNumericalIntegrationPosition);



static void RevisedNumericalIntegrationQuaternion(benchmark::State& t_state)
{

    const Eigen::MatrixXd Dn = getDn<number_of_Chebyshev_points>();
    const Eigen::MatrixXd Dn_NN = Dn.block<number_of_Chebyshev_points-1, number_of_Chebyshev_points-1>(0, 0);
    const Eigen::MatrixXd Dn_IN = Dn.block<number_of_Chebyshev_points-1, 1>(0, number_of_Chebyshev_points-1);

    const Eigen::MatrixXd D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);
    const MatrixNN D_NN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn_NN);

    const Eigen::MatrixXd D_IN = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn_IN);

    Eigen::Matrix<double, ne*na, 1> qe;
    qe <<   0,
            0,
            0,
            1.2877691307032,
           -1.63807499160786,
            0.437406679142598,
            0,
            0,
            0;

    MatrixNN A_NN = D_NN;
    VectorNd b_NN = VectorNd::Zero();

    Eigen::VectorXd q_init(4);
    q_init << 1, 0, 0, 0;

    Eigen::Matrix<double, problem_dimension, 1> ivp;
    Eigen::Matrix<double, problem_dimension, 1> Q_stack;

    while (t_state.KeepRunning()){

        updateA(qe, A_NN, D_NN);
        ivp = D_IN*q_init;
        Q_stack = A_NN.inverse() * (b_NN - ivp);
    }
}
BENCHMARK(RevisedNumericalIntegrationQuaternion);


static void OriginalNumericalIntegrationQuaternion(benchmark::State& t_state)
{
    constexpr unsigned int prob_dimension = state_dimension * number_of_Chebyshev_points;
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_Chebyshev_points - 1);

        typedef Eigen::Matrix<double, number_of_Chebyshev_points, state_dimension, Eigen::ColMajor> MatrixNchebNs;

        typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
        typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

        typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNpNs;

        typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
        typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

        typedef Eigen::Matrix<double, number_of_Chebyshev_points, number_of_Chebyshev_points> MatrixNchebNcheb;

        typedef Eigen::Matrix<double, number_of_Chebyshev_points, state_dimension, Eigen::ColMajor> MatrixNchebNs;

        const MatrixNpNp  P = getP<state_dimension, number_of_Chebyshev_points>();

        const MatrixNchebNcheb Dn = getDn<number_of_Chebyshev_points>();
        const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

        Eigen::VectorXd qe(ne*na);
        qe <<   0,
                0,
                0,
                1.2877691307032,
               -1.63807499160786,
                0.437406679142598,
                0,
                0,
                0;
        Eigen::VectorXd q_init(4);
        q_init << 1, 0, 0, 0;


        MatrixNpNp A;
        const auto b = Eigen::Matrix<double, prob_dimension, 1>::Zero();

        MatrixNpNp Ap;
        const MatrixNpNp Dp = P.transpose() * D * P;
        const VectorNp bp   = P * b;

        const MatrixNpNs D_IT = Dp.block(0, 0, prob_dimension, state_dimension);
        MatrixNpNs A_IT;
        const VectorNp b_IT = ( D_IT - A_IT ) * q_init;

        const MatrixNuNu D_NN = Dp.block(state_dimension, state_dimension, unknow_state_dimension, unknow_state_dimension);
        MatrixNuNu A_NN;
        VectorNu ivp;
        const VectorNu b_NN   = bp.block(state_dimension, 0, unknow_state_dimension, 1);

        VectorNu X_NN;

        VectorNp X_tilde;

        MatrixNchebNs X_stack;


        while (t_state.KeepRunning()){

            A = getQuaternionA(qe);

            Ap = P.transpose() * A * P;

            A_IT = Ap.block(0, 0, prob_dimension, state_dimension);

            A_NN = Ap.block(state_dimension, state_dimension, unknow_state_dimension, unknow_state_dimension);
            ivp = b_IT.block(state_dimension, 0, unknow_state_dimension, 1);


            X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

            X_tilde = P * (VectorNp() << q_init, X_NN).finished();

            X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());
        }

}
BENCHMARK(OriginalNumericalIntegrationQuaternion);


static void OriginalNumericalIntegrationPosition(benchmark::State& t_state)
{
    const auto Q = integrateQold();

    constexpr unsigned int positionStateDimension = 3;
    constexpr unsigned int positionProblemDimension = positionStateDimension * number_of_Chebyshev_points;
    const auto position_A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();



    const Eigen::Vector3d initial_position(0, 0, 0);


    constexpr unsigned int prob_dimension = 3 * number_of_Chebyshev_points;
    constexpr unsigned int unknow_state_dimension = 3 * (number_of_Chebyshev_points - 1);

    typedef Eigen::Matrix<double, number_of_Chebyshev_points, 3, Eigen::ColMajor> MatrixNchebNs;

    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, 3> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_Chebyshev_points, number_of_Chebyshev_points> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_Chebyshev_points, 3, Eigen::ColMajor> MatrixNchebNs;

    const MatrixNpNp  P = getP<3, number_of_Chebyshev_points>();

    const MatrixNchebNcheb Dn = getDn<number_of_Chebyshev_points>();
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(3, 3), Dn);

    const auto A = Eigen::Matrix<double, positionProblemDimension, positionProblemDimension>::Zero();
    VectorNp b = getPositionb<positionStateDimension>(Q);

    const MatrixNpNp Ap = P.transpose() * A * P;
    const MatrixNpNp Dp = P.transpose() * D * P;
    VectorNp bp;


//    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);
    const MatrixNpNs D_IT = Dp.block(0, 0, prob_dimension, 3);
    const MatrixNpNs A_IT = Ap.block(0, 0, prob_dimension, 3);
    VectorNp b_IT;


    const MatrixNuNu D_NN = Dp.block(3, 3, unknow_state_dimension, unknow_state_dimension);
    const MatrixNuNu A_NN = Ap.block(3, 3, unknow_state_dimension, unknow_state_dimension);
    VectorNu ivp;
    VectorNu b_NN;

    VectorNu X_NN;

    VectorNp X_tilde;

    MatrixNchebNs X_stack;



    while (t_state.KeepRunning()){

        b = getPositionb<positionStateDimension>(Q);

        bp   = P * b;

        b_IT = ( D_IT - A_IT ) * initial_position;

        ivp = b_IT.block(3, 0, unknow_state_dimension, 1);
        b_NN   = bp.block(3, 0, unknow_state_dimension, 1);

        X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);

        X_tilde = P * (VectorNp() << initial_position, X_NN).finished();

        X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());

    }

}
BENCHMARK(OriginalNumericalIntegrationPosition);


//int main(int argc, char *argv[])
//{
//        positionOld();

//        positionNew();

//        return 0;
//}
BENCHMARK_MAIN();
