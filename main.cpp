#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>


#include <boost/math/special_functions/chebyshev.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Core>





static Eigen::MatrixXd getD(const unsigned int t_N,
                            const int t_min=0,
                            const int t_max=1)
{
    Eigen::MatrixXd DN(t_N+1, t_N+1);

    //  Define the integration lenght (default will be 1, undimensional)
    const double L = t_max - t_min;



    //  Define the Chebyshev points on the unit circle
    std::vector<double> x(t_N+1);
    std::generate(x.begin(), x.end(), [&](){static unsigned int j;
        return (L/2)*(1 +cos( M_PI * static_cast<double>(j++) / static_cast<double>(t_N) ));
    });


    //  Create a matrix every row filled with a point value
    Eigen::MatrixXd X(t_N+1, t_N+1);
    for(unsigned int i=0; i<=t_N;i++) {
        for(unsigned int j=0; j<=t_N;j++) {
            X(i,j) = x[i];
        }
    }


    std::vector<double> c(t_N+1);

    for(unsigned int i=0; i<=t_N;i++) {
        unsigned int ci;
        if(i==0 or i==t_N)
            ci = 2;
        else ci=1;
        c[i] = pow(-1, i)*ci;
    }

    Eigen::MatrixXd C(t_N+1, t_N+1);
    for(unsigned int i=0; i<=t_N;i++) {
        for(unsigned int j=0; j<=t_N;j++) {
            C(i,j) = c[i]*(1/c[j]);
        }
    }

    //  Definition of the temporary matrix A
    Eigen::MatrixXd A(t_N+1, t_N+1);
    A = X - X.transpose() + Eigen::MatrixXd::Identity(t_N+1, t_N+1);

    //  Obtain off diagonal element of DN
    for(unsigned int i=0; i<=t_N;i++) {
        for(unsigned int j=0; j<=t_N;j++) {
            DN(i,j) = C(i, j) / A(i, j);
        }
    }


    Eigen::VectorXd row_sum = Eigen::VectorXd::Zero(t_N+1);

    //  Sum every row of Dn
    for(unsigned int i=0; i<=t_N;i++) {
        for(unsigned int j=0; j<=t_N;j++) {
            row_sum[i] += DN(i, j);
        }
    }

    //  Correct the dyagonal element
    for(unsigned int i=0; i<=t_N;i++) {
        DN(i, i) -= row_sum[i];
    }


    return DN;
}



template<unsigned int na, unsigned int ne>
static const Eigen::MatrixXd Phi(const double t_s, const double t_L=1.0){

    //  Normalize the cenerline coordinates
    const double X = t_s/t_L;

    //  Compute the values of the polynomial for every element of the strain field
    Eigen::VectorXd Base(ne);
    for(unsigned int i=0; i<ne; i++)
        Base[i] = boost::math::chebyshev_t(i, X);


    //  Define the matrix of bases
    Eigen::MatrixXd Phi = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(na, na), Base.transpose());


return Phi;
}

template<unsigned int t_ny, unsigned int t_N, unsigned int t_ns>
Eigen::MatrixXd getP()
{

    //  Define a vector of index from 1 to t_ny
    Eigen::VectorXi idxY(t_ny);
    unsigned int i=0;
    for(auto& index : idxY)
        index = ++i;

    //  Now the index for the first element of each vector
    //  needs to skyp number of chebyshev points
    Eigen::VectorXi idxX0 = idxY * t_N;

    //  Report them back to C++ correct indexing
    idxY -= Eigen::VectorXi::Ones(t_ny);
    idxX0 -= Eigen::VectorXi::Ones(t_ny);

    //  Initialization as Identity matrix
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(t_ns, t_ns);

    //  Start applying mapping principles
    P(idxY,idxY) << Eigen::MatrixXd::Zero(idxY.rows(),
                                         idxY.rows());
    P(idxX0,idxX0) = Eigen::MatrixXd::Zero(idxX0.rows(),
                                           idxX0.rows());
    P(idxY,idxX0) = Eigen::MatrixXd::Identity(t_ny, t_ny);
    P(idxX0,idxY) = Eigen::MatrixXd::Identity(t_ny, t_ny);

    return P;
}


template<unsigned int t_ny, unsigned int t_N, unsigned int t_ns, unsigned int t_L, unsigned int na, unsigned int ne>
Eigen::MatrixXd getA(Eigen::VectorXd &t_qe)
{
    //  Define the Chebyshev points on the unit circle
    std::vector<double> x(t_N+1);

    for(unsigned int j=0; j<=t_N;j++) {
        x[j] = (static_cast<double>(t_L)/2)*(1 +cos( M_PI * static_cast<double>(j) / static_cast<double>(t_N) ));
    }



    std::vector<Eigen::MatrixXd> A_stack;

    for(unsigned int i=0; i<=t_N+1; i++){
        //  Extract the curvature from the strain
        const Eigen::Vector3d K = Phi<na, ne>(x[i])*t_qe;

        //  Compute the A matrix of Q' = 1/2 A(K) Q
        Eigen::MatrixXd A_local(4,4);
        A_local <<   0, -K(0),  -K(1),  -K(2),
                  K(0),     0,   K(2),  -K(1),
                  K(1), -K(2),      0,   K(0),
                  K(2),  K(1),  -K(0),      0;

        A_stack.push_back(0.5*A_local);
    }

    //  Declare the matrix for the system Ax = b
    Eigen::MatrixXd A(t_ns, t_ns);

    //  Define a vector of index from 1 to t_ny
    Eigen::VectorXi idxY(t_ny);
    unsigned int i=0;
    for(auto& index : idxY)
        index = ++i;

    //  Now the index for the first element of each vector
    //  needs to skyp number of chebyshev points
    Eigen::VectorXi idxX0 = idxY * t_N;

    Eigen::VectorXi idxYN = (idxY - Eigen::VectorXi::Ones(t_ny)) * t_N;

    //  Report to C++ indexing
    idxY -= Eigen::VectorXi::Ones(t_ny);
    idxX0 -= Eigen::VectorXi::Ones(t_ny);
    idxYN -= Eigen::VectorXi::Ones(t_ny);


    //  Populate this matrix with all the elements in the right order
    Eigen::VectorXi index(t_ny);
    for(unsigned int in=0; in<t_N; in++){
        Eigen::VectorXi tmp(t_ny);
        tmp.setConstant(t_ny, 1, in+1);
        index = tmp + idxYN;
        A(index, index) = A_stack[in] ;
    }

    return A;
}

void writeToFile(const std::string &t_name, const Eigen::MatrixXd t_matrix, const Eigen::IOFormat &t_format=Eigen::IOFormat())
{
    std::ofstream file(t_name.c_str());
    file << t_matrix.format(t_format);
 }

int main()
{

    Eigen::Vector4d Y0(1, 0, 0, 0);

    //  Dimension of the state
    constexpr unsigned int state_dimension = Y0.rows();
    //  Number of Chebyshev nodes
    constexpr unsigned int number_of_chebyshev_nodes = 29;
    //  Problem size is the total number of elements
    constexpr unsigned int prob_dimension = state_dimension * number_of_chebyshev_nodes;
    //  The subset of unknows in the problem
    constexpr unsigned int unknow_state_dimension = state_dimension * (number_of_chebyshev_nodes - 1);

    //  Consider unit length rod
    constexpr unsigned int L = 1;

    //  Number od admitted strain fields and number of mode per strain field
    constexpr unsigned int na = 3;
    constexpr unsigned int ne = 3;

    //  Const curvature strain field
    Eigen::VectorXd qe(ne*na);
    qe <<   0,
            0,
            0,
            0,
            0,
            0,
         -2.9630,
          0.0000,
          0.0000;


    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNsNs;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNs;

    typedef Eigen::Matrix<double, prob_dimension, state_dimension> MatrixNsNy;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, state_dimension, Eigen::ColMajor> MatrixNpNy;

    const MatrixNsNs  P = getP<state_dimension, number_of_chebyshev_nodes, prob_dimension>();

    const MatrixNsNs A = getA<state_dimension, number_of_chebyshev_nodes, prob_dimension, L, na, ne>(qe);
    const VectorNs b = Eigen::Matrix<double, prob_dimension, 1>::Zero();
    const Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> Dn = getD(number_of_chebyshev_nodes-1);

    const MatrixNsNs D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(state_dimension, state_dimension), Dn);

    //  Apply transformation of initial condition onto ODE's matrices
    const MatrixNsNs Ap = P.transpose() * A * P;
    const MatrixNsNs Dp = P.transpose() * D * P;
    const VectorNs bp   = P.transpose() * b;




    //  Compute the ivp
    const MatrixNsNy D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);
    const MatrixNsNy A_IT = Ap.block<prob_dimension, state_dimension>(0, 0);
    const VectorNs ivp = ( D_IT - A_IT ) * Y0;


    // Calculation of solution
    const MatrixNuNu D_NN = Dp.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const MatrixNuNu A_NN = Ap.block<unknow_state_dimension, unknow_state_dimension>(state_dimension, state_dimension);
    const VectorNu ivp_NN = ivp.block<unknow_state_dimension, 1>(state_dimension, 0);
    const VectorNu b_NN   = bp.block<unknow_state_dimension, 1>(state_dimension, 0);

    const MatrixNuNu A_tilde = D_NN - A_NN;
    const VectorNu b_tilde = b_NN - ivp_NN;


    const VectorNu Yn = A_tilde.inverse() * b_tilde;


    const VectorNs Y = P * (VectorNs() << Y0, Yn).finished();


    const MatrixNpNy Y_stack = Eigen::Map<const MatrixNpNy>(Y.data());
    std::cout<< "Q_stack = " << std::endl << Y_stack <<std::endl << std::endl;


    writeToFile("A_stack.csv", A_tilde, Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ",\n"));
    writeToFile("B_stack.csv", b_tilde, Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ",\n"));



    writeToFile("Q_stack.csv", Y_stack);



    return 0;
}
