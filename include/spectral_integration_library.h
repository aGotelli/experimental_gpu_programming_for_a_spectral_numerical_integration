#ifndef SPECTRAL_INTEGRATION_LIBRARY_H
#define SPECTRAL_INTEGRATION_LIBRARY_H

#include <Eigen/Dense>
#include <fstream>
#include "chebyshev_differentiation.h"
#include "spectral_integration_utilities.h"


constexpr unsigned int na = 3;  //  Kirkhoff rod
constexpr unsigned int ne = 3;  // dimesion of qe
constexpr unsigned int number_of_chebyshev_nodes = 16;

/*!
 * \brief writeToFile writes a Eigen matrix into file
 * \param t_name    name of the file
 * \param t_matrix  the Eigen matrix to write into the file
 * \param t_relative_path_from_build the relative path from the build folder to the file location. Default is none so the file is written in the build directory)
 * \param t_format  the specification for writing. (Default in column major allignment, with comma column separator and 8 digits precision)
 */
void writeToFile(std::string t_name,
                 const Eigen::MatrixXd &t_matrix,
                 std::string t_relative_path_from_build = "",
                 const Eigen::IOFormat &t_format = Eigen::IOFormat(16, 0, ","))
{
    if(not t_relative_path_from_build.empty()){
        //  Ensure relative path ends with a backslash only if a path is given
        if(not t_relative_path_from_build.ends_with('/'))
            t_relative_path_from_build.append("/");
    }


    //  Ensure it ends with .csv
    if(t_name.find(".csv") == std::string::npos)
        t_name.append(".csv");

    //  The file will be created in the location given by the realtive path and with the given name
    const auto file_name_and_location = t_relative_path_from_build + t_name;

    //  Create file in given location with given name
    std::ofstream file(file_name_and_location.c_str());

    //  Put matrix in this file
    file << t_matrix.format(t_format);

    //  Close the file
    file.close();
 }

Eigen::Matrix<double, 3, 3> getHat(const Eigen::Vector3d t_v) {
    Eigen::Matrix<double, 3, 3> hatMatrix;

    hatMatrix << 0, -t_v(2), t_v(1),
                 t_v(2), 0, -t_v(0),
                 -t_v(1), t_v(0), 0;

    return hatMatrix;
}

template <unsigned int t_state_dimension>
static const Eigen::MatrixXd integrateODE(const Eigen::VectorXd &t_initial_state,
                                          const Eigen::MatrixXd A,
                                          const Eigen::VectorXd b,
                                          const integrationDirection t_direction,
                                          const std::string filename){

    constexpr unsigned int prob_dimension = t_state_dimension * number_of_chebyshev_nodes;
    constexpr unsigned int unknow_state_dimension = t_state_dimension * (number_of_chebyshev_nodes - 1);

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, t_state_dimension, Eigen::ColMajor> MatrixNchebNs;

    typedef Eigen::Matrix<double, prob_dimension, prob_dimension> MatrixNpNp;
    typedef Eigen::Matrix<double, prob_dimension, 1> VectorNp;

    typedef Eigen::Matrix<double, prob_dimension, t_state_dimension> MatrixNpNs;

    typedef Eigen::Matrix<double, unknow_state_dimension, unknow_state_dimension> MatrixNuNu;
    typedef Eigen::Matrix<double, unknow_state_dimension, 1> VectorNu;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, number_of_chebyshev_nodes> MatrixNchebNcheb;

    typedef Eigen::Matrix<double, number_of_chebyshev_nodes, t_state_dimension, Eigen::ColMajor> MatrixNchebNs;
    //Quaternion: 28171 ns

    const MatrixNpNp  P = getP<t_state_dimension, number_of_chebyshev_nodes>();
    //Quaternion: 179401 ns

    const MatrixNchebNcheb Dn = getDn<number_of_chebyshev_nodes>(t_direction);
    //Quaternion: 260589 ns
    const MatrixNpNp D = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(t_state_dimension, t_state_dimension), Dn);
    //Quaternion: 348201 ns

    const MatrixNpNp Ap = P.transpose() * A * P;
    //Quaternion: 4847982 ns
    const MatrixNpNp Dp = P.transpose() * D * P;
    //Quaternion: 9259881 ns
    const VectorNp bp   = P * b;

//    const MatrixNpNs D_IT = Dp.block<prob_dimension, state_dimension>(0, 0);
    const MatrixNpNs D_IT = Dp.block(0, 0, prob_dimension, t_state_dimension);
    //Quaternion: 9271547 ns
    const MatrixNpNs A_IT = Ap.block(0, 0, prob_dimension, t_state_dimension);
    //Quaternion: 9271737 ns
    const VectorNp b_IT = ( D_IT - A_IT ) * t_initial_state;
    //Quaternion: 9348664 ns

    const MatrixNuNu D_NN = Dp.block(t_state_dimension, t_state_dimension, unknow_state_dimension, unknow_state_dimension);
    //Quaternion: 9362063 ns
    const MatrixNuNu A_NN = Ap.block(t_state_dimension, t_state_dimension, unknow_state_dimension, unknow_state_dimension);
    //Quaternion: 9417226 ns
    const VectorNu ivp = b_IT.block(t_state_dimension, 0, unknow_state_dimension, 1);
    //Quaternion: 9423439 ns
    const VectorNu b_NN   = bp.block(t_state_dimension, 0, unknow_state_dimension, 1);
    //Quaternion: 9510764 ns

    const VectorNu X_NN = (D_NN - A_NN).inverse() * (b_NN - ivp);
    //Quaternion: 12917772 ns

    const VectorNp X_tilde = P * (VectorNp() << t_initial_state, X_NN).finished();
    //Quaternion: 12950755 ns

    const MatrixNchebNs X_stack = Eigen::Map<const MatrixNchebNs>(X_tilde.data());\
    //Quaternion: 13037364 ns

//    //writeToFile(filename, X_stack);

    return X_stack;
//    return Eigen::Matrix<double, prob_dimension, prob_dimension>::Zero();
}

#endif
