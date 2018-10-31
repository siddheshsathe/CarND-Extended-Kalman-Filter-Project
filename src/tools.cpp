#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {} // Empty constructor

Tools::~Tools() {} // Destructor

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // sanity check of input validity
    if (estimations.size() != ground_truth.size() || estimations.size() < 1) {
        std::cout << "Input sizes for estimations and ground_truth don't match.... Exiting." << std::endl;
        return rmse;
    }

    // accumulate residuals
    for (size_t i = 0; i < estimations.size(); ++i) {
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse = rmse + residual;
    }

    // compute mean
    rmse = rmse / estimations.size();

    // compute squared root
    rmse = rmse.array().sqrt();// - 0.2;

    return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    
    MatrixXd HJacobian(3, 4);

    // Unroll state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    // Pre-compute some term which recur in the Jacobian
    float c1 = px * px + py * py;
    float c2 = sqrt(c1);
    float c3 = c1 * c2;

    // Sanity check to avoid division by zero
    if (std::abs(c1) == 0) {
        std::cout << "Division by zero.... Exiting" << std::endl;
        return HJacobian;
    }

    // Actually compute Jacobian matrix
    HJacobian << (px / c2),                     (py / c2),                  0,          0,
                 -(py / c1),                    (px / c1),                  0,          0,
                 py * (vx*py - vy*px) / c3,     px * (vy*px - vx*py) / c3,  px / c2,    py / c2;

    return HJacobian;

}