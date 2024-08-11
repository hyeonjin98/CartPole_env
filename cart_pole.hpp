#ifndef CART_POLE_HPP
#define CART_POLE_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>

class CartPole {
public:
    double g;
    double e;
    double cart_range;
    double l;
    double m_cart;
    double m_pendulum;
    double cart_position;
    double cart_velocity;
    double pendulum_angle;
    double pendulum_angular_velocity;
    double simulation_dt;
    double control_dt;
    double initial_cart_position;
    double initial_cart_velocity;
    double initial_pendulum_angle;
    double initial_pendulum_angular_velocity;
    
    CartPole(const std::string& cfg);
    void reset();
    std::vector<Eigen::VectorXd> step(double force);
    std::vector<double> get_state();
    bool is_done();
    
private:
    void to_first_order_ODE(const Eigen::VectorXd& y, Eigen::VectorXd& yp, double force);
    void contact(Eigen::VectorXd& y);
};

#endif // CART_POLE_HPP
