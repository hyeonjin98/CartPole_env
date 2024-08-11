#include <iostream>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "cart_pole.hpp"
#include "runge_kutta4.hpp"

using namespace std;
using namespace Eigen;


CartPole::CartPole(const string& cfg) {
    YAML::Node config = YAML::LoadFile(cfg);

    g = config["simulation"]["parameters"]["g"].as<double>();
    e = config["simulation"]["parameters"]["coeff_of_restitution"].as<double>();
    cart_range = config["simulation"]["parameters"]["cart_range"].as<double>();
    l = config["simulation"]["parameters"]["pendulum_length"].as<double>();
    m_cart = config["simulation"]["parameters"]["m_cart"].as<double>();
    m_pendulum = config["simulation"]["parameters"]["m_pendulum"].as<double>();
    cart_position = config["simulation"]["init"]["cart_position"].as<double>();
    cart_velocity = config["simulation"]["init"]["cart_velocity"].as<double>();
    pendulum_angle = config["simulation"]["init"]["pendulum_angle"].as<double>();
    pendulum_angular_velocity = config["simulation"]["init"]["pendulum_angular_velocity"].as<double>();
    simulation_dt = config["simulation"]["time"]["simulation_dt"].as<double>();
    control_dt = config["simulation"]["time"]["control_dt"].as<double>();

    initial_cart_position = cart_position;
    initial_cart_velocity = cart_velocity;
    initial_pendulum_angle = pendulum_angle;
    initial_pendulum_angular_velocity = pendulum_angular_velocity;
}

void CartPole::reset() {
    
    cart_position = initial_cart_position;
    pendulum_angle = initial_pendulum_angle;
    cart_velocity = initial_cart_velocity;
    pendulum_angular_velocity = initial_pendulum_angular_velocity;
}

vector<Eigen::VectorXd> CartPole::step(double force) {
    Eigen::VectorXd current_state(4);
    current_state << cart_position, pendulum_angle, cart_velocity, pendulum_angular_velocity;

    // 멤버 함수 바인딩: 람다 함수를 사용하여 멤버 함수를 바인딩
    RungeKutta4::ODEs odes = [this, force](const Eigen::VectorXd& y, Eigen::VectorXd& yp) {
        this->to_first_order_ODE(y, yp, force); // this 포인터를 사용하여 멤버 함수 호출
    };
    RungeKutta4::Interrupt contact = [this](Eigen::VectorXd& y) {
        this->contact(y); // this 포인터를 사용하여 멤버 함수 호출
    };

    RungeKutta4 rk4(odes, simulation_dt, contact);
    std::vector<Eigen::VectorXd> output_data = rk4.calculate(current_state, 0, control_dt);

    cart_position = output_data.back()[0];
    pendulum_angle = output_data.back()[1];
    cart_velocity = output_data.back()[2];
    pendulum_angular_velocity = output_data.back()[3];

    return output_data;
}

vector<double> CartPole::get_state() {
    return {cart_position, pendulum_angle, cart_velocity, pendulum_angular_velocity};
}

bool CartPole::is_done() {
    return false;
}

void CartPole::to_first_order_ODE(const VectorXd& y, VectorXd& yp, double force) {
    double d = y[0];
    double th = y[1];
    double dp = y[2];
    double thp = y[3];

    Matrix2d D;
    D << m_cart + m_pendulum, l * m_pendulum * cos(th), 
         l * m_pendulum * cos(th), pow(l, 2) * m_pendulum;
    Matrix2d C;
    C << 0, -l * m_pendulum * sin(th) * thp, 
         0, 0;
    Vector2d P;
    P << 0, 
         g * l * m_pendulum * sin(th);
    Vector2d f;
    f << force, 0;

    Vector2d qp, qpp;
    qp << dp, thp;
    qpp = D.inverse() * (f - C * qp - P);

    yp.resize(4);
    yp << qp, qpp;
}

void CartPole::contact(VectorXd& y) {
    double position = y[0];
    double angle = y[1];
    double velocity = y[2];
    double angular_velocity = y[3];

    y[1] = fmod(angle, 2 * M_PI);
    if (angle < 0) { y[1] += 2 * M_PI; }
    if (abs(position) >= cart_range) {
        if (position > 0) { y[0] = cart_range; }
        else { y[0] = -cart_range; }

        y[2] = -e * velocity;
        // y[3] = angular_velocity + (1 + e) * (m_cart + m_pendulum) * velocity / (m_pendulum * l * std::cos(angle));
    }
}
