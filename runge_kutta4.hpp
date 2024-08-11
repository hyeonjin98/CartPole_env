#ifndef RUNGE_KUTTA4_HPP
#define RUNGE_KUTTA4_HPP

#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>

class RungeKutta4 {
public:
    using ODEs = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    using Interrupt = std::function<void(Eigen::VectorXd&)>;

    /**
    * @brief Construct a new 4th order runge-kutta solver.
    * 
    * @param ode The system of first-order ODEs.
    * @param dt The time step size for calculation.
    * @param interr The interrupt function to be called each time step.
    */
    RungeKutta4(ODEs ode, double dt, Interrupt interr = [](Eigen::VectorXd&){}) : ode(ode), dt(dt), interr(interr) {}

    /**
    * @brief Calculate the solution using the 4th order runge-kutta method.
    * 
    * @param y Initial condition vector.
    * @param ti Initial time.
    * @param tf Final time.
    * @return std::vector<Eigen::VectorXd> Solution vectors at each time step.
    */
    std::vector<Eigen::VectorXd> calculate(Eigen::VectorXd& y, double ti, double tf) {
        std::vector<Eigen::VectorXd> output;
        double t = ti;
        Eigen::VectorXd k1(y.size()), k2(y.size()), k3(y.size()), k4(y.size());

        while (t < tf - dt) {
            try{
                interr(y);
                ode(y, k1);
                ode(y + 0.5 * dt * k1, k2);
                ode(y + 0.5 * dt * k2, k3);
                ode(y + dt * k3, k4);
                y += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0;
                t += dt;
                output.push_back(y);
            } catch (const std::exception& e) {
                std::cerr << "Error doing the 4th order runge-kutta calculation: " << e.what() << std::endl;
                break;
            }
        }
        return output;
    }

private:
    ODEs ode;
    double dt;
    Interrupt interr;
};

#endif // RUNGE_KUTTA4_HPP