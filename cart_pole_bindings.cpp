#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "cart_pole.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cart_pole_module, m) {
    py::class_<CartPole>(m, "CartPole")
        .def(py::init<const std::string &>())
        .def("reset", &CartPole::reset)
        .def("step", &CartPole::step)
        .def("get_state", &CartPole::get_state)
        .def("is_done", &CartPole::is_done);
}

