#include <pybind11/pybind11.h>
#include <string>

std::string displayFn(std::string i) {
    return i;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "Simple python function that displays something"     
    m.def("display", &display, "Echo a string",
          pybind11::arg("inp"));
}