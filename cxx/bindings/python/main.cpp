#include <pybind11/pybind11.h>
#include <core.h>
#include <serviceManager.h>

namespace py = pybind11;
using namespace cinrt::model;

PYBIND11_MODULE(ortcxx, m) {
    py::class_<Model, std::shared_ptr<Model>>(m, "Model").def(py::init<std::string, bool, int, int, int>());

    py::class_<modelManager, std::shared_ptr<modelManager>>(m, "ModelManager")
        .def(py::init<std::shared_ptr<Ort::Env>>())
        .def("createModel", &modelManager::createModel)
        .def("getModel", &modelManager::getModel)
        .def("delModel", &modelManager::delModel);

    py::class_<serviceManager, modelManager, std::shared_ptr<serviceManager>>(m, "ServiceManager")
        .def(py::init<std::shared_ptr<Ort::Env>>())
        .def("updateSessionClock", &serviceManager::updateSessionClock)
        .def("getSessionClock", &serviceManager::getSessionClock)
        .def("startGC", &serviceManager::startGC)
        .def("stopGC", &serviceManager::stopGC);
}