#include <iostream>
#include "core.h"
#include "interface.h"

using namespace cinrt::model;

Model* modelManager::createModel(char* model){
    std::shared_ptr<Model> obj = std::make_shared<Model>(new Model(model));
    this->_models[model] = obj;
    return obj.get();
}

Model* modelManager::getModel(char* model){
    auto it = this->_models.find(model);
    if (it != this->_models.end()){
        return it->second.get();
    } else {
        std::cout << "Model not found" << std::endl;
        return nullptr;
    }
}

void modelManager::delModel(char* model){
    auto it = this->_models.find(model);
    if (it != this->_models.end()){
        this->_models.erase(it);
    } else {
        std::cout << "Model not found" << std::endl;
    }
}