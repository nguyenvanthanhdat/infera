#include <iostream>
#include "core.h"
#include "interface.h"
#include <onnxruntime_cxx_api.h>

using namespace cinrt::model;

modelManager::modelManager(std::shared_ptr<Ort::Env> env) : _env(std::move(env)){}

modelManager::~modelManager(){
    _models.clear();
}

Model* modelManager::createModel(
    std::string model, 
    bool parallel, 
    int graphOpLevel,
    int interThreads, 
    int intraThreads){
    std::unique_ptr<Ort::SessionOptions> _sessionOptions = Model::getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
    std::unique_ptr<Ort::Session> _session = std::make_unique<Ort::Session>(*_env, model.c_str(), *_sessionOptions);
    this->_allocator = std::make_shared<Ort::Allocator>(*_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    std::shared_ptr<Model> newModel = Model::create(_env, _allocator, model, parallel, graphOpLevel, interThreads, intraThreads);
    // std::shared_ptr<Model> newModel = Model::(_env, _allocator, model, parallel, graphOpLevel, interThreads, intraThreads);
    this->_models[model] = newModel;
    return newModel.get();
}

Model* modelManager::getModel(std::string model){
    auto it = this->_models.find(model);
    if (it != this->_models.end()){
        return it->second.get();
    } else {
        std::cout << "Model not found" << std::endl;
        return nullptr;
    }
}

void modelManager::delModel(std::string model){
    auto it = this->_models.find(model);
    if (it != this->_models.end()){
        this->_models.erase(it);
    } else {
        std::cout << "Model not found" << std::endl;
    }
}