#include <include/core.h>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <future>

using namespace cinrt::model;

Model::Model(
  char* model,
  bool parallel = true,
  int graphOpLevel = 0,
  int interThreads = 0,
  int intraThreads = 0,
  std::vector<std::string>* providers = nullptr
) {
  this->_env = std::make_shared<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_INFO));
  this->_sessionOptions = this->getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
  this->_session = std::make_unique<Ort::Session>(*this->_env, model, *this->_sessionOptions);
  this->inputNames = std::make_shared<std::array<const char*, 1>>(new std::array<const char*, 1>{this->_session->GetInputNameAllocated(0, *this->_allocator).get()});
  this->outputNames = std::make_shared<std::array<const char*, 1>>(new std::array<const char*, 1>{this->_session->GetOutputNameAllocated(0, *this->_allocator).get()});
}

Model::Model(
  std::shared_ptr<Ort::Env> env,
  std::shared_ptr<Ort::Allocator> allocator,
  char* model,
  bool parallel = true,
  int graphOpLevel = 0,
  int interThreads = 0,
  int intraThreads = 0
) {
  this->_env = env;
  this->_allocator = allocator;
  this->_sessionOptions = this->getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
  this->_session = std::make_unique<Ort::Session>(new Ort::Session(*this->_env, model, *_sessionOptions));
  this->inputNames = std::make_shared<std::array<const char*, 1>>(new std::array<const char*, 1>{this->_session->GetInputNameAllocated(0, *this->_allocator).get()});
  this->outputNames = std::make_shared<std::array<const char*, 1>>(new std::array<const char*, 1>{this->_session->GetOutputNameAllocated(0, *this->_allocator).get()});
}

std::unique_ptr<Ort::SessionOptions> Model::getSessionOptions
(
  bool parallel = true, 
  int graphOpLevel = 0, 
  int intraThreads = 0, 
  int interThreads = 0
) {
  std::unique_ptr<Ort::SessionOptions> sessionOptions = std::make_unique<Ort::SessionOptions>(new Ort::SessionOptions());
  // std::unique_ptr<Ort::SessionOptions> sessionOptions = std::make_unique<Ort::SessionOptions>();
  if (parallel)
    sessionOptions->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  else
    sessionOptions->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  if (intraThreads > 0)
    sessionOptions->SetIntraOpNumThreads(intraThreads);
  if (interThreads > 0)
    sessionOptions->SetInterOpNumThreads(interThreads);
  switch (graphOpLevel)
  {
  case 0:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    break;
  case 1:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    break;
  case 2:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    break;
  case 3:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    break;
  default:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    break;
  }
  return sessionOptions;
}

std::shared_ptr<Ort::Value> Model::run(const Ort::Value& inputs, const Ort::RunOptions& runOptions = Ort::RunOptions()){
  if (this->_session == nullptr)
    throw std::runtime_error("Session is not initialized");
  this->inputNames.get();
  try {
    std::shared_ptr<Ort::Value> output = std::make_shared<Ort::Value>(this->_session->Run(runOptions, this->inputNames->data(), &inputs, 1, this->outputNames->data(), 1));
  }
  catch (Ort::Exception& exception) {
    std::cout << "Error: " << exception.what() << std::endl;
  }

}

std::future<std::shared_ptr<Ort::Value>> Model::runAsync(const Ort::Value& inputs, const Ort::RunOptions runOptions = Ort::RunOptions()){
  if (this->_session == nullptr)
    throw std::runtime_error("Session is not initialized");
  return std::async(std::launch::async, &Model::run, this, std::cref(inputs), std::cref(runOptions));
  // return future.get();
}