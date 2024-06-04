#include "core.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <future>
#include <thread>

using namespace cinrt::model;

Model::Model(
  std::string model,
  bool parallel,
  int graphOpLevel,
  int interThreads,
  int intraThreads
) {
  this->_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
  this->_sessionOptions = this->getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
  this->_session = std::make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
  this->_allocator = std::make_shared<Ort::Allocator>(*this->_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(0, *this->_allocator);
  Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(0, *this->_allocator);
  this->inputNames = std::make_shared<const char*>(inputName.release());
  this->outputNames = std::make_shared<const char*>(outputName.release());
}

Model::Model(
  std::shared_ptr<Ort::Env> env,
  std::shared_ptr<Ort::Allocator> allocator,
  std::string model,
  bool parallel,
  int graphOpLevel,
  int interThreads,
  int intraThreads
) {
  _env = env;
  _allocator = allocator;
  _sessionOptions = getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
  _session = std::make_unique<Ort::Session>(*_env, model.c_str(), *_sessionOptions);
  this->_allocator = std::make_shared<Ort::Allocator>(*this->_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(0, *this->_allocator);
  Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(0, *this->_allocator);
  this->inputNames = std::make_shared<const char*>(inputName.release());
  this->outputNames = std::make_shared<const char*>(outputName.release());
}

std::unique_ptr<Ort::SessionOptions> Model::getSessionOptions(
  bool parallel, 
  int graphOpLevel, 
  int intraThreads, 
  int interThreads
) {
  std::unique_ptr<Ort::SessionOptions> sessionOptions = std::make_unique<Ort::SessionOptions>(Ort::SessionOptions());
  if (parallel)
    sessionOptions->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  else
    sessionOptions->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  if (intraThreads > 0)
    sessionOptions->SetIntraOpNumThreads(intraThreads);
  if (interThreads > 0)
    sessionOptions->SetInterOpNumThreads(interThreads);
  switch (graphOpLevel){
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

std::shared_ptr<std::vector<Ort::Value>> Model::run(
  const Ort::Value& inputs,
  std::shared_ptr<const char*> outputHead,
  const Ort::RunOptions& runOptions){
  if (outputHead != nullptr)
    this->outputNames = outputHead;
  if (this->_session == nullptr)
    throw std::runtime_error("Session is not initialized");
  try {
    std::vector<Ort::Value> output_vector = this->_session->Run(runOptions, &*inputNames, &inputs, 1, &*outputNames, 1);
    return std::make_shared<std::vector<Ort::Value>>(std::move(output_vector));
  }
  catch (Ort::Exception& exception) {
    std::cout << "Error: " << exception.what() << std::endl;
  }
  return nullptr;
}

std::future<std::shared_ptr<std::vector<Ort::Value>>> Model::runAsync(
  const Ort::Value& inputs, 
  std::shared_ptr<const char*> outputHead,
  const Ort::RunOptions runOptions){
  if (outputHead != nullptr)
    this->outputNames = outputHead;
  if (this->_session == nullptr)
    throw std::runtime_error("Session is not initialized");
  return std::async(std::launch::async, &Model::run, this, std::cref(inputs), std::cref(outputNames), std::cref(runOptions));
}