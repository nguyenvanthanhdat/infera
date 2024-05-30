#include<core.h>

Model::Model(
  std::string model,
  bool parallel = true,
  int graphOpLevel = 0,
  int interThreads = 0,
  int intraThreads = 0
) {
  this->_env = std::make_shared(new Ort::Env(ORT_LOGGING_LEVEL_INFO));
  this->_sessionOptions = this->getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
  this->_session = new Ort::Session(this->_env, model.c_str(), _sessionOptions);
}

Model::Model(
  std::shared_ptr<Ort::Env> env,
  std::shared_ptr<Ort::Allocator> allocator,
  bool parallel = true,
  int graphOpLevel = 0,
  int interThreads = 0,
  int intraThreads = 0
) {
  this->_env = env;
  this->_allocator = allocator;
  this->_sessionOptions = this->getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
  this->_session = new Ort::Session(this->_env, model.c_str(), _sessionOptions);
}

Model::getSessionOptions
(
  bool parallel = true, 
  int graphOpLevel = 0, 
  int interThreads = 0, 
  int intraThreads = 0
) {
  std::unique_ptr<Ort::SessionOptions> sessionOptions = new ORT::SessionOptions();
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
  case 3:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    break;
  default:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL)
    break;
  }
  return sessionOptions
}