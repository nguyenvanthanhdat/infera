#ifndef __CRT_CORE_H__
#define __CRT_CORE_H__

#include<string>
#include <#include <onnxruntime_cxx_api.h>

namespace cinrt::model
{
  class Model
  {
  protected:
    std::shared_ptr<ORT::Env> _env;           
    std::shared_ptr<ORT::Allocator> _allocator;
    std::unique_ptr<ORT::Session> _session;
    std::unique_ptr<ORT::SessionOptions> _sessionOptions;

  public: 
    Model(
      std::string model, 
      bool parallel = true, 
      int graphOpLevel = 0, 
      int interThreads = 0, 
      int intraThreads = 0,
      std::<vector<std::string>> providers = nullptr
    );

  protected: 
    Model(
      std::shared_ptr<ORT::Env> env, 
      std::shared_ptr<ORT::Allocator> allocator, 
      std::string model,
      bool parallel = true,
      int graphOpLevel = 0,
      int intraThreads = 0,
      int interThreads = 0,
      std::<vector<std::string>> providers = nullptr
    );

    std::unique_ptr<Ort::SessionOptions> getSessionOptions(
      bool parallel = true,
      int graphLevel = 0,
      int intraThreads = 0,
      int interThreads = 0
    );

  public:
    std::shared_ptr<Ort::Value> run(const Ort::Value& inputs);
  };
};

#endif