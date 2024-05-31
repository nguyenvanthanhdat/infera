#ifndef __CRT_CORE_H__
#define __CRT_CORE_H__

#include<map>
#include<string>
#include<onnxruntime_cxx_api.h>

namespace cinrt::model
{
  class Model
  {
  protected:
    std::shared_ptr<Ort::Env> _env;           
    std::shared_ptr<Ort::Allocator> _allocator;
    std::unique_ptr<Ort::Session> _session;
    std::unique_ptr<Ort::SessionOptions> _sessionOptions;

  public: 
    Model(
      std::string model, 
      bool parallel = true, 
      int graphOpLevel = 0, 
      int interThreads = 0, 
      int intraThreads = 0,
      std::vector<std::string>* providers = nullptr
    );

  protected: 
    Model(
      std::shared_ptr<Ort::Env> env, 
      std::shared_ptr<Ort::Allocator> allocator, 
      std::string model,
      bool parallel = true,
      int graphOpLevel = 0,
      int interThreads = 0,
      int intraThreads = 0,
      std::vector<std::string>* providers = nullptr
    );

    std::unique_ptr<Ort::SessionOptions> getSessionOptions(
      bool parallel = true,
      int graphOpLevel = 0,
      int interThreads = 0,
      int intraThreads = 0
    );

  public:
    std::shared_ptr<Ort::Value> run(const Ort::Value& inputs);
  };


  class ModelManager
  {
    protected:
      std::map<std::string, std::shared_ptr<Model>> _models;

    public:
      Model* createModel(std::string model);
      Model* getModel(std::string model);
      void delModel(std::string model);
  };
};

#endif