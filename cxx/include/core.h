#ifndef __CRT_CORE_H__
#define __CRT_CORE_H__

#include <string>
#include <map>
#include <future>
#include <onnxruntime_cxx_api.h>
// #include <include/interface.h>

namespace cinrt::model
{
  class Model
  {
  protected:
    std::shared_ptr<Ort::Env> _env;           
    std::shared_ptr<Ort::Allocator> _allocator;
    std::shared_ptr<std::array<const char*, 1>> inputNames;
    std::shared_ptr<std::array<const char*, 1>> outputNames;
    std::unique_ptr<Ort::Session> _session;
    std::unique_ptr<Ort::SessionOptions> _sessionOptions;

  public: 
    Model(
      char* model, 
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
      char* model,
      bool parallel = true,
      int graphOpLevel = 0,
      int interThreads = 0,
      int intraThreads = 0
      // std::vector<std::string>* providers = nullptr
    );
    std::unique_ptr<Ort::SessionOptions> getSessionOptions(
      bool parallel = true,
      int graphOpLevel = 0,
      int intraThreads = 0,
      int interThreads = 0
    );

    friend class ModelManager;

  public:
    std::shared_ptr<Ort::Value> run(const Ort::Value& inputs, const Ort::RunOptions& runOptions);
    // std::shared_ptr<Ort::Value> runAsync(const Ort::Value& inputs, std::function<void(std::shared_ptr<Ort::Value>)> callback);
    std::future<std::shared_ptr<Ort::Value>> runAsync(const Ort::Value& inputs, const Ort::RunOptions runOptions);
  };


  class modelManager
  {
    protected:
      std::map<char*, std::shared_ptr<Model>> _models;

    public:
      Model* createModel(char* model);
      Model* getModel(char* model);
      void delModel(char* model);
  };
};

#endif