#ifndef __SERVICE_H__
#define __SERVICE_H__
#include "core.h"
#include <thread>

using namespace cinrt::model;

class serviceManager : public modelManager
{
private:
    // std::map<char*, float> sessionClock;
    std::map<std::string, std::chrono::steady_clock::time_point> sessionClock;
    std::thread gc;
    std::mutex clockMutex;
    bool stopGCFlag = false;
    
    void garbageCollector();
public:
    // serviceManager(std::shared_ptr<Ort::Env> env, std::shared_ptr<Ort::Allocator> allocator);
    serviceManager(std::shared_ptr<Ort::Env> env);
    ~serviceManager();
    void updateSessionClock(std::string model);
    float getSessionClock(std::string model);
    void startGC();
    void stopGC();
};

#endif // __SERVICE_H__