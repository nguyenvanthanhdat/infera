#ifndef __SERVICE_H__
#define __SERVICE_H__
#include "core.h"

using namespace cinrt::model;

class serviceManager : public modelManager
{
private:
    // std::map<char*, float> sessionClock;
    std::map<char*, std::chrono::steady_clock::time_point> sessionClock;
    std::thread gc;
    std::mutex clockMutex;
    bool stopGCFlag = false;
    
    void garbageCollector();
public:
    serviceManager();
    ~serviceManager();
    void updateSessionClock(char* model);
    float getSessionClock(char* model);
    void startGC();
    void stopGC();
};

#endif // __SERVICE_H__