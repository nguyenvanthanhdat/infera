#include "serviceManager.h"
#include <iostream>
#include <thread>
#include <chrono>

serviceManager::serviceManager(){
    startGC();
}

serviceManager::~serviceManager(){
    stopGC();
    if (gc.joinable()){
        gc.join();
    }
}

void serviceManager::updateSessionClock(char* model){
    std::lock_guard<std::mutex> lock(clockMutex);
    sessionClock[model] = std::chrono::steady_clock::now();
}

void serviceManager::updateSessionClock(char* model){
    std::lock_guard<std::mutex> lock(clockMutex);
    sessionClock[model] = std::chrono::steady_clock::now();
}

void serviceManager::garbageCollector(){
    while (!stopGCFlag)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Run every 5ms
        std::lock_guard<std::mutex> lock(clockMutex);
        auto currentTime = std::chrono::steady_clock::now();
        for (auto it = sessionClock.begin(); it != sessionClock.end();){
            if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - it->second).count() > 500){
                delModel(it->first);
                it = sessionClock.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    
}

void serviceManager::startGC(){
    if (gc.joinable()){
        gc.join();
    }
    stopGCFlag = false;
    gc = std::thread(&serviceManager::garbageCollector, this);
}

void serviceManager::stopGC(){
    stopGCFlag = true;
}