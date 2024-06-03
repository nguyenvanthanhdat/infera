#include "serviceManager.h"
#include <iostream>
#include <thread>
#include <chrono>

serviceManager::serviceManager(std::shared_ptr<Ort::Env> env) : modelManager(env){
    startGC();
}

serviceManager::~serviceManager(){
    stopGC();
    if (gc.joinable()){
        gc.join();
    }
}

void serviceManager::updateSessionClock(std::string model){
    std::lock_guard<std::mutex> lock(clockMutex);
    sessionClock[model] = std::chrono::steady_clock::now();
    // std::cout << "Updated session clock for model: " << model << std::endl;
}

float serviceManager::getSessionClock(std::string model){
    std::lock_guard<std::mutex> lock(clockMutex);
    // auto currentTime = std::chrono::steady_clock::now();
    auto it = sessionClock.find(model);
    if (it != sessionClock.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - it->second).count();
        // std::cout << "Session duration for model " << model << ": " << duration << " seconds" << std::endl;  // Debugging statement
        return duration;
    }
    // std::cout << "Model " << model << " not found in sessionClock" << std::endl;
    return 0.0f;
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