#include <iostream>
#include "core.h"
#include <chrono>
#include <thread>
#include <onnxruntime_cxx_api.h>

using namespace cinrt::model;

Ort::Value createMockInput(Ort::MemoryInfo& memoryInfo, int64_t batchSize = 1, int64_t channels = 9, int64_t height = 256, int64_t width = 256) {
    // const std::array<int64_t, 4> inputShape = {1, 9, 256, 256};
    // std::vector<float> inputValues(1 * 9 * 256 * 256, 1.0f);
    const std::array<int64_t, 4> inputShape = {batchSize, channels, height, width};
    std::vector<float> inputValues(batchSize * channels * height * width, 1.0f);
    return Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());
}   

int main() {
    std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    // Ort::SessionOptions sessionOptions;
    std::string modelPath1 = "../models/test_wb.onnx";
    std::string modelPath2 = "../models/yolov7-headface-v1.onnx";
    modelManager manager(env);
    Model* model1 = manager.createModel(modelPath1);
    Model* model2 = manager.createModel(modelPath2);
    // run model
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor1 = createMockInput(memoryInfo, 1, 9, 256, 256);
    Ort::Value inputTensor2 = createMockInput(memoryInfo, 1, 3, 640, 640);
    try {
        std::shared_ptr<std::vector<Ort::Value>> outputTensor = model1->run(inputTensor1);
        float* outputData = outputTensor->at(0).GetTensorMutableData<float>();
        std::cout << "Model1 output: " << outputData[0] << std::endl;

        std::shared_ptr<std::vector<Ort::Value>> outputTensor2 = model2->run(inputTensor2);
        float* outputData2 = outputTensor2->at(0).GetTensorMutableData<float>();
        std::cout << "Model2 output: " << outputData2[0] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    return 0;
}