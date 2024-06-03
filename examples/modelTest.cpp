// This code use to test model

#include <iostream>
#include "core.h"
#include "interface.h"
#include <onnxruntime_cxx_api.h>

using namespace cinrt::model;

Ort::Value createMockInput(Ort::MemoryInfo& memoryInfo) {
    const std::array<int64_t, 4> inputShape = {1, 9, 256, 256};
    std::vector<float> inputValues(1 * 9 * 256 * 256, 1.0f);
    return Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());
}

int main(){
    std::shared_ptr<Model> model = std::make_shared<Model>("../models/test_wb.onnx");
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = createMockInput(memoryInfo);
    try {
        // std::shared_ptr<Ort::Value> outputTensor = model->run(inputTensor, runOptions);
        std::shared_ptr<std::vector<Ort::Value>> outputTensor = model->run(inputTensor);
        // float* outputData = outputTensor->GetTensorMutableData<float>();
        float* outputData = outputTensor->at(0).GetTensorMutableData<float>();
        std::cout << "Model output: " << outputData[0] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    return 0;
}