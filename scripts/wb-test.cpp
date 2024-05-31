#define UNICODE

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <utils/image.cpp>

#include "ORTManager.h"
#include <benchmark/benchmark.h>

void Run_session(ORTManager &ortManager, Ort::Value &inputTensor, Ort::Value &outputTensor1, std::string sessionName){ 
     // inter + intra, test from ORTSesstionInterface.h
    ortManager.run(sessionName, inputTensor, outputTensor1); 
}

template <class ...Args>

static void BM_Run_Function(benchmark::State& state, Args&&... args) {
    //get args tuple
    auto args_tuple = std::make_tuple(std::move(args)...);

    // This code gets timed
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 256;
    constexpr int64_t height = 256;
    constexpr int64_t numPixels = width * height;
    constexpr int64_t numElements = numChannels * numPixels;

    //image path
    const std::string imageFileD = "../tests/sample/8D5U5524_D.png"; 
    const std::string imageFileS = "../tests/sample/8D5U5524_S.png";
    const std::string imageFileT = "../tests/sample/8D5U5524_T.png";
    const std::string outputImage = "../tests/sample/output_1.png";
    const auto modelPath = "../models/test_wb.onnx";
    
    // load images
    cv::Mat imageD = loadImage(imageFileD);
    cv::Mat imageS = loadImage(imageFileS);
    cv::Mat imageT = loadImage(imageFileT);

    // notificate the user if the image is not loaded
    if (imageD.empty() || imageS.empty() || imageT.empty()) {
        std::cout << "Image not loaded.\n";
    }

    int originalWidth = imageD.cols;
    int originalHeight = imageD.rows;

    // resize image
    cv::Mat imaged, images, imaget; 
    cv::resize(imageD, imaged, cv::Size(width, height));
    cv::resize(imageS, images, cv::Size(width, height));
    cv::resize(imageT, imaget, cv::Size(width, height));

    imageD = fixedImage(imageD);
    imageS = fixedImage(imageS);
    imageT = fixedImage(imageT);

    std::vector<std::vector<std::vector<float>>> imagedVec = mat2vec(imaged); //List[List[List[float]]]
    std::vector<std::vector<std::vector<float>>> imagesVec = mat2vec(images);
    std::vector<std::vector<std::vector<float>>> imagetVec = mat2vec(imaget);
    std::vector<std::vector<std::vector<float>>> imageDVec = mat2vec(imageD);
    std::vector<std::vector<std::vector<float>>> imageSVec = mat2vec(imageS);
    std::vector<std::vector<std::vector<float>>> imageTVec = mat2vec(imageT);
    
    std::vector<std::vector<std::vector<float>>> inputMatrix;    
    inputMatrix.insert(inputMatrix.end(), imagedVec.begin(), imagedVec.end());
    inputMatrix.insert(inputMatrix.end(), imagesVec.begin(), imagesVec.end());
    inputMatrix.insert(inputMatrix.end(), imagetVec.begin(), imagetVec.end());
 
    // define shape
    const std::array<int64_t, 4> inputShape = {1, 9, height, width};
    const std::array<int64_t, 4> outputShape = {1, 3, height, width};

    // define array with new shape
    std::vector<float> input(height * width * 3 * 3); // *vectors -> flatten list[list[list[float]]] -> list[float]
    std::vector<float> output(height * width * 3);

    // flatten inputMatrix to input
    size_t index = 0;
    for (int c = 0; c < 3 * 3; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                input[index] = inputMatrix[c][i][j];
                index++;
            }
        }
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), inputShape.data(), inputShape.size()
    );
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        memory_info, output.data(), output.size(), outputShape.data(), outputShape.size()
    );
    
    ORTManager ortManager;
    std::string sessionName = "test";
    std::string mode = "async";
    std::string optimize_path = "";

    ortManager.createSession(
        sessionName, 
        modelPath, 
        std::get<0>(args_tuple), //threadOptions
        state.range(0), //optmize_level
        optimize_path, 
        mode, 
        std::get<1>(args_tuple), //execution
        state.range(1) //num_thread
    );
    // Perform setup here
    for (auto _ : state) {
        Run_session(
            ortManager,
            inputTensor,
            outputTensor,
            sessionName
        );
        
    }
}


// Register the function as a benchmark
BENCHMARK_CAPTURE(BM_Run_Function, "test_wb", std::string("inter_intra"), std::string("sequential"))->ArgsProduct({
    { 0, 1, 2, 3},
    { 0, 1, 2, 3, 4 }
})->Unit(
    benchmark::kMillisecond
);
// Run the benchmark
BENCHMARK_MAIN();