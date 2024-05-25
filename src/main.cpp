#define UNICODE

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <unistd.h>

#include "utils/image.cpp"


int main() {
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 256;
    constexpr int64_t height = 256;
    constexpr int64_t numPixels = width * height;
    constexpr int64_t numElements = numChannels * numPixels;

    // print present path
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working dir: " << cwd << std::endl;
    } else {
        printf("getcwd() error");
        return 0;
    }
    const std::string imageFileD = "../tests/sample/8D5U5524_D.png";
    const std::string imageFileS = "../tests/sample/8D5U5524_S.png";
    const std::string imageFileT = "../tests/sample/8D5U5524_T.png";
    const std::string outputImage = "../tests/sample/output.png";
    const auto modelPath = "../models/test_wb.onnx";
    
    // load images
    cv::Mat imageD = loadImage(imageFileD);
    cv::Mat imageS = loadImage(imageFileS);
    cv::Mat imageT = loadImage(imageFileT);

    // notificate the user if the image is not loaded
    if (imageD.empty() || imageS.empty() || imageT.empty()) {
        std::cout << "Image not loaded.\n";
        return 0;
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
    std::cout << "Image shape: " << imageD.size() << std::endl;

    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Convert image to vec";
    std::vector<std::vector<std::vector<float>>> imagedVec = mat2vec(imaged);
    std::vector<std::vector<std::vector<float>>> imagesVec = mat2vec(images);
    std::vector<std::vector<std::vector<float>>> imagetVec = mat2vec(imaget);
    std::vector<std::vector<std::vector<float>>> imageDVec = mat2vec(imageD);
    std::vector<std::vector<std::vector<float>>> imageSVec = mat2vec(imageS);
    std::vector<std::vector<std::vector<float>>> imageTVec = mat2vec(imageT);
    std::cout << "Image shape: " << imagedVec.size() << "x" << imagedVec[0].size() << "x" <<  imagedVec[0][0].size() << std::endl;

    // transpose image (channels, newHeight, newWidth) -> (newHeight, newWidth, channels)
    // std::vector<int> af_shape = {1, 2, 0};
    // std::vector<std::vector<std::vector<float>>> transposeImageD = transposeImage(
    //     imageDVec, af_shape);
    // std::vector<std::vector<std::vector<float>>> transposeImageS = transposeImage(
    //     imageSVec, af_shape);
    // std::vector<std::vector<std::vector<float>>> transposeImageT = transposeImage(
    //     imageTVec, af_shape);
    // std::cout << "Transposed image shape: " << transposeImageD.size() << "x" << transposeImageD[0].size() << "x" <<  transposeImageD[0][0].size() << std::endl;
    
    // concatenate images shape (3, newHeight, newWidth) -> (3 * 3, newHeight, newWidth)
    std::vector<std::vector<std::vector<float>>> inputMatrix;
    // inputMatrix.insert(inputMatrix.end(), transposeImageD.begin(), transposeImageD.end());
    // inputMatrix.insert(inputMatrix.end(), transposeImageS.begin(), transposeImageS.end());
    // inputMatrix.insert(inputMatrix.end(), transposeImageT.begin(), transposeImageT.end());
    
    inputMatrix.insert(inputMatrix.end(), imagedVec.begin(), imagedVec.end());
    inputMatrix.insert(inputMatrix.end(), imagesVec.begin(), imagesVec.end());
    inputMatrix.insert(inputMatrix.end(), imagetVec.begin(), imagetVec.end());
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Concatenate image shape: " << inputMatrix.size() << "x" << inputMatrix[0].size() << "x" <<  inputMatrix[0][0].size() << std::endl;

    // define shape
    const std::array<int64_t, 4> inputShape = {1, 3 * 3, height, width};
    const std::array<int64_t, 4> outputShape = {1, 3, height, width};

    // define array with new shape
    // std::array<float, static_cast<std::size_t>(newHeight * newWidth * 3 * 3)> input;
    // std::array<float, newHeight * newWidth> output; // output shape is [batch_size,3,256,256]
    std::vector<float> input(height * width * 3 * 3);
    std::vector<float> output(height * width * 3);

    size_t index = 0;
    for (const auto& channel : inputMatrix) {
        for (const auto& row : channel) {
            for (float value : row) {
                input[index++] = value;
            }
        }
    }

    // use CUDA if available
    Ort::SessionOptions ort_session_options;
    OrtCUDAProviderOptions options;
    options.device_id = 0;

    // OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);

    // create session
    session = Ort::Session(env, modelPath, ort_session_options);

    // define Tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), inputShape.data(), inputShape.size()
    );

    // Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    //     memory_info, input.data(), input.size(), inputShape.data(), inputShape.size()
    // );
    Ort::Value outputTensor1 = Ort::Value::CreateTensor<float>(
        memory_info, output.data(), output.size(), outputShape.data(), outputShape.size()
    );

    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = {inputName.get()};
    const std::array<const char*, 1> outputNames = {outputName.get()};
    
    // run inference
    try {
        // session.Run(
        //     runOptions, 
        //     inputNames.data(), &inputTensor, 1, 
        //     outputNames.data(), &outputTensor1, 1);
        auto outputValues = session.Run(
            runOptions, 
            inputNames.data(), &inputTensor, 1, 
            outputNames.data(), &outputTensor1, 1);
    }
    catch (Ort::Exception& exception) {
        std::cout << "Error: " << exception.what() << std::endl;
        return 1;
    }

    // Access the data in the output tensor
    float* outputData = outputTensor1.GetTensorMutableData<float>();

    // Convert output tensor data to cv::Mat
    cv::Mat outputMat(height, width, CV_32FC3, outputData);

    // Resize outputMat to original size
    cv::resize(outputMat, outputMat, cv::Size(originalWidth, originalHeight));

    // Convert outputMat to vector for further processing
    auto outputVec = mat2vec(outputMat);

    // Create save image tensor initialized to 0
    std::vector<std::vector<std::vector<float>>> saveImageVec(3, std::vector<std::vector<float>>(originalHeight, std::vector<float>(originalWidth, 0.0f)));

    // Combine output with original images
    std::vector<std::vector<std::vector<float>>> imgs[] = {imageDVec, imageSVec, imageTVec};
    for (int opt = 0; opt < 3; ++opt) {
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < originalHeight; ++i) {
                for (int j = 0; j < originalWidth; ++j) {
                    saveImageVec[c][i][j] += outputVec[c][i][j] * imgs[opt][c][i][j];
                }
            }
        }
    }

    // Transpose the save image
    std::vector<int> bf = {1, 2, 0};
    auto saveImageVecT = transposeImage(saveImageVec, bf);

    // Convert saveImageVecT to cv::Mat
    cv::Mat saveImageMat(originalHeight, originalWidth, CV_32FC3);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < originalHeight; ++i) {
            for (int j = 0; j < originalWidth; ++j) {
                saveImageMat.at<cv::Vec3f>(i, j)[c] = saveImageVecT[c][i][j];
            }
        }
    }

    // Normalize saveImageMat to [0, 255]
    saveImageMat.convertTo(saveImageMat, CV_8UC3, 255.0f, 0);

    // Convert RGB to BGR
    cv::cvtColor(saveImageMat, saveImageMat, cv::COLOR_RGB2BGR);

    // Save image
    cv::imwrite(outputImage, saveImageMat);

    return 0;


    //// code temp
    // std::cout << "Check output: " << outputTensor1. << std::endl;
    // // convert output to cv::Mat
    // cv::Mat outputMat = cv::Mat(height, width, CV_32FC3, output.data());

    // // resize outputMat to original size
    // std::cout << std::string(50, '-') << std::endl;
    // std::cout << "Resize outputMat to original size\n";
    // cv::resize(outputMat, outputMat, cv::Size(originalWidth, originalHeight));

    // // convert outputMat to vector
    // std::vector<std::vector<std::vector<float>>> outputVec = mat2vec(outputMat);
    // std::cout << "Output shape: " << outputVec.size() << "x" << outputVec[0].size() << "x" <<  outputVec[0][0].size() << std::endl;

    // // create save image tensor with 0 value
    // std::cout << std::string(50, '-') << std::endl;
    // std::cout << "Create save image tensor with 0 value\n";
    // std::vector<std::vector<std::vector<float>>> saveImageVec(
    //     3, std::vector<std::vector<float>>(
    //         originalHeight, std::vector<float>(
    //             originalWidth)));
    // for (int c = 0; c < 3; ++c) {
    //     for (int i = 0; i < originalHeight; ++i) {
    //         for (int j = 0; j < originalWidth; ++j) {
    //             saveImageVec[c][i][j] = 0;
    //         }
    //     }
    // }

    // // sum (outputVec * imageD, outputVec * imageS, outputVec * imageT)
    // std::cout << std::string(50, '-') << std::endl;
    // std::cout << "Sum (outputVec * imageD, outputVec * imageS, outputVec * imageT)\n";
    
    // std::cout << "Save image shape bf: " << saveImageVec.size() << "x" << saveImageVec[0].size() << "x" <<  saveImageVec[0][0].size() << std::endl;

    // std::vector<std::vector<std::vector<std::vector<float>>>> imgs = {imageDVec, imageSVec, imageTVec};

    // for (int opt = 0; opt < 3; ++opt) {
    //     for (int c = 0; c < 3; ++c) {
    //         for (int i = 0; i < originalHeight; ++i) {
    //             for (int j = 0; j < originalWidth; ++j) {
    //                 saveImageVec[c][i][j] += outputVec[c][i][j] * imgs[opt][c][i][j];
    //             }
    //         }
    //     }
    // }
    // std::cout << "Save image shape af: " << saveImageVec.size() << "x" << saveImageVec[0].size() << "x" <<  saveImageVec[0][0].size() << std::endl;

    // std::cout << std::string(50, '-') << std::endl;
    // std::cout << "Transpose save image\n";
    // std::vector<int> bf = {1, 2, 0};
    // std::vector<std::vector<std::vector<float>>> saveImageVecT = transposeImage(
    //     saveImageVec, bf);
    // std::cout << "Save image shape bf: " << saveImageVecT.size() << "x" << saveImageVecT[0].size() << "x" <<  saveImageVecT[0][0].size() << std::endl;

    // // convert saveImageVecT to cv::Mat
    // cv::Mat saveImageMat = cv::Mat(originalHeight, originalWidth, CV_32FC3);
    // for (int c = 0; c < 3; ++c) {
    //     for (int i = 0; i < originalHeight; ++i) {
    //         for (int j = 0; j < originalWidth; ++j) {
    //             saveImageMat.at<cv::Vec3f>(i, j)[c] = saveImageVecT[c][i][j];
    //         }
    //     }
    // }

    // // normalize saveImageMat to [0, 255]
    // saveImageMat.convertTo(saveImageMat, CV_8UC3, 255.0f, 0);

    // // convert RGB to BGR
    // cv::cvtColor(saveImageMat, saveImageMat, cv::COLOR_RGB2BGR);

    // // save image
    // cv::imwrite(outputImage, saveImageMat);

    // // // print outputMat shape
    // // std::cout << "OutputMat shape: " << outputMat.size() << std::endl;
    // // std::cout << "OutputMat channels: " << outputMat.channels() << std::endl;
    
    // // // normalize output to [0, 255]
    // // outputMat.convertTo(outputMat, CV_8UC3, 255.0f, 0);

    // // // convert from RGB to BGR
    // // cv::cvtColor(outputMat, outputMat, cv::COLOR_RGB2BGR);

    // // // resize output
    // // cv::resize(outputMat, outputMat, cv::Size(originalWidth, originalHeight));
    
    // // // save output
    // // cv::imwrite(outputImage, outputMat);
    // return 0;
}