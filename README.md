# Cinnamon Runtime
Cross-platform ML/DL inference library optimized for edge computer vision and natural language processing tasks. 

---
## Benchmark Model's Inference
### Create environment:
```
$ mamba env create -f environment.yml
$ mamba activate cinnamon

# default install cuda 11.8 and opencv 4.9.0
$ python install_deps.py 
```
### Build Prerequisites:
```
# Build google benchmark
$ git clone https://github.com/google/benchmark.git
$ cd benchmark

# clone googletest
$ git clone https://github.com/google/googletest.git

# Make a build directory to place the build output.
$ cmake -E make_directory "build"

# Generate build system files with cmake, and download any dependencies.
$ cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../

# Build the library.
$ cmake --build "build" --config Release
```

### Build source
```
# comeback to root source
$ cd ..

# create folder build
$ mkdir build

# build from source
$ cd build
$ cmake ../
```

### Run benchmark
**Setup config for onnxruntime session**
First, at the end of files `*.cpp` in folder `scripts/`, you will see the following code:
```
BENCHMARK_CAPTURE(
    BM_Function, 
    test_wb, 
    std::string("inter_intra"), 
    std::string("sequential")
)->ArgsProduct({
    { 0, 1, 2, 3},
    { 0, 1, 2, 3, 4 }
})->Unit(
    benchmark::kMillisecond
);
```
At the **third** and **fourth** arguments of function `BENCHMARK_CAPTURE()` corresponding to `threadOption` and `executionMode`, you can customize which value to put into. These are following available options:
* `std::string threadOption: [inter, intra, inter_intra, intra_inter]` 
* `const std::string executionMode: [parallel, sequential]`

Currently, benchmark is run on `optimizerLevel` and `numThreads` with following values:
* `int optimizerLevel: [0, 1, 2, 3]`
* `int numThreads: [0, 1, 2, 3, 4]`

**Make file and run benchmark**
```
$ make
$ ./runExample 
```
You can extend config benchmark for more test in [User_Guide](https://google.github.io/benchmark/user_guide.html)

**Test with more models**
1. In line 23 in file `./CMakeLists.txt`, change to any custom file `scripts/*.cpp`
2. Ensure input, output shape of model `.onnx` and path to model 


---

## License
There are two licensing options to accommodate diverse use case:

* **AGPL-3.0 License**:
This OSI-approved open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](LICENSE) file for more details.

* **Enterprise License**:
Designed for commercial use, this license permits seamless integration into commercial goods and services, bypassing the open-source requirements of AGPL-3.0.

Copyright &copy; 2024 [Hieu Pham](https://github.com/hieupth). All rights reserved.