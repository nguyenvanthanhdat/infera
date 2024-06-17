# Cinnamon Runtime
Cross-platform C++ high performance neural network inference library for computer vision, natural language processing and content generative tasks from edge devices to cloud servers.

The library is based on Microsoft Onnxruntime, Tencent NCNN, OpenCV, Tokenizers and more. 

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=plastic&logo=c%2B%2B&logoColor=white)
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=plastic&logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=plastic&logo=python&logoColor=ffdd54)
![C#](https://img.shields.io/badge/c%23-%23239120.svg?style=plastic&logo=csharp&logoColor=white)
![Android](https://img.shields.io/badge/Android-3DDC84?style=plastic&logo=android&logoColor=white)
![iOS](https://img.shields.io/badge/iOS-000000?style=plastic&logo=ios&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-%2302569B.svg?style=plastic&logo=Flutter&logoColor=white)
![React Native](https://img.shields.io/badge/react_native-%2320232a.svg?style=plastic&logo=react&logoColor=%2361DAFB)

## License
There are two licensing options to accommodate diverse use case:

* **AGPL-3.0 License**:
This OSI-approved open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](LICENSE) file for more details.

* **Enterprise License**:
Designed for commercial use, this license permits seamless integration into commercial goods and services, bypassing the open-source requirements of AGPL-3.0.

Copyright &copy; 2024 [Hieu Pham](https://github.com/hieupth). All rights reserved.

## Development

### Project structure
The project structure is compatible with CMakeList.txt file to make that internal source file can simple #include "something.h" while other libraries will #include <cinnamon/something.h> when this library is linked/installed.
```
.
+- docs               // Documents
+- include/cinnamon   // Public header files
+- lib                // 3rd libraries
+- cmake              // Cmake config files
+- src                // Source code tree
|  +- headers         // Header files
|  +- modules         // Platform-independent source
|  +- blindings       // Bindings to other languages
|  |  +- rust
|  |  +- python
|  |  +- ...
|  +- platforms       // Platform-specific code
|  |  +- ios
|  |  +- android
|  |  +- ...
+- tests              // Automated test scripts
+- tools              // Development utilities
```

### Install Dependencies
Most dependencies should be built from source to be used with C/C++, and they can be built as conda packages using [Rattler-Build](https://prefix-dev.github.io/rattler-build/latest/). The build recipe file for each library can be found at **tools/deps/[library-name]/recipe.yaml**.