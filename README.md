# Cinnamon Runtime
Cross-platform ML/DL inference library optimized for edge computer vision and natural language processing tasks. 

## Development Setup
Use conda/mamba environment for package compatible resolved.
```bash
mamba env create -f environment.yml
mamba activate cinnamon
# See detail and choose installation.
python install_deps.py -h
```
**Notice:** remove your system installed CUDA and CuDNN because they may cause errors when build for GPUs.
```bash
sudo apt-get remove --auto-remove nvidia-cuda-toolkit
```
## License
There are two licensing options to accommodate diverse use case:

* **AGPL-3.0 License**:
This OSI-approved open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](LICENSE) file for more details.

* **Enterprise License**:
Designed for commercial use, this license permits seamless integration into commercial goods and services, bypassing the open-source requirements of AGPL-3.0.

Copyright &copy; 2024 [Hieu Pham](https://github.com/hieupth). All rights reserved.