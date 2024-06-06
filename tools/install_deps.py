import os
import sys
import shutil
import tarfile
import requests
import argparse
import platform
from tqdm import tqdm
from pathlib import Path


# Get OS name.
OS = platform.system()
OS = 'MacOS' if OS == 'Darwin' else OS
# Make deps directory.
DEPS_DIR = Path(os.getcwd(), '_deps')
DEPS_DIR.mkdir(parents=True, exist_ok=True)


def download(desc: str, url: str, out: str):
  """
  Download file.
  :param desc:  description info.
  :param url:   URL to be downloaded.
  :param out:   output file path.
  """
  if not os.path.isfile(out):
    res = requests.get(url=url, stream=True)
    bsize, tsize = 1024, int(res.headers.get('content-length', 0))
    with tqdm(desc=desc, total=tsize, unit='B', unit_scale=True) as pbar:
      with open(out, 'wb') as file:
        for data in res.iter_content(bsize):
          pbar.update(len(data))
          file.write(data)
    if tsize != 0 and pbar.n != tsize:
      raise RuntimeError('Could not download file.')


def tarextract(src: str, dst: str):
  """
  Extract TAR file.
  :param src:   source file to be extracted.
  :param dst:   extract destination.
  """
  with tarfile.open(src) as tar:
    tar.extractall(dst)


def onnxruntime(args):
  """
  Build and install OnnxRuntime from source.
  :param args:    additional arguments.
  """
  print('Install OnnxRuntime from source...')
  release = 'Release'
  url = f'https://github.com/microsoft/onnxruntime/archive/refs/tags/v{args.version}.tar.gz'
  tar = DEPS_DIR.joinpath(f'onnxruntime-{args.version}.tar.gz')
  dir = DEPS_DIR.joinpath(f'onnxruntime-{args.version}').resolve()
  build = DEPS_DIR.joinpath(f'onnxruntime-{args.version}', 'build', OS, release).resolve()
  # Download and extract source.
  download(f'Download {url}', url, tar)
  tarextract(tar, DEPS_DIR)
  if build.is_dir():
    shutil.rmtree(build)
  build.mkdir(parents=True, exist_ok=True)
  # Build command.
  build_command = [
    f'cd {dir} &&',
    f'bash ./build.sh --config {release} --build_shared_lib --build_wheel --compile_no_warning_as_error --skip_submodule_sync --allow_running_as_root',
    f'--parallel {int(os.cpu_count() / 2)} --nvcc_threads 1'
  ]
  if not args.tests:
      build_command.append('--skip_tests')
  if args.openvino:
    os.system('mamba install -y --no-update-deps openvino')
    build_command.append(f'--use_openvino {args.openvino_hardware}')
  if args.cuda:
    os.system(' '.join([
      'mamba install -y cudnn',
      f'cuda-nvcc~={args.cuda_version}', 
      f'cuda-cudart~={args.cuda_version}', 
      f'cuda-runtime~={args.cuda_version}', 
      f'cuda-libraries~={args.cuda_version}', 
      f'cuda-libraries-dev~={args.cuda_version}', 
      '-c nvidia'
    ]))
    build_command.append(f'--use_cuda --cuda_version {args.cuda_version}')
    build_command.append('--cuda_home ${CONDA_PREFIX} --cudnn_home ${CONDA_PREFIX}')
    if len(args.cuda_disable_types) > 0:
      build_command.append(f'--disable_types {args.cuda_disable_types}')
  # Cmake extras.
  build_command.append(' '.join([
    '--cmake_extra_defines',
      'CMAKE_OSX_ARCHITECTURES=$(arch)',
      'CMAKE_CUDA_ARCHITECTURES="75;80;86"',
      'CMAKE_INSTALL_PREFIX=${CONDA_PREFIX}',
      'CMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/cmake',
    f'&& cd {build} && make -j {os.cpu_count()} && pip install dist/* && make install'
  ]))
  # Run build command.
  os.system(' '.join(build_command))


def opencv(args):
  """
  Build and install OpenCV from source.
  :param args:    additional arguments.
  """
  # Download and extract source.
  url = f'https://github.com/opencv/opencv/archive/refs/tags/{args.version}.tar.gz'
  tar = DEPS_DIR.joinpath(f'opencv-{args.version}.tar.gz')
  dir = DEPS_DIR.joinpath(f'opencv-{args.version}')
  build = dir.joinpath('build')
  download(f'Download OpenCV', url, tar)
  tarextract(tar, DEPS_DIR)
  if build.is_dir():
    shutil.rmtree(build)
  build.mkdir(parents=True, exist_ok=True)
  # Download and extract contrib source.
  contrib_url = f'https://github.com/opencv/opencv_contrib/archive/refs/tags/{args.version}.tar.gz'
  contrib_tar = DEPS_DIR.joinpath(f'opencv-contrib-{args.version}.tar.gz')
  contrib_dir = DEPS_DIR.joinpath(f'opencv_contrib-{args.version}', 'modules')
  download(f'Download OpenCV Contrib', contrib_url, contrib_tar)
  tarextract(contrib_tar, DEPS_DIR)
  # Build command.
  cmd = [
    f'cd {build} &&',
    'cmake ..',
      f'-DOPENCV_EXTRA_MODULES_PATH={contrib_dir}',
      '-DBUILD_TESTS=OFF',
      '-DBUILD_PERF_TESTS=OFF',
      '-DBUILD_EXAMPLES=OFF',
      '-DBUILD_opencv_apps=OFF',
      '-DWITH_FFMPEG=ON -DWITH_VA=OFF',
      '-DPYTHON_DEFAULT_EXECUTABLE=$(which python)',
      '-DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}',
      '-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/cmake'
    f'&& make -j {os.cpu_count()} && make install'
  ]
  os.system(' '.join(cmd))


parser = argparse.ArgumentParser('Cinnamon Runtime install dependencies')
subparsers = parser.add_subparsers(help='deps')
# OnnxRuntime.
ortparser = subparsers.add_parser('onnxruntime')
ortparser.add_argument('--version', default='1.18.0', help='Onnxruntime version.')
ortparser.add_argument('--openvino', default=False, help='Enable OpenVINO', action=argparse.BooleanOptionalAction)
ortparser.add_argument('--openvino_hardware', default='CPU', help='OpenVINO hardwares')
ortparser.add_argument('--cuda', default=False, help='Enable CUDA.', action=argparse.BooleanOptionalAction)
ortparser.add_argument('--cuda_version', default='11.8', help='CUDA version.')
ortparser.add_argument('--cuda_disable_types', default='float8', help='CUDA disbale types.')
ortparser.add_argument('--tests', default=False, help='Build tests.', action=argparse.BooleanOptionalAction)
ortparser.set_defaults(func=onnxruntime)
# OpenCV.
cvparser = subparsers.add_parser('opencv')
cvparser.add_argument('--version', default='4.9.0', help='OpenCV version.')
cvparser.set_defaults(func=opencv)


if __name__ == '__main__':
  args = parser.parse_args()
  args.func(args)
