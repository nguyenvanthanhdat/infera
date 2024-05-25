import os
import sys
import tarfile
import requests
import argparse
import platform
from tqdm import tqdm
from pathlib import Path


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


def ort_cuda_patch(_os: str, cuda: str):
  """
  Patch Onnxruntime CUDA errors.
  :param _os:   OS name.
  """
  cmd = []
  # Patch cublas linking errors.
  _include = os.path.join('${CONDA_PREFIX}', 'include')
  _missing = os.path.join('${CONDA_PREFIX}', 'targets', '$(arch)')
  _missing = os.path.join(f'{_missing}-{_os.lower()}', 'include')
  os.system(f'mkdir -p {_missing}')
  _ln_cmd = f'ln -s {_include} {_missing}'
  os.system(_ln_cmd)
  # Disable types.
  _cu = [int(x) for x in cuda.split('.')]
  if _cu[0] < 12 and _cu[1] <= 8:
    cmd.append('--disable_types float8')
  # Return command.
  return ' '.join(cmd)


def ort(_os: str, deps: Path, args):
  """
  Build and install OnnxRuntime from source.
  :param ver:     desired version
  :param dir:     deps directory.
  :param _os:     operation system name.
  :param args:    additional arguments.
  :param kwds:    additional keyword arguments.
  """
  print('Install OnnxRuntime from source...')
  url = f'https://github.com/microsoft/onnxruntime/archive/refs/tags/v{args.v}.tar.gz'
  tar = deps.joinpath(f'onnxruntime-{args.v}.tar.gz')
  dir = deps.joinpath(f'onnxruntime-{args.v}').resolve()
  build = deps.joinpath(f'onnxruntime-{args.v}', 'build', _os, 'RelWithDebInfo').resolve()
  # Download and extract source.
  download(f'Download {url}', url, tar)
  tarextract(tar, deps)
  # Build command.
  cmd = [
    f'cd {dir} &&',
    f'bash ./build.sh --config RelWithDebInfo --build_shared_lib --build_wheel --parallel {os.cpu_count()} --compile_no_warning_as_error --skip_submodule_sync'
  ]
  # Skip tests.
  if args.skip_tests:
    cmd.append('--skip_tests')
  # Enable CUDA.
  if args.gpu and _os != 'MacOS':
    cmd.append('--use_cuda')
    if args.conda_cuda:
      _libs = ' '.join([f'{x}~={args.cuda}' for x in ['cuda-runtime', 'cuda-nvcc', 'cuda-cudart', 'cuda-libraries', 'cuda-libraries-dev']])
      os.system(f'mamba install -y {_libs} cudnn -c nvidia')
      cmd.append(f'--cuda_version {args.cuda} {ort_cuda_patch(_os, args.cuda)}')
      cmd.append('--cuda_home ${CONDA_PREFIX} --cudnn_home ${CONDA_PREFIX}')
  # Cmake extras.
  cmd.append(' '.join([
    f'--cmake_extra_defines',
    f'CMAKE_OSX_ARCHITECTURES=$(arch)',
    f'CMAKE_CUDA_ARCHITECTURES="75;80;86"',
    f'CMAKE_INSTALL_PREFIX=$CONDA_PREFIX',
    f'CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/cmake',
    f'&& cd {build} && make -j {os.cpu_count()} && pip install dist/* && make install'
  ]))
  os.system(' '.join(cmd))


parser = argparse.ArgumentParser('Cinnamon Runtime install dependencies')
subparsers = parser.add_subparsers(help='deps')
#
ortparser = subparsers.add_parser('ort')
ortparser.add_argument('--v', default='1.18.0', help='Onnxruntime version.')
ortparser.add_argument('--gpu', default=False, help='Enable GPU.', action=argparse.BooleanOptionalAction)
ortparser.add_argument('--cuda', default='11.8', help='CUDA version')
ortparser.add_argument('--cudnn', default='8', help='CuDNN version.')
ortparser.add_argument('--conda_cuda', default=False, help='Use conda CUDA.', action=argparse.BooleanOptionalAction)
ortparser.add_argument('--skip_tests', default=True, help='Skip tests', action=argparse.BooleanOptionalAction)
ortparser.set_defaults(func=ort)
#
cvparser = subparsers.add_parser('cv')
cvparser.add_argument('--v', default='4.9.0', help='OpenCV version.')

if __name__ == '__main__':
  args = parser.parse_args()
  # Get OS name.
  _os = platform.system()
  _os = 'MacOS' if _os == 'Darwin' else _os
  # Make deps directory.
  _deps = Path(os.getcwd(), '_deps')
  _deps.mkdir(parents=True, exist_ok=True)
  # Run.
  args.func(_os, _deps, args)
