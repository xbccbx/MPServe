from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "kernel"))


def _k(name: str) -> str:
    return os.path.join(_KERNEL_DIR, name)

ext_modules = [
    CUDAExtension(
        name='MPServe_cuda',
        sources=[
            _k('build_huffman.cu'),
            _k('decode_huffman.cu'),
            _k('decode_huffman_encode_only.cu'),
            _k('awq_missing_stubs.cpp'),
            _k('wrapper.cpp'),
        ],
        include_dirs=[
            _KERNEL_DIR,
            pybind11.get_cmake_dir()
        ],
        extra_compile_args={
            'cxx': ['-O3', '-DNDEBUG'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode=arch=compute_90,code=sm_90',
            ]
        }
    )
]

setup(
    name='trie_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)