

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'triangle_hash',
    sources=[
        'triangle_hash.pyx'
    ],
    #libraries=['m']  # Unix-like specific
)

ext_modules = [
    triangle_hash_module,
]

setup(
    ext_modules=cythonize(ext_modules),
)
