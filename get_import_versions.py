import importlib
import_list = ['numpy', 'pandas', 'scanpy', 'sklearn', 'scipy']

for pkg_name in import_list:
    pkg = importlib.import_module(pkg_name)
    print(f"{pkg_name} == {pkg.__version__}")
