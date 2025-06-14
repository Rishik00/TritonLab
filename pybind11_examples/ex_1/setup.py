import pybind11
import argparse
import sysconfig
import subprocess
import shlex

def run_bind_cmd(file_name: str, output_name: str):
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    include_pybind = pybind11.get_include()
    include_python = sysconfig.get_paths()['include']

    cmd = f"g++ -O3 -Wall -shared -std=c++11 -fPIC " \
          f"{file_name} -o {output_name}{ext_suffix} " \
          f"-I{include_pybind} -I{include_python}"

    print(f"Running:\n{cmd}")
    subprocess.run(shlex.split(cmd), check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile C++ file with pybind11")
    parser.add_argument("f", type=str, help="C++ source file to compile")
    parser.add_argument("o", type=str, help="Output module name (no extension)")
    args = parser.parse_args()

    run_bind_cmd(args.f, args.o)
