#!/usr/bin/env python3
"""
Script to migrate from conda environment.yml to Poetry pyproject.toml
"""

import yaml
import subprocess
import sys
from pathlib import Path

# Mapping from conda package names to PyPI package names
CONDA_TO_PYPI_MAPPING = {
    'pytorch': 'torch',
    'pytorch-cuda': None,  # Skip CUDA-specific packages
    'pytorch-mutex': None,
    'torchtriton': None,
    'pyg': 'torch-geometric',
    'pytorch-scatter': 'torch-scatter',
    'python-dateutil': 'python-dateutil',
    'python-tzdata': 'pytz',
    'python_abi': None,
    'libgcc-ng': None,  # Skip system libraries
    'libstdcxx-ng': None,
    'cuda-cudart': None,
    'cuda-cupti': None,
    'cuda-libraries': None,
    'cuda-nvrtc': None,
    'cuda-nvtx': None,
    'cuda-runtime': None,
    '_libgcc_mutex': None,
    '_openmp_mutex': None,
}

# Packages that should be skipped (system libs, etc.)
SKIP_PACKAGES = {
    'blas', 'mkl', 'mkl-service', 'mkl_fft', 'mkl_random',
    'intel-openmp', 'tbb',
    'libcublas', 'libcufft', 'libcufile', 'libcurand', 
    'libcusolver', 'libcusparse', 'libnpp', 'libnvjpeg',
    'ca-certificates', 'certifi', 'openssl',
    'ld_impl_linux-64', 'libgcc-ng', 'libstdcxx-ng',
    'libgomp', 'libgfortran-ng', 'libgfortran5',
    'ncurses', 'readline', 'sqlite', 'tk', 'xz', 'zlib',
    'libffi', 'libuuid', 'expat', 'libxml2', 'libxslt',
    'pcre', 'glib', 'dbus', 'fontconfig', 'freetype',
    'libbrotlicommon', 'libbrotlidec', 'libbrotlienc',
    'brotli', 'brotli-bin', 'bzip2', 'c-ares', 'cyrus-sasl',
    'gmp', 'gmpy2', 'mpc', 'mpfr', 'nettle', 'gnutls',
    'krb5', 'libedit', 'libevent', 'libpng', 'libpq',
    'libprotobuf', 'libsodium', 'libtasn1', 'libtiff',
    'libunistring', 'libwebp', 'libwebp-base', 'libxcb',
    'libxkbcommon', 'lightning-utilities', 'lz4-c',
    'nspr', 'nss', 'tzdata'
}

def load_environment_yml(file_path):
    """Load environment.yml file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def extract_packages(env_data):
    """Extract package names and versions from environment data"""
    packages = []
    
    # Process conda dependencies
    for dep in env_data.get('dependencies', []):
        if isinstance(dep, str):
            # Simple package specification
            if '=' in dep:
                name = dep.split('=')[0]
                version = dep.split('=')[1]
            else:
                name = dep
                version = None
            packages.append(('conda', name, version))
        elif isinstance(dep, dict) and 'pip' in dep:
            # Pip dependencies
            for pip_dep in dep['pip']:
                if '==' in pip_dep:
                    name = pip_dep.split('==')[0]
                    version = pip_dep.split('==')[1]
                else:
                    name = pip_dep
                    version = None
                packages.append(('pip', name, version))
    
    return packages

def filter_packages(packages):
    """Filter out system packages and map conda names to PyPI names"""
    filtered = []
    
    for source, name, version in packages:
        # Skip system packages
        if name in SKIP_PACKAGES:
            continue
            
        # Skip packages that start with lib (usually system libs)
        if name.startswith(('lib', 'cuda-', '_')):
            continue
            
        # Map conda names to PyPI names
        if source == 'conda':
            pypi_name = CONDA_TO_PYPI_MAPPING.get(name, name)
            if pypi_name is None:
                continue
            name = pypi_name
        
        filtered.append((name, version))
    
    return filtered

def add_packages_to_poetry(packages):
    """Add packages to Poetry"""
    for name, version in packages:
        if version:
            # Convert conda version format to poetry format
            if version.startswith('='):
                version = version[1:]
            package_spec = f"{name}^{version}"
        else:
            package_spec = name
        
        print(f"Adding {package_spec}...")
        try:
            result = subprocess.run([
                'poetry', 'add', package_spec
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Failed to add {package_spec}: {result.stderr}")
                # Try without version constraint
                print(f"Trying to add {name} without version constraint...")
                subprocess.run(['poetry', 'add', name], check=False)
        except subprocess.TimeoutExpired:
            print(f"Timeout adding {package_spec}, skipping...")
        except Exception as e:
            print(f"Error adding {package_spec}: {e}")

def main():
    """Main migration function"""
    env_file = Path('environment.yml')
    
    if not env_file.exists():
        print("environment.yml not found!")
        sys.exit(1)
    
    print("Loading environment.yml...")
    env_data = load_environment_yml(env_file)
    
    print("Extracting packages...")
    packages = extract_packages(env_data)
    
    print("Filtering packages...")
    filtered_packages = filter_packages(packages)
    
    print(f"Found {len(filtered_packages)} packages to migrate:")
    for name, version in filtered_packages:
        version_str = f" ({version})" if version else ""
        print(f"  - {name}{version_str}")
    
    print("\nStarting migration to Poetry...")
    add_packages_to_poetry(filtered_packages)
    
    print("\nMigration complete!")
    print("\nNote: Some packages (especially MNE-related) might need to be installed via conda:")
    print("conda install -c conda-forge mne mne-connectivity mne-icalabel")

if __name__ == "__main__":
    main()