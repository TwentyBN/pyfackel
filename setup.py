import os
import re
import sys
import platform
import subprocess

from setuptools import setup


MAC_TEMPLATE = ('http://download.pytorch.org/whl/'
                'torch-0.1.12.post2-{}m-macosx_10_7_x86_64.whl')
LINUX_TEMPLATE = ('http://download.pytorch.org/whl/{}/'
                  'torch-0.1.12.post2-{}m-linux_x86_64.whl')


class VersionError(Exception):
    pass


class UnsupportedPlatformException(Exception):
    pass


class InstallationError(Exception):
    pass


def get_cuda_placeholder():
    if os.system('which nvcc > /dev/null') == 0:
        output = subprocess.check_output('nvcc --version', shell=True)
        match = re.search(r'Cuda compilation tools, release (\d+\.\d+)',
                          output.decode('utf-8'))
        if match:
            version = 'cu' + ''.join(match.group(1).split('.'))
        else:
            raise VersionError('Could not parse cuda version')

        if version in ['cu75', 'cu80']:
            return version
        else:
            raise VersionError(
                'Binaries are only available for cuda 7.5 and 8.0')
    else:
        return 'cu75'


def get_python_version():
    version = ''.join(platform.python_version().split('.')[:2])
    if version in ['27', '35', '36']:
        return version
    else:
        raise VersionError('Only python versions 2.7, 3.5, 3.6 are supported')


def get_platform():
    platform_name = sys.platform
    if platform_name.startswith('linux'):
        return 'linux'
    elif platform_name.startswith('darwin'):
        return 'macosx'
    else:
        raise UnsupportedPlatformException(
            'Only linux and mac are supported but you are on {}'
            .format(platform_name))


def format_python_version(python_version):
    if python_version == '27':
        return 'cp27-none'
    elif python_version in ['35', '36']:
        return '-'.join(['cp{}'.format(python_version)]*2)


def get_torch_url():
    cuda_version = get_cuda_placeholder()
    python_version = get_python_version()
    platform = get_platform()
    python_placeholder = format_python_version(python_version)
    if platform == 'macosx':
        sys.stderr.write('No precompiled version with cuda support available')
        return MAC_TEMPLATE.format(python_placeholder)
    else:
        return LINUX_TEMPLATE.format(cuda_version, python_placeholder)


def install_pytorch():
    cmd = 'pip install -U {}'.format(get_torch_url())
    if os.system(cmd):
        raise InstallationError()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'install':
        install_pytorch()

    setup(name='pyfackel',
          version='0.1.12.2',
          description='meta package for installing pytorch',
          classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Developers',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'License :: OSI Approved :: MIT License',
              'Topic :: Scientific/Engineering :: Artificial Intelligence',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Software Development :: Libraries',
              'Topic :: Software Development :: Libraries :: Python Modules',
          ],
          license='MIT',
          url='https://github.com/TwentyBN/pyfackel',
          author='Ingo Fruend, Guillaume Berger',
          author_email='ingo.fruend@twentybn.com, gberger.eclille@gmail.com',
          )
