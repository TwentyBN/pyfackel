import unittest
try:
    from unittest import mock
except ImportError:
    import mock

import setup


class TestGetPlatform(unittest.TestCase):

    @mock.patch('sys.platform', 'linux-ANY_FURTHER_INFORMATION')
    def test_should_return_linux_on_linux(self):
        self.assertEqual(setup.get_platform(), 'linux')

    @mock.patch('sys.platform', 'darwin-ANY_FURTHER_INFORMATION')
    def test_should_return_mac_on_mac(self):
        self.assertEqual(setup.get_platform(), 'macosx')

    @mock.patch('sys.platform', 'ANY_OTHER_PLATFORM')
    def test_should_raise_on_other_platforms(self):
        with self.assertRaises(setup.UnsupportedPlatformException):
            setup.get_platform()


class TestGetPythonVersion(unittest.TestCase):

    @mock.patch('platform.python_version', return_value='2.7.1')
    def test_should_return_correct_version_for_py27(self, mock_py_version):
        self.assertEqual(setup.get_python_version(), '27')

    @mock.patch('platform.python_version', return_value='3.5.6')
    def test_should_return_correct_version_for_py35(self, mock_py_version):
        self.assertEqual(setup.get_python_version(), '35')

    @mock.patch('platform.python_version', return_value='3.6.1')
    def test_should_return_correct_version_for_py36(self, mock_py_version):
        self.assertEqual(setup.get_python_version(), '36')

    @mock.patch('platform.python_version', return_value='3.7.1')
    def test_should_raise_for_py37(self, mock_py_version):
        with self.assertRaises(setup.VersionError):
            setup.get_python_version()

    @mock.patch('platform.python_version', return_value='3.4.1')
    def test_should_raise_for_py34(self, mock_py_version):
        with self.assertRaises(setup.VersionError):
            setup.get_python_version()

    @mock.patch('platform.python_version', return_value='2.6.1')
    def test_should_raise_for_py26(self, mock_py_version):
        with self.assertRaises(setup.VersionError):
            setup.get_python_version()


class TestGetCudaPlaceholder(unittest.TestCase):

    @mock.patch('os.system')
    @mock.patch('subprocess.check_output')
    def test_should_return_none_if_no_cuda(self, mock_check_outp, mock_system):
        mock_system.return_value = 256

        version = setup.get_cuda_placeholder()

        self.assertEqual(version, 'cu75')
        mock_system.assert_called_once_with('which nvcc > /dev/null')
        mock_check_outp.assert_not_called()

    @mock.patch('os.system')
    @mock.patch('subprocess.check_output')
    def test_should_return_cu80_for_cuda80(self, mock_check_outp, mock_system):
        mock_system.return_value = 0
        mock_check_outp.return_value = (
            "nvcc: NVIDIA (R) Cuda compiler driver"
            "Copyright (c) 2005-2016 NVIDIA Corporation"
            "Built on Sun_Sep__4_22:14:01_CDT_2016"
            "Cuda compilation tools, release 8.0, V8.0.44")

        version = setup.get_cuda_placeholder()

        self.assertEqual(version, 'cu80')
        mock_system.assert_called_once_with('which nvcc > /dev/null')
        mock_check_outp.assert_called_once_with('nvcc --version', shell=True)

    @mock.patch('os.system')
    @mock.patch('subprocess.check_output')
    def test_should_return_cu75_for_cuda75(self, mock_check_outp, mock_system):
        mock_system.return_value = 0
        mock_check_outp.return_value = (
            "nvcc: NVIDIA (R) Cuda compiler driver"
            "Copyright (c) 2005-2016 NVIDIA Corporation"
            "Built on Sun_Sep__4_22:14:01_CDT_2016"
            "Cuda compilation tools, release 7.5, V7.5.xx")

        version = setup.get_cuda_placeholder()

        self.assertEqual(version, 'cu75')
        mock_system.assert_called_once_with('which nvcc > /dev/null')
        mock_check_outp.assert_called_once_with('nvcc --version', shell=True)

    @mock.patch('os.system')
    @mock.patch('subprocess.check_output')
    def test_should_raise_for_cuda70(self, mock_check_outp, mock_system):
        mock_system.return_value = 0
        mock_check_outp.return_value = (
            "nvcc: NVIDIA (R) Cuda compiler driver"
            "Copyright (c) 2005-2016 NVIDIA Corporation"
            "Built on Sun_Sep__4_22:14:01_CDT_2016"
            "Cuda compilation tools, release 7.0, V7.0.xx")

        with self.assertRaises(setup.VersionError):
            setup.get_cuda_placeholder()

        mock_system.assert_called_once_with('which nvcc > /dev/null')
        mock_check_outp.assert_called_once_with('nvcc --version', shell=True)


class TestGetTorchUrl(unittest.TestCase):

    @mock.patch('setup.get_python_version', mock.Mock())
    @mock.patch('setup.get_platform')
    @mock.patch('setup.format_python_version')
    @mock.patch('setup.get_cuda_placeholder')
    def test_mac(self, mock_cuda_pl, mock_pyversion, mock_platform):
        mock_cuda_pl.return_value = 'ANY_CUDA_VERSION'
        mock_pyversion.return_value = 'ANY_PYTHON_SPEC'
        mock_platform.return_value = 'macosx'

        url = setup.get_torch_url()
        self.assertEqual(
            url,
            'http://download.pytorch.org/whl/'
            'torch-0.1.12.post2-ANY_PYTHON_SPECm-macosx_10_7_x86_64.whl')

    @mock.patch('setup.get_python_version', mock.Mock())
    @mock.patch('setup.get_platform')
    @mock.patch('setup.format_python_version')
    @mock.patch('setup.get_cuda_placeholder')
    def test_linux(self, mock_cuda_pl, mock_pyversion, mock_platform):
        mock_cuda_pl.return_value = 'ANY_CUDA_VERSION'
        mock_pyversion.return_value = 'ANY_PYTHON_SPEC'
        mock_platform.return_value = 'linux'

        url = setup.get_torch_url()
        self.assertEqual(
            url,
            'http://download.pytorch.org/whl/ANY_CUDA_VERSION/'
            'torch-0.1.12.post2-ANY_PYTHON_SPECm-linux_x86_64.whl')
