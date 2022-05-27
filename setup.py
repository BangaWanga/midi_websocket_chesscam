from setuptools import setup, find_packages

setup(
    name="rz-hub",
    version="0.2.2",
    packages=find_packages(),
    install_requires=['websockets', "numpy", "opencv-python", "rtmidi", "scikit-learn"],
)
