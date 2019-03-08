from setuptools import setup, find_packages

setup(
    name="rz-hub",
    version="0.1",
    packages=find_packages(),
    install_requires = ['websockets','PyQt5',"pygame"],
)