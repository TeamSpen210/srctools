# srctools

Modules for working with Valve's Source Engine file formats, as well as a 
variety of tools using these.


# Installation
First clone the git repo to your computer. You'll likely want the `dev` branch, 
a specific tag. You'll need Python 3.6+.

Run the following to install dependencies:
```shell script
pip install PyInstaller, cython, importlib_resources, aenum
```
To build the optional extension modules to speed up the code, run the following:
```shell script
python setup.py build_ext
```
Then run the following to build the postcompiler.
```shell script
pyinstaller postcompiler.spec
```
