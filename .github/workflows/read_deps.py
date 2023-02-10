"""Extract dependencies from pyproject.toml into requirements.txt"""
import tomllib

with open('pyproject.toml', 'rb') as src:
    pyproject = tomllib.load(src)
with open('pyproject-deps.txt', 'w') as dest:
    for req in pyproject['build-system']['requires']:
        dest.write(req + '\n')
    for req in pyproject['project']['dependencies']:
        dest.write(req + '\n')
