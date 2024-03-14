# `srctools`

[![Documentation Status](https://readthedocs.org/projects/srctools/badge/?version=latest)](https://srctools.readthedocs.io/en/latest/?badge=latest)

Modules for working with Valve's Source Engine file formats, as well as a 
variety of tools using these.

## Installation
Simply `pip install srctools`, you'll need Python 3.8+.

## Core Modules:

* `math`: Core vector, angles and matrix classes, with Cython accelerated versions.
* `cmdseq`: Reads and writes Hammer's expert compile command list format.
* `filesys`: Allows accessing VPKs, zips, folders etc using a unified interface, 
as well as a prioritised chain like the engine's game folder system.
* `logger`: Wrappers around the `logging` module allowing `str.format` interpolation support, among others.
* `const`: Various shared constants and enums.
* `tokenizer`: Cython-accelerated tokenizer for parsing the various text files.
* `binformat`: Some tools for handling binary file formats.

## File formats:
* `keyvalues`: Reads and writes KeyValues1 property trees.
* `dmx`: Reads and writes DMX / KeyValues2 format files.
* `vmf`: Reads and writes VMF map files.
* `bsp`: Reads and writes compiled BSP maps files. 
* `fgd`: Reads and writes FGD entity definitions. 
A compressed database of definitions from most games is also included, from [HammerAddons]. 
Note that this parses a superset of the FGD format, including "tags" to allow specifying which entities and keyvalues are supported for different engine branches.
* `mdl`: Reads some parts of compiled MDL/VTX/VVD/PHY models.
* `smd`: Reads and writes SMD geometry data.
* `sndscript`: Reads and writes soundscripts.
* `vmt`: Reads and writes VMT material files.
* `vpk`: Reads and writes VPK packages.
* `vtf`: Reads and writes VTF images, including DXT compression.
* `particles`: Reads and writes PCF particle systems.

## Tools:
* `game`: Parses `gameinfo.txt` files, and handles accessing the searchpaths.
* `instancing`: Implements logic for collapsing `func_instance` into maps.
* `packlist`: Stores a list of files of various types, then computes dependencies recursively. 
This also includes a database of resources required by game code for different entity classes.
* `run`: Code to run a compiler, logging the output as it executes while still storing it.

[HammerAddons]: https://github.com/TeamSpen/HammerAddons
