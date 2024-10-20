"""Parse out all material proxies used in the given game."""
from typing import Counter, DefaultDict, List
import argparse
import sys
import traceback

from srctools.filesys import FileSystem, RawFileSystem
from srctools.game import Game
from srctools.vmt import Material


def main(args: List[str]) -> None:
    """Parse out all material proxies used in the given game."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "game",
        help="Either location of a gameinfo.txt file, or a game folder."
    )

    result = parser.parse_args(args)
    fsys: FileSystem
    try:
        fsys = Game(result.game).get_filesystem()
    except FileNotFoundError:
        fsys = RawFileSystem(result.game)

    # Shader/proxy -> parameter -> use count
    shader_params = DefaultDict[str, Counter[str]](Counter)
    shader_proxies = DefaultDict[str, Counter[str]](Counter)

    for file in fsys.walk_folder('materials/'):
        if not file.path.endswith('.vmt'):
            continue

        print('.', end='', flush=True)
        try:
            with file.open_str() as f:
                mat = Material.parse(f)
            mat = mat.apply_patches(fsys)
        except Exception:
            traceback.print_exc()
            continue

        param_count = shader_params[mat.shader.casefold()]
        for name in mat:
            param_count[name.casefold()] += 1

        for prox in mat.proxies:
            param_count = shader_proxies[prox.name]
            for prop in prox:
                param_count[prop.name] += 1

    print('\n\nShaders:')
    for shader in sorted(shader_params):
        print(f'"{shader.title()}"\n\t{{')
        param_count = shader_params[shader]
        for param_name in sorted(param_count):
            print(f'\t{param_name} = {param_count[param_name]}')
        print('\t}')

    print('\n\nProxies:')
    for proxy in sorted(shader_proxies):
        print(f'"{proxy.title()}"\n\t{{')
        param_count = shader_proxies[proxy]
        for param_name in sorted(param_count):
            print(f'\t{param_name} = {param_count[param_name]}')
        print('\t}')


if __name__ == '__main__':
    main(sys.argv[1:])
