"""Parse the 'Quake C' configuration files used for compiling models."""
from enum import Enum, Flag
from typing import List, Dict, Tuple, Iterator, IO
from pathlib import PurePath
from contextlib import ExitStack

from srctools.filesys import File, RawFileSystem
from srctools.tokenizer import Tokenizer, IterTokenizer, Token


class Contents(Flag):
    """$contents specifies the kind of collision for a model."""
    notsolid = 0x0
    solid = 0x1
    grate = 0x8
    monster = 0x2000000
    ladder = 0x20000000


def process_includes(
    stack: ExitStack,
    file: File,
) -> Iterator[Tuple[Token, str]]:
    """Implement includes handling."""
    fsys = file.sys
    fileobj = file.open_str()
    path = PurePath(file.path).parent
    stack.push(fileobj)
    tok = Tokenizer(fileobj, allow_escapes=False, allow_star_comments=True)
    tok_stack: List[Tuple[Tokenizer, IO[str], PurePath]] = [
        (tok, fileobj, path)
    ]

    while tok_stack:
        tok_type, tok_value = tok_and_value = tok()
        if tok_type is Token.EOF:
            # Reached the end, return to the last file.
            tok, fileobj, path = tok_stack.pop()
            fileobj.close()
            if not tok_stack:
                return
            tok, fileobj, path = tok_stack[-1]
        elif tok_type is Token.STRING and tok_value.casefold() == '$include':
            rel_path = tok.expect(Token.STRING)
            path /= rel_path
            try:
                file = fsys[str(path)]
            except FileNotFoundError:
                raise tok.error('Include file "{}" does not exist!', rel_path)
            fileobj = file.open_str()
            stack.push(fileobj)
            tok = Tokenizer(fileobj, allow_escapes=False, allow_star_comments=True)
            tok_stack.append((tok, fileobj, path))
        else:
            yield tok_and_value


def process_macros(
    tok_source: Iterator[Tuple[Token, str]],
    variables: Dict[str, str],
    macros: Dict[str, object],
) -> Iterator[Tuple[Token, str]]:
    """Handle macro and variables."""
    yield from tok_source


class QC:
    """A QC file represents an uncompiled model."""
    def __init__(self) -> None:
        self.model_name = ''
        self.cdmaterials: List[str] = []
        self.surfaceprop = 'default'
        self.contents = Contents.solid
        self.included_models: List[str] = []
        # Original material -> list of replacements
        self.skins: Dict[str, List[str]] = {}

        # Variables and macros defined during parsing.
        self.variables: Dict[str, str] = {}
        self.macros: Dict[str, object] = {}

    @classmethod
    def parse(cls, file: File) -> 'QC':
        """Parse a QC file."""
        qc = cls()
        with file.sys, ExitStack() as stack:
            tok = IterTokenizer(
                process_macros(
                    process_includes(stack, file),
                    qc.variables, qc.macros,
                ))
            for tok_type, tok_value in tok:
                pass

        return qc
