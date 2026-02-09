#!/usr/bin/env python3
"""
MassScript: Domain-Specific Language for Partition Synthesis

MassScript is a declarative language for virtual mass spectrometry.
Programs are sequences of ternary operations that synthesize spectra
rather than simulate physical processes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum, auto
import re

from ternary_core import (
    TernaryAddress, PartitionState, SEntropyCoord,
    Spectrum, SpectrumExtractor, MoleculeEncoder
)


class TokenType(Enum):
    """MassScript token types."""
    PARTITION = auto()
    OBSERVE = auto()
    FRAGMENT = auto()
    EXTEND = auto()
    INJECT = auto()
    CHROMATOGRAPH = auto()
    IONIZE = auto()
    DETECT = auto()
    COMPLETE = auto()
    AT = auto()
    BY = auto()
    AS = auto()
    FROM = auto()
    TO = auto()
    ARROW = auto()
    IDENTIFIER = auto()
    TERNARY = auto()
    NUMBER = auto()
    STRING = auto()
    NEWLINE = auto()
    EOF = auto()


@dataclass
class Token:
    """MassScript token."""
    type: TokenType
    value: Any
    line: int
    column: int


class Lexer:
    """MassScript lexer."""

    KEYWORDS = {
        'partition': TokenType.PARTITION,
        'observe': TokenType.OBSERVE,
        'fragment': TokenType.FRAGMENT,
        'extend': TokenType.EXTEND,
        'inject': TokenType.INJECT,
        'chromatograph': TokenType.CHROMATOGRAPH,
        'ionize': TokenType.IONIZE,
        'detect': TokenType.DETECT,
        'complete': TokenType.COMPLETE,
        'at': TokenType.AT,
        'by': TokenType.BY,
        'as': TokenType.AS,
        'from': TokenType.FROM,
        'to': TokenType.TO,
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> List[Token]:
        """Tokenize source into token list."""
        tokens = []

        while self.pos < len(self.source):
            char = self.source[self.pos]

            # Skip whitespace (except newlines)
            if char in ' \t':
                self.advance()
                continue

            # Comments
            if char == '#':
                self.skip_comment()
                continue

            # Newlines
            if char == '\n':
                tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                self.line += 1
                self.column = 1
                continue

            # Arrow
            if char == '-' and self.peek() == '>':
                tokens.append(Token(TokenType.ARROW, '->', self.line, self.column))
                self.advance()
                self.advance()
                continue

            # Numbers
            if char.isdigit() and not self.is_ternary_context():
                tokens.append(self.read_number())
                continue

            # Ternary literals (sequences of 0, 1, 2)
            if char in '012' and self.is_ternary_context():
                tokens.append(self.read_ternary())
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                tokens.append(self.read_identifier())
                continue

            # Strings
            if char in '"\'':
                tokens.append(self.read_string())
                continue

            # Unknown character - skip
            self.advance()

        tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return tokens

    def advance(self):
        self.pos += 1
        self.column += 1

    def peek(self) -> Optional[str]:
        if self.pos + 1 < len(self.source):
            return self.source[self.pos + 1]
        return None

    def is_ternary_context(self) -> bool:
        """Check if we're expecting a ternary literal."""
        # Look back for keywords that precede ternary
        lookback = self.source[max(0, self.pos-20):self.pos].lower()
        return any(kw in lookback for kw in ['partition', 'extend by', 'as'])

    def skip_comment(self):
        while self.pos < len(self.source) and self.source[self.pos] != '\n':
            self.advance()

    def read_number(self) -> Token:
        start = self.pos
        start_col = self.column

        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] == '.'):
            self.advance()

        value = self.source[start:self.pos]
        if '.' in value:
            return Token(TokenType.NUMBER, float(value), self.line, start_col)
        return Token(TokenType.NUMBER, int(value), self.line, start_col)

    def read_ternary(self) -> Token:
        start = self.pos
        start_col = self.column

        while self.pos < len(self.source) and self.source[self.pos] in '012':
            self.advance()

        value = self.source[start:self.pos]
        return Token(TokenType.TERNARY, value, self.line, start_col)

    def read_identifier(self) -> Token:
        start = self.pos
        start_col = self.column

        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self.advance()

        value = self.source[start:self.pos]
        token_type = self.KEYWORDS.get(value.lower(), TokenType.IDENTIFIER)
        return Token(token_type, value, self.line, start_col)

    def read_string(self) -> Token:
        quote = self.source[self.pos]
        start_col = self.column
        self.advance()

        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos] != quote:
            self.advance()

        value = self.source[start:self.pos]
        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, value, self.line, start_col)


@dataclass
class ASTNode:
    """Base AST node."""
    line: int = 0


@dataclass
class PartitionNode(ASTNode):
    """partition <address>"""
    address: str = ""
    name: Optional[str] = None


@dataclass
class ObserveNode(ASTNode):
    """observe [name]"""
    target: Optional[str] = None


@dataclass
class FragmentNode(ASTNode):
    """fragment [name] at <position>"""
    target: Optional[str] = None
    position: int = 0


@dataclass
class ExtendNode(ASTNode):
    """extend by <trits>"""
    extension: str = ""


@dataclass
class InjectNode(ASTNode):
    """inject [name] as <address>"""
    name: Optional[str] = None
    address: str = ""


@dataclass
class ChromatographNode(ASTNode):
    """chromatograph extend by <trits>"""
    extension: str = ""


@dataclass
class IonizeNode(ASTNode):
    """ionize extend by <trits>"""
    extension: str = ""


@dataclass
class DetectNode(ASTNode):
    """detect"""
    pass


@dataclass
class CompleteNode(ASTNode):
    """complete trajectory"""
    pass


class Parser:
    """MassScript parser."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> List[ASTNode]:
        """Parse token stream into AST."""
        nodes = []

        while not self.at_end():
            if self.check(TokenType.NEWLINE):
                self.advance()
                continue

            node = self.parse_statement()
            if node:
                nodes.append(node)

        return nodes

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        if self.check(TokenType.PARTITION):
            return self.parse_partition()
        elif self.check(TokenType.OBSERVE):
            return self.parse_observe()
        elif self.check(TokenType.FRAGMENT):
            return self.parse_fragment()
        elif self.check(TokenType.EXTEND):
            return self.parse_extend()
        elif self.check(TokenType.INJECT):
            return self.parse_inject()
        elif self.check(TokenType.CHROMATOGRAPH):
            return self.parse_chromatograph()
        elif self.check(TokenType.IONIZE):
            return self.parse_ionize()
        elif self.check(TokenType.DETECT):
            return self.parse_detect()
        elif self.check(TokenType.COMPLETE):
            return self.parse_complete()
        else:
            # Skip unknown tokens
            self.advance()
            return None

    def parse_partition(self) -> PartitionNode:
        line = self.current().line
        self.advance()  # consume 'partition'

        address = ""
        name = None

        if self.check(TokenType.IDENTIFIER):
            name = self.advance().value
            if self.check(TokenType.TERNARY):
                address = self.advance().value
        elif self.check(TokenType.TERNARY):
            address = self.advance().value

        return PartitionNode(line=line, address=address, name=name)

    def parse_observe(self) -> ObserveNode:
        line = self.current().line
        self.advance()  # consume 'observe'

        target = None
        if self.check(TokenType.IDENTIFIER):
            target = self.advance().value

        return ObserveNode(line=line, target=target)

    def parse_fragment(self) -> FragmentNode:
        line = self.current().line
        self.advance()  # consume 'fragment'

        target = None
        if self.check(TokenType.IDENTIFIER):
            target = self.advance().value

        position = 0
        if self.check(TokenType.AT):
            self.advance()
            if self.check(TokenType.NUMBER):
                position = self.advance().value

        return FragmentNode(line=line, target=target, position=position)

    def parse_extend(self) -> ExtendNode:
        line = self.current().line
        self.advance()  # consume 'extend'

        extension = ""
        if self.check(TokenType.BY):
            self.advance()
            if self.check(TokenType.TERNARY):
                extension = self.advance().value

        return ExtendNode(line=line, extension=extension)

    def parse_inject(self) -> InjectNode:
        line = self.current().line
        self.advance()  # consume 'inject'

        name = None
        if self.check(TokenType.IDENTIFIER):
            name = self.advance().value

        address = ""
        if self.check(TokenType.AS):
            self.advance()
            if self.check(TokenType.TERNARY):
                address = self.advance().value

        return InjectNode(line=line, name=name, address=address)

    def parse_chromatograph(self) -> ChromatographNode:
        line = self.current().line
        self.advance()  # consume 'chromatograph'

        extension = ""
        if self.check(TokenType.EXTEND):
            self.advance()
            if self.check(TokenType.BY):
                self.advance()
                if self.check(TokenType.TERNARY):
                    extension = self.advance().value

        return ChromatographNode(line=line, extension=extension)

    def parse_ionize(self) -> IonizeNode:
        line = self.current().line
        self.advance()  # consume 'ionize'

        extension = ""
        if self.check(TokenType.EXTEND):
            self.advance()
            if self.check(TokenType.BY):
                self.advance()
                if self.check(TokenType.TERNARY):
                    extension = self.advance().value

        return IonizeNode(line=line, extension=extension)

    def parse_detect(self) -> DetectNode:
        line = self.current().line
        self.advance()  # consume 'detect'
        return DetectNode(line=line)

    def parse_complete(self) -> CompleteNode:
        line = self.current().line
        self.advance()  # consume 'complete'
        return CompleteNode(line=line)

    def check(self, token_type: TokenType) -> bool:
        if self.at_end():
            return False
        return self.current().type == token_type

    def advance(self) -> Token:
        if not self.at_end():
            self.pos += 1
        return self.tokens[self.pos - 1]

    def current(self) -> Token:
        return self.tokens[self.pos]

    def at_end(self) -> bool:
        return self.current().type == TokenType.EOF


@dataclass
class ExecutionResult:
    """Result of MassScript execution."""
    spectra: List[Spectrum] = field(default_factory=list)
    fragments: List[Tuple[TernaryAddress, TernaryAddress]] = field(default_factory=list)
    variables: Dict[str, TernaryAddress] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


class Interpreter:
    """MassScript interpreter."""

    def __init__(self):
        self.current_address = TernaryAddress(())
        self.variables: Dict[str, TernaryAddress] = {}
        self.extractor = SpectrumExtractor(m_ref=100.0)
        self.result = ExecutionResult()

    def execute(self, source: str) -> ExecutionResult:
        """Execute MassScript program."""
        # Lex
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Parse
        parser = Parser(tokens)
        ast = parser.parse()

        # Execute
        for node in ast:
            self.execute_node(node)

        return self.result

    def execute_node(self, node: ASTNode):
        """Execute single AST node."""
        if isinstance(node, PartitionNode):
            self.exec_partition(node)
        elif isinstance(node, ObserveNode):
            self.exec_observe(node)
        elif isinstance(node, FragmentNode):
            self.exec_fragment(node)
        elif isinstance(node, ExtendNode):
            self.exec_extend(node)
        elif isinstance(node, InjectNode):
            self.exec_inject(node)
        elif isinstance(node, ChromatographNode):
            self.exec_chromatograph(node)
        elif isinstance(node, IonizeNode):
            self.exec_ionize(node)
        elif isinstance(node, DetectNode):
            self.exec_detect(node)
        elif isinstance(node, CompleteNode):
            self.exec_complete(node)

    def exec_partition(self, node: PartitionNode):
        addr = TernaryAddress.from_string(node.address)
        self.current_address = addr
        if node.name:
            self.variables[node.name] = addr
        self.result.log.append(f"Partition: {addr}")

    def exec_observe(self, node: ObserveNode):
        if node.target and node.target in self.variables:
            addr = self.variables[node.target]
        else:
            addr = self.current_address

        spectrum = self.extractor.extract_spectrum(addr)
        self.result.spectra.append(spectrum)
        self.result.log.append(f"Observe: m/z={spectrum.mz:.4f}, RT={spectrum.retention_time:.2f}")

    def exec_fragment(self, node: FragmentNode):
        if node.target and node.target in self.variables:
            addr = self.variables[node.target]
        else:
            addr = self.current_address

        frag1, frag2 = addr.fragment_at(node.position)
        self.result.fragments.append((frag1, frag2))
        self.result.log.append(f"Fragment at {node.position}: {frag1} + {frag2}")

    def exec_extend(self, node: ExtendNode):
        extension = TernaryAddress.from_string(node.extension)
        self.current_address = self.current_address.extend(extension)
        self.result.log.append(f"Extend by {node.extension}: {self.current_address}")

    def exec_inject(self, node: InjectNode):
        addr = TernaryAddress.from_string(node.address)
        self.current_address = addr
        if node.name:
            self.variables[node.name] = addr
        self.result.log.append(f"Inject as {addr}")

    def exec_chromatograph(self, node: ChromatographNode):
        extension = TernaryAddress.from_string(node.extension)
        self.current_address = self.current_address.extend(extension)
        self.result.log.append(f"Chromatograph: {self.current_address}")

    def exec_ionize(self, node: IonizeNode):
        extension = TernaryAddress.from_string(node.extension)
        self.current_address = self.current_address.extend(extension)
        self.result.log.append(f"Ionize: {self.current_address}")

    def exec_detect(self, node: DetectNode):
        spectrum = self.extractor.extract_spectrum(self.current_address)
        self.result.spectra.append(spectrum)
        self.result.log.append(f"Detect: m/z={spectrum.mz:.4f}, RT={spectrum.retention_time:.2f}")

    def exec_complete(self, node: CompleteNode):
        # Trajectory completion: find minimal extension for determined spectrum
        self.result.log.append(f"Complete trajectory at: {self.current_address}")


def run(source: str) -> ExecutionResult:
    """Run MassScript program and return results."""
    interpreter = Interpreter()
    return interpreter.execute(source)


def run_file(path: str) -> ExecutionResult:
    """Run MassScript program from file."""
    with open(path, 'r') as f:
        source = f.read()
    return run(source)


# Example programs
EXAMPLE_PHOSPHOLIPID = """
# Phospholipid analysis
partition PC_34_1 201102012021

observe PC_34_1

fragment PC_34_1 at 6
"""

EXAMPLE_VIRTUAL_EXPERIMENT = """
# Virtual mass spectrometry experiment
inject sample as 000

chromatograph extend by 111011

ionize extend by 220

detect
"""


if __name__ == '__main__':
    print("MassScript Interpreter")
    print("=" * 60)

    print("\nExample 1: Phospholipid Analysis")
    print("-" * 40)
    result = run(EXAMPLE_PHOSPHOLIPID)
    for log in result.log:
        print(f"  {log}")
    if result.spectra:
        print(f"\nSpectrum: m/z={result.spectra[0].mz:.2f}, RT={result.spectra[0].retention_time:.2f}")

    print("\nExample 2: Virtual Experiment")
    print("-" * 40)
    result = run(EXAMPLE_VIRTUAL_EXPERIMENT)
    for log in result.log:
        print(f"  {log}")
    if result.spectra:
        print(f"\nSpectrum: m/z={result.spectra[0].mz:.2f}, RT={result.spectra[0].retention_time:.2f}")
