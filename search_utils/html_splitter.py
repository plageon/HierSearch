###
# modified from langchain
# https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/html.py
###

from __future__ import annotations

import copy
from dataclasses import dataclass
import pathlib
import re
from io import StringIO
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

import requests

import re
from typing import Any, List, Literal, Optional, Union

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from bs4 import Comment, BeautifulSoup

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")

@dataclass
class Document:
    """Document class to represent a text document with metadata."""

    page_content: str
    metadata: Dict[str, Any]


class TextSplitter():
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: Union[bool, Literal["start", "end"]] = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it
                            in each corresponding chunk (True='start')
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def create_documents(
        self, texts: list[str], metadatas: Optional[list[dict[Any, Any]]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
        """Text splitter that uses HuggingFace tokenizer to count length."""
        try:
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Tokenizer received was not an instance of PreTrainedTokenizerBase"
                )

            def _huggingface_tokenizer_length(text: str) -> int:
                return len(tokenizer.tokenize(text))

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
        return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @classmethod
    def from_tiktoken_encoder(
        cls: Type[TS],
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> TS:
        """Text splitter that uses tiktoken encoder to count length."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        if issubclass(cls, TokenTextSplitter):
            extra_kwargs = {
                "encoding_name": encoding_name,
                "model_name": model_name,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_tiktoken_encoder, **kwargs)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))


class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        """Splits the input text into smaller chunks based on tokenization.

        This method uses a custom tokenizer configuration to encode the input text
        into tokens, processes the tokens in chunks of a specified size with overlap,
        and decodes them back into text chunks. The splitting is performed using the
        `split_text_on_tokens` function.

        Args:
            text (str): The input text to be split into smaller chunks.

        Returns:
            List[str]: A list of text chunks, where each chunk is derived from a portion
            of the input text based on the tokenization and chunking rules.
        """

        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)


class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"
    HASKELL = "haskell"
    ELIXIR = "elixir"
    POWERSHELL = "powershell"


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: Callable[[List[int]], str]
    """ Function to decode a list of token ids to a string"""
    encode: Callable[[str], List[int]]
    """ Function to encode a string to a list of token ids"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(
        self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )
        splits = _split_text_with_regex(text, separator, self._keep_separator)
        _separator = "" if self._keep_separator else self._separator
        return self._merge_splits(splits, _separator)


def _split_text_with_regex(
    text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                (splits + [_splits[-1]])
                if keep_separator == "end"
                else ([_splits[0]] + splits)
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
        cls, language: Language, **kwargs: Any
    ) -> RecursiveCharacterTextSplitter:
        """Return an instance of this class based on a specific language.

        This method initializes the text splitter with language-specific separators.

        Args:
            language (Language): The language to configure the text splitter for.
            **kwargs (Any): Additional keyword arguments to customize the splitter.

        Returns:
            RecursiveCharacterTextSplitter: An instance of the text splitter configured
            for the specified language.
        """
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        """Retrieve a list of separators specific to the given language.

        Args:
            language (Language): The language for which to get the separators.

        Returns:
            List[str]: A list of separators appropriate for the specified language.
        """
        if language == Language.C or language == Language.CPP:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.GO:
            return [
                # Split along function definitions
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JAVA:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.KOTLIN:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\ncase ",
                "\nelse ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JS:
            return [
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PHP:
            return [
                # Split along function definitions
                "\nfunction ",
                # Split along class definitions
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PROTO:
            return [
                # Split along message definitions
                "\nmessage ",
                # Split along service definitions
                "\nservice ",
                # Split along enum definitions
                "\nenum ",
                # Split along option definitions
                "\noption ",
                # Split along import statements
                "\nimport ",
                # Split along syntax declarations
                "\nsyntax ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PYTHON:
            return [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RST:
            return [
                # Split along section titles
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                # Split along directive markers
                "\n\n.. *\n\n",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUBY:
            return [
                # Split along method definitions
                "\ndef ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.ELIXIR:
            return [
                # Split along method function and module definition
                "\ndef ",
                "\ndefp ",
                "\ndefmodule ",
                "\ndefprotocol ",
                "\ndefmacro ",
                "\ndefmacrop ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\ncase ",
                "\ncond ",
                "\nwith ",
                "\nfor ",
                "\ndo ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUST:
            return [
                # Split along function definitions
                "\nfn ",
                "\nconst ",
                "\nlet ",
                # Split along control flow statements
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SCALA:
            return [
                # Split along class definitions
                "\nclass ",
                "\nobject ",
                # Split along method definitions
                "\ndef ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SWIFT:
            return [
                # Split along function definitions
                "\nfunc ",
                # Split along class definitions
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.LATEX:
            return [
                # First, try to split along Latex sections
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                # Now split by environments
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                # Now split by math environments
                "\n\\\\begin{align}",
                "$$",
                "$",
                # Now split by the normal type of lines
                " ",
                "",
            ]
        elif language == Language.HTML:
            return [
                # First, try to split along HTML tags
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        elif language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                # Split along class definitions
                "\nclass ",
                "\nabstract ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                # Split along control flow statements
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                # Split by exceptions
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SOL:
            return [
                # Split along compiler information definitions
                "\npragma ",
                "\nusing ",
                # Split along contract definitions
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                # Split along method definitions
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.COBOL:
            return [
                # Split along divisions
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                # Split along sections within DATA DIVISION
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                # Split along sections within PROCEDURE DIVISION
                "\nINPUT-OUTPUT SECTION.",
                # Split along paragraphs and common statements
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                # Split by the normal type of lines
                "\n",
                " ",
                "",
            ]
        elif language == Language.LUA:
            return [
                # Split along variable and table definitions
                "\nlocal ",
                # Split along function definitions
                "\nfunction ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nrepeat ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.HASKELL:
            return [
                # Split along function definitions
                "\nmain :: ",
                "\nmain = ",
                "\nlet ",
                "\nin ",
                "\ndo ",
                "\nwhere ",
                "\n:: ",
                "\n= ",
                # Split along type declarations
                "\ndata ",
                "\nnewtype ",
                "\ntype ",
                "\n:: ",
                # Split along module declarations
                "\nmodule ",
                # Split along import statements
                "\nimport ",
                "\nqualified ",
                "\nimport qualified ",
                # Split along typeclass declarations
                "\nclass ",
                "\ninstance ",
                # Split along case expressions
                "\ncase ",
                # Split along guards in function definitions
                "\n| ",
                # Split along record field declarations
                "\ndata ",
                "\n= {",
                "\n, ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.POWERSHELL:
            return [
                # Split along function definitions
                "\nfunction ",
                # Split along parameter declarations (escape parentheses)
                "\nparam ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                # Split along class definitions (for PowerShell 5.0 and above)
                "\nclass ",
                # Split along try-catch-finally blocks
                "\ntry ",
                "\ncatch ",
                "\nfinally ",
                # Split by normal lines and empty spaces
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language in Language._value2member_map_:
            raise ValueError(f"Language {language} is not implemented yet!")
        else:
            raise ValueError(
                f"Language {language} is not supported! "
                f"Please choose from {list(Language)}"
            )





class ElementType(TypedDict):
    """Element type as typed dict."""

    url: str
    xpath: str
    content: str
    metadata: Dict[str, str]


class HTMLHeaderTextSplitter:
    """Split HTML content into structured Documents based on specified headers.

    Splits HTML content by detecting specified header tags (e.g., <h1>, <h2>) and
    creating hierarchical Document objects that reflect the semantic structure
    of the original content. For each identified section, the splitter associates
    the extracted text with metadata corresponding to the encountered headers.

    If no specified headers are found, the entire content is returned as a single
    Document. This allows for flexible handling of HTML input, ensuring that
    information is organized according to its semantic headers.

    The splitter provides the option to return each HTML element as a separate
    Document or aggregate them into semantically meaningful chunks. It also
    gracefully handles multiple levels of nested headers, creating a rich,
    hierarchical representation of the content.

    Args:
        headers_to_split_on (List[Tuple[str, str]]): A list of (header_tag,
            header_name) pairs representing the headers that define splitting
            boundaries. For example, [("h1", "Header 1"), ("h2", "Header 2")]
            will split content by <h1> and <h2> tags, assigning their textual
            content to the Document metadata.
        return_each_element (bool): If True, every HTML element encountered
            (including headers, paragraphs, etc.) is returned as a separate
            Document. If False, content under the same header hierarchy is
            aggregated into fewer Documents.

    Returns:
        List[Document]: A list of Document objects. Each Document contains
        `page_content` holding the extracted text and `metadata` that maps
        the header hierarchy to their corresponding titles.

    Example:
        .. code-block:: python

            from langchain_text_splitters.html_header_text_splitter import (
                HTMLHeaderTextSplitter,
            )

            # Define headers for splitting on h1 and h2 tags.
            headers_to_split_on = [("h1", "Main Topic"), ("h2", "Sub Topic")]

            splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                return_each_element=False
            )

            html_content = \"\"\"
            <html>
              <body>
                <h1>Introduction</h1>
                <p>Welcome to the introduction section.</p>
                <h2>Background</h2>
                <p>Some background details here.</p>
                <h1>Conclusion</h1>
                <p>Final thoughts.</p>
              </body>
            </html>
            \"\"\"

            documents = splitter.split_text(html_content)

            # 'documents' now contains Document objects reflecting the hierarchy:
            # - Document with metadata={"Main Topic": "Introduction"} and
            #   content="Introduction"
            # - Document with metadata={"Main Topic": "Introduction"} and
            #   content="Welcome to the introduction section."
            # - Document with metadata={"Main Topic": "Introduction",
            #   "Sub Topic": "Background"} and content="Background"
            # - Document with metadata={"Main Topic": "Introduction",
            #   "Sub Topic": "Background"} and content="Some background details here."
            # - Document with metadata={"Main Topic": "Conclusion"} and
            #   content="Conclusion"
            # - Document with metadata={"Main Topic": "Conclusion"} and
            #   content="Final thoughts."
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_element: bool = False,
        chunk_size=256,
        chunk_overlap=0,
        length_function=len,
        separators: Optional[List[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
    ) -> None:
        """Initialize with headers to split on.

        Args:
            headers_to_split_on: A list of tuples where
                each tuple contains a header tag and its corresponding value.
            return_each_element: Whether to return each HTML
                element as a separate Document. Defaults to False.
        """
        # Sort headers by their numeric level so that h1 < h2 < h3...
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda x: int(x[0][1:])
        )
        self.header_mapping = dict(self.headers_to_split_on)
        self.header_tags = [tag for tag, _ in self.headers_to_split_on]
        self.return_each_element = return_each_element
        if separators is None:
            separators = ["\n\n", "\n", ".", ";", ",", " ", ""]
        self.chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
        )

    def split_text(self, text: str) -> List[Document]:
        """Split the given text into a list of Document objects.

        Args:
            text: The HTML text to split.

        Returns:
            A list of split Document objects.
        """
        html_text = simplify_html(BeautifulSoup(text, "html.parser"))
        header_documents = self.split_text_from_file(StringIO(html_text))
        all_documents = []
        for hdoc in header_documents:
            docs = self.chunk_splitter.split_text(hdoc.page_content)
            for doc in docs:
                all_documents.append(Document(page_content=doc, metadata=hdoc.metadata))
        return all_documents


    def split_text_from_url(
        self, url: str, timeout: int = 10, **kwargs: Any
    ) -> List[Document]:
        """Fetch text content from a URL and split it into documents.

        Args:
            url: The URL to fetch content from.
            timeout: Timeout for the request. Defaults to 10.
            **kwargs: Additional keyword arguments for the request.

        Returns:
            A list of split Document objects.

        Raises:
            requests.RequestException: If the HTTP request fails.
        """
        kwargs.setdefault("timeout", timeout)
        response = requests.get(url, **kwargs)
        response.raise_for_status()
        return self.split_text(response.text)

    def split_text_from_file(self, file: Any) -> List[Document]:
        """Split HTML content from a file into a list of Document objects.

        Args:
            file: A file path or a file-like object containing HTML content.

        Returns:
            A list of split Document objects.
        """
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8") as f:
                html_content = f.read()
        else:
            html_content = file.read()
        return list(self._generate_documents(html_content))

    def _generate_documents(self, html_content: str) -> Any:
        """Private method that performs a DFS traversal over the DOM and yields.

        Document objects on-the-fly. This approach maintains the same splitting
        logic (headers vs. non-headers, chunking, etc.) while walking the DOM
        explicitly in code.

        Args:
            html_content: The raw HTML content.

        Yields:
            Document objects as they are created.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "Unable to import BeautifulSoup. Please install via `pip install bs4`."
            ) from e

        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.body if soup.body else soup

        # Dictionary of active headers:
        #   key = user-defined header name (e.g. "Header 1")
        #   value = (header_text, level, dom_depth)
        active_headers: Dict[str, Tuple[str, int, int]] = {}
        current_chunk: List[str] = []

        def finalize_chunk() -> Optional[Document]:
            """Finalize the accumulated chunk into a single Document."""
            if not current_chunk:
                return None

            final_text = "  \n".join(line for line in current_chunk if line.strip())
            current_chunk.clear()
            if not final_text.strip():
                return None

            final_meta = {k: v[0] for k, v in active_headers.items()}
            return Document(page_content=final_text, metadata=final_meta)

        # We'll use a stack for DFS traversal
        stack = [body]
        while stack:
            node = stack.pop()
            children = list(node.children)
            from bs4.element import Tag

            for child in reversed(children):
                if isinstance(child, Tag):
                    stack.append(child)

            tag = getattr(node, "name", None)
            if not tag:
                continue

            text_elements = [
                str(child).strip()
                for child in node.find_all(string=True, recursive=False)
            ]
            node_text = " ".join(elem for elem in text_elements if elem)
            if not node_text:
                continue

            dom_depth = len(list(node.parents))

            # If this node is one of our headers
            if tag in self.header_tags:
                # If we're aggregating, finalize whatever chunk we had
                if not self.return_each_element:
                    doc = finalize_chunk()
                    if doc:
                        yield doc

                # Determine numeric level (h1->1, h2->2, etc.)
                try:
                    level = int(tag[1:])
                except ValueError:
                    level = 9999

                # Remove any active headers that are at or deeper than this new level
                headers_to_remove = [
                    k for k, (_, lvl, d) in active_headers.items() if lvl >= level
                ]
                for key in headers_to_remove:
                    del active_headers[key]

                # Add/Update the active header
                header_name = self.header_mapping[tag]
                active_headers[header_name] = (node_text, level, dom_depth)

                # Always yield a Document for the header
                header_meta = {k: v[0] for k, v in active_headers.items()}
                yield Document(page_content=node_text, metadata=header_meta)

            else:
                headers_out_of_scope = [
                    k for k, (_, _, d) in active_headers.items() if dom_depth < d
                ]
                for key in headers_out_of_scope:
                    del active_headers[key]

                if self.return_each_element:
                    # Yield each element's text as its own Document
                    meta = {k: v[0] for k, v in active_headers.items()}
                    yield Document(page_content=node_text, metadata=meta)
                else:
                    # Accumulate text in our chunk
                    current_chunk.append(node_text)

        # If we're aggregating and have leftover chunk, yield it
        if not self.return_each_element:
            doc = finalize_chunk()
            if doc:
                yield doc


class HTMLSectionSplitter:
    """Splitting HTML files based on specified tag and font sizes.

    Requires lxml package.
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        xslt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new HTMLSectionSplitter.

        Args:
            headers_to_split_on: list of tuples of headers we want to track mapped to
                (arbitrary) keys for metadata. Allowed header values: h1, h2, h3, h4,
                h5, h6 e.g. [("h1", "Header 1"), ("h2", "Header 2"].
            xslt_path: path to xslt file for document transformation.
            Uses a default if not passed.
            Needed for html contents that using different format and layouts.
            **kwargs (Any): Additional optional arguments for customizations.

        """
        self.headers_to_split_on = dict(headers_to_split_on)

        if xslt_path is None:
            self.xslt_path = (
                pathlib.Path(__file__).parent / "converting_to_header.xslt"
            ).absolute()
        else:
            self.xslt_path = pathlib.Path(xslt_path).absolute()
        self.kwargs = kwargs
        self.chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str) -> List[Document]:
        """Split HTML text string.

        Args:
            text: HTML text
        """
        html_text = simplify_html(BeautifulSoup(text, "html.parser"))
        header_documents = self.split_text_from_file(StringIO(html_text))
        all_documents = []
        for hdoc in header_documents:
            docs = self.chunk_splitter.split_text(hdoc.page_content)
            for doc in docs:
                all_documents.append(Document(page_content=doc, metadata=hdoc.metadata))
        return all_documents

    def create_documents(
        self, texts: list[str], metadatas: Optional[list[dict[Any, Any]]] = None
    ) -> list[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])

                for key in chunk.metadata.keys():
                    if chunk.metadata[key] == "#TITLE#":
                        chunk.metadata[key] = metadata["Title"]
                metadata = {**metadata, **chunk.metadata}
                new_doc = Document(page_content=chunk.page_content, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_html_by_headers(self, html_doc: str) -> List[Dict[str, Optional[str]]]:
        """Split an HTML document into sections based on specified header tags.

        This method uses BeautifulSoup to parse the HTML content and divides it into
        sections based on headers defined in `headers_to_split_on`. Each section
        contains the header text, content under the header, and the tag name.

        Args:
            html_doc (str): The HTML document to be split into sections.

        Returns:
            List[Dict[str, Optional[str]]]: A list of dictionaries representing
            sections.
                Each dictionary contains:
                - 'header': The header text or a default title for the first section.
                - 'content': The content under the header.
                - 'tag_name': The name of the header tag (e.g., "h1", "h2").
        """
        try:
            from bs4 import BeautifulSoup
            from bs4.element import PageElement
        except ImportError as e:
            raise ImportError(
                "Unable to import BeautifulSoup/PageElement, \
                    please install with `pip install \
                    bs4`."
            ) from e

        soup = BeautifulSoup(html_doc, "html.parser")
        headers = list(self.headers_to_split_on.keys())
        sections: list[dict[str, str | None]] = []

        headers = soup.find_all(["body"] + headers)  # type: ignore[assignment]

        for i, header in enumerate(headers):
            header_element = cast(PageElement, header)
            if i == 0:
                current_header = "#TITLE#"
                current_header_tag = "h1"
                section_content: list[str] = []
            else:
                current_header = header_element.text.strip()
                current_header_tag = header_element.name  # type: ignore[attr-defined]
                section_content = []
            for element in header_element.next_elements:
                if i + 1 < len(headers) and element == headers[i + 1]:  # type: ignore[comparison-overlap]
                    break
                if isinstance(element, str):
                    section_content.append(element)
            content = " ".join(section_content).strip()

            if content != "":
                sections.append(
                    {
                        "header": current_header,
                        "content": content,
                        "tag_name": current_header_tag,
                    }
                )

        return sections

    def convert_possible_tags_to_header(self, html_content: str) -> str:
        """Convert specific HTML tags to headers using an XSLT transformation.

        This method uses an XSLT file to transform the HTML content, converting
        certain tags into headers for easier parsing. If no XSLT path is provided,
        the HTML content is returned unchanged.

        Args:
            html_content (str): The HTML content to be transformed.

        Returns:
            str: The transformed HTML content as a string.
        """
        if self.xslt_path is None:
            return html_content

        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # use lxml library to parse html document and return xml ElementTree
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(html_content), parser)

        xslt_tree = etree.parse(self.xslt_path)
        transform = etree.XSLT(xslt_tree)
        result = transform(tree)
        return str(result)

    def split_text_from_file(self, file: Any) -> List[Document]:
        """Split HTML content from a file into a list of Document objects.

        Args:
            file: A file path or a file-like object containing HTML content.

        Returns:
            A list of split Document objects.
        """
        file_content = file.getvalue()
        file_content = self.convert_possible_tags_to_header(file_content)
        sections = self.split_html_by_headers(file_content)

        return [
            Document(
                cast(str, section["content"]),
                metadata={
                    self.headers_to_split_on[str(section["tag_name"])]: section[
                        "header"
                    ]
                },
            )
            for section in sections
        ]

def simplify_html(soup, keep_attr: bool = False) -> str:
    for script in soup(["script", "style"]):
        script.decompose()
    #  remove all attributes
    if not keep_attr:
        for tag in soup.find_all(True):
            tag.attrs = {}
    #  remove empty tags recursively
    while True:
        removed = False
        for tag in soup.find_all():
            if not tag.text.strip():
                tag.decompose()
                removed = True
        if not removed:
            break
    #  remove href attributes
    for tag in soup.find_all("a"):
        del tag["href"]
    #  remove comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    def concat_text(text):
        text = "".join(text.split("\n"))
        text = "".join(text.split("\t"))
        text = "".join(text.split(" "))
        return text

    # remove all tags with no text
    for tag in soup.find_all():
        children = [child for child in tag.contents if not isinstance(child, str)]
        if len(children) == 1:
            tag_text = tag.get_text()
            child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
            if concat_text(child_text) == concat_text(tag_text):
                tag.replace_with_children()
    #  if html is not wrapped in a html tag, wrap it

    # remove empty lines
    res = str(soup)
    lines = [line for line in res.split("\n") if line.strip()]
    res = "\n".join(lines)
    return res


if __name__ == "__main__":
    url = "https://techcommunity.microsoft.com/blog/aiplatformblog/introducing-phi-4-microsoft%E2%80%99s-newest-small-language-model-specializing-in-comple/4357090"
    html_text = requests.get(url).text

    html_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
        return_each_element=False,
    )



    html_text = simplify_html(BeautifulSoup(html_text, "html.parser"))
    documents = html_splitter.split_text(html_text)

    for doc in documents:
        print(doc.page_content)
        print("-"*100)
