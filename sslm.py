"""SSLM, a Small and Stupid Language Model.

A variable-order n-gram model in pure numpy. Training counts, for every context
of 1 to K tokens seen in a corpus, which tokens came next and how often.
Generation starts from the longest context it knows and backs off to shorter
ones. Two knobs:

- ``T``, the temperature: 0 always picks the most frequent continuation,
  1 follows the corpus frequencies, above 1 flattens them;
- ``min_ch``, the memorization knob: a context is used only if the corpus shows
  at least ``min_ch`` distinct continuations for it. At 1 the model happily
  follows contexts seen once, and recites its corpus; at 3 it must back off to
  shorter, shared contexts, and recombines.

A context with a single continuation is not always a quote: aujourd ' is
always followed by hui. ``min_seen`` (default 50) tells the two apart: a
context seen that often is a rule of the language, and is used whatever
``min_ch`` says. Without it, ``min_ch=3`` makes non-words such as "aujourd'ai".

Typical use::

    tokens = load_an("syseron.xml.zip")
    model = train(tokens, K=5)
    out = generate(model, tokenize_text("La parole est à"), n=80, T=0.7, min_ch=3, seed=0)
    print(detokenize_text(out))
"""

import html
import os
import re
import sysconfig
import unicodedata
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------

# Prose: a word, or a single punctuation mark. Spaces are dropped.
TEXT_TOKEN = re.compile(r"\w+|[^\w\s]")

# Code: a word, a line break with the spaces around it (so the indentation of
# the next line is part of the token), a run of spaces, or any other single
# character. Nothing is dropped: ''.join(tokens) gives back the source.
CODE_TOKEN = re.compile(r"\w+|[ \t]*\n[ \t]*|[ \t]+|.", re.DOTALL)


# A passage between straight double quotes, on one line.
QUOTED = re.compile(r'"([^"\n]*)"')


def tokenize_text(text):
    """Split prose into words and punctuation marks."""
    text = unicodedata.normalize("NFKC", text).replace("’", "'")  # one apostrophe
    text = QUOTED.sub(r"“\1”", text)  # "..." -> “...”: opening and closing quotes become distinct tokens
    return TEXT_TOKEN.findall(text)


def tokenize_code(source):
    """Split source code into tokens, without losing any character."""
    return CODE_TOKEN.findall(source)


NO_SPACE_BEFORE = set(".,;:!?)]'-\n”")
NO_SPACE_AFTER = set("(['-\n“")


def detokenize_text(tokens):
    """Join prose tokens with spaces, except before .,;:!?)]” and after ([“.

    Apostrophes, hyphens and line breaks are glued on both sides.
    """
    pieces, prev = [], "\n"
    for tok in tokens:
        glue = prev in NO_SPACE_AFTER or tok in NO_SPACE_BEFORE
        pieces.append(tok if glue else " " + tok)
        prev = tok
    return "".join(pieces)


def detokenize_code(tokens):
    """Code tokens keep their whitespace: just concatenate them."""
    return "".join(tokens)


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------

# One <texte> element (the 9 empty <texte/> elements are skipped: no "/").
AN_TEXTE = re.compile(r"<texte\b[^>/]*>(.*?)</texte>", re.DOTALL)
XML_TAG = re.compile(r"<[^>]+>")


def iter_an_texts(path="syseron.xml.zip"):
    """Yield the plain text of every <texte> element of the zipped XML debates."""
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            xml = archive.read(name).decode("utf-8")
            for fragment in AN_TEXTE.findall(xml):
                yield html.unescape(XML_TAG.sub("", fragment))


def load_an(path="syseron.xml.zip"):
    """French National Assembly debates, as a list of tokens.

    A newline token marks the end of each speech.
    """
    tokens = []
    for text in iter_an_texts(path):
        tokens += tokenize_text(text)
        tokens.append("\n")
    return tokens


def load_tinystories(path, max_chars=None):
    """TinyStories (a .txt file of stories separated by <|endoftext|>) as tokens.

    A newline token marks the end of each story. ``max_chars`` reads only the
    beginning of the file.
    """
    with open(path, encoding="utf-8") as f:
        stories = f.read(max_chars).split("<|endoftext|>")
    tokens = []
    for story in stories:
        if story.strip():
            tokens += tokenize_text(story)
            tokens.append("\n")
    return tokens


# Tests, installed packages, the IDLE application, and lib2to3 (gone in Python 3.13).
SKIPPED_DIRS = {"test", "tests", "idle_test", "site-packages", "__pycache__", "idlelib", "lib2to3"}


def stdlib_files():
    """Sorted paths of the .py files of the running Python's standard library (see SKIPPED_DIRS)."""
    root = sysconfig.get_paths()["stdlib"]
    files = []
    for folder, subfolders, filenames in os.walk(root):
        subfolders[:] = [d for d in subfolders if d not in SKIPPED_DIRS]  # do not descend there
        files += [Path(folder, f) for f in filenames
                  if f.endswith(".py") and not f.startswith("test_")]
    return sorted(files)


def load_stdlib():
    """Source code of the standard library (tests excluded), as lossless tokens."""
    tokens = []
    for path in stdlib_files():
        tokens += tokenize_code(path.read_text(encoding="utf-8"))
    return tokens


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

HASH_MULT = np.uint64(0x9E3779B97F4A7C15)  # odd constant, about 2**64 / golden ratio


def extend_hash(h, token_id):
    """Hash of a context extended by one more (older) token.

    Works on a single hash or on an array of hashes. The uint64 product wraps
    around modulo 2**64: this overflow *is* the hash, not a bug. numpy warns
    about it on single numbers (not on arrays), so the warning is silenced.
    The +1 makes token 0 count (h ^ 0 would leave h unchanged).

    Two contexts may share a hash (a collision). The model keeps 64 - shift
    bits (about 47), so with a few million contexts we expect less than one.
    """
    with np.errstate(over="ignore"):
        return (h ^ (token_id + np.uint64(1))) * HASH_MULT


@dataclass
class Table:
    """The contexts of one length k, and what followed them in the corpus.

    ``contexts`` holds sorted context hashes. The continuations of ``contexts[j]`` are
    ``nexts[offsets[j]:offsets[j + 1]]`` (token ids), seen ``counts[...]`` times.
    """

    contexts: np.ndarray
    offsets: np.ndarray
    nexts: np.ndarray
    counts: np.ndarray

    def lookup(self, context):
        """(token ids, counts) of the continuations of ``context``, or None if never seen."""
        j = np.searchsorted(self.contexts, context)
        if j == len(self.contexts) or self.contexts[j] != context:
            return None
        start, end = self.offsets[j], self.offsets[j + 1]
        return self.nexts[start:end], self.counts[start:end]


@dataclass
class Model:
    vocab: list  # token id -> token
    index: dict  # token -> token id
    shift: int  # number of low bits used to store a token id
    tables: list  # tables[k - 1] describes the contexts of k tokens
    unigram: np.ndarray  # count of each token id, used when no context is known

    @property
    def K(self):
        """Longest context length."""
        return len(self.tables)


def encode(model, tokens):
    """Token ids of ``tokens`` (all must be in the vocabulary)."""
    return np.fromiter((model.index[tok] for tok in tokens), dtype=np.uint64, count=len(tokens))


def make_table(contexts, nexts, shift):
    """Build the Table of one context length from aligned arrays (context, next token id).

    Each (context, next) pair is packed into one uint64: the context in the
    high bits, the next token id in the ``shift`` low bits. Sorting the packed
    keys groups the pairs by context, and np.unique counts them.
    """
    keys, counts = np.unique((contexts << shift) | nexts, return_counts=True)
    key_contexts = keys >> shift
    starts = np.flatnonzero(np.r_[True, key_contexts[1:] != key_contexts[:-1]])
    return Table(contexts=key_contexts[starts],
                 offsets=np.append(starts, len(keys)),
                 nexts=keys & np.uint64((1 << shift) - 1),
                 counts=counts)


def train(tokens, K=5):
    """Count, for every context of 1 to K tokens in ``tokens``, which token came next."""
    vocab = list(dict.fromkeys(tokens))  # distinct tokens, by first appearance
    index = {tok: i for i, tok in enumerate(vocab)}
    model = Model(vocab, index, shift=len(vocab).bit_length(), tables=[], unigram=None)
    ids = encode(model, tokens)
    model.unigram = np.bincount(ids.astype(np.int64), minlength=len(vocab))

    n = len(ids) - K  # positions that have K tokens of context
    nexts = ids[K:]  # the token to predict at each position
    h = np.zeros(n, dtype=np.uint64)  # hash of the empty context
    for k in range(1, K + 1):
        h = extend_hash(h, ids[K - k:K - k + n])  # add the k-th previous token
        model.tables.append(make_table(h >> model.shift, nexts, model.shift))
    return model


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def sample(token_ids, counts, T, rng):
    """Draw one of ``token_ids`` with probability proportional to ``counts ** (1 / T)``.

    T = 0 always returns the most frequent one.
    """
    if T == 0:
        return int(token_ids[np.argmax(counts)])
    weights = (counts / counts.max()) ** (1 / T)  # dividing by the max avoids overflow at small T
    return int(rng.choice(token_ids, p=weights / weights.sum()))


def next_token(model, history, rng, T=1.0, min_ch=1, min_seen=50):
    """Draw the token that follows ``history`` (token ids).

    Backs off from the longest known context to the shortest, and uses the
    first one that is a choice (at least ``min_ch`` distinct continuations),
    a rule (seen at least ``min_seen`` times, like aujourd ' -> hui), or a
    single token. Returns (token id, length of the context used); length 0
    means no known context, and the token is drawn from the overall frequencies.
    """
    # Hashes of the last 1, 2, ..., K tokens, newest token first as in train.
    h, contexts = np.uint64(0), []
    for tok in reversed(history[-model.K:]):
        h = extend_hash(h, tok)
        contexts.append(h >> model.shift)
    # Back off from the longest context to the shortest.
    for k in range(len(contexts), 0, -1):
        found = model.tables[k - 1].lookup(contexts[k - 1])
        if found is None:
            continue
        nexts, counts = found
        if len(nexts) >= min_ch or counts.sum() >= min_seen or k == 1:
            return sample(nexts, counts, T, rng), k
    return sample(np.arange(len(model.vocab)), model.unigram, T, rng), 0


def generate(model, prompt_tokens, n=80, T=1.0, min_ch=1, seed=None, return_orders=False, min_seen=50):
    """Continue ``prompt_tokens`` with ``n`` new tokens; return prompt + new tokens.

    Same ``seed``, same text. With ``return_orders=True``, also return the
    context length used for each new token.
    """
    rng = np.random.default_rng(seed)
    unknown = [tok for tok in prompt_tokens if tok not in model.index]
    if unknown:
        warnings.warn(f"tokens never seen in training, ignored: {unknown}", stacklevel=2)
    history = [model.index[tok] for tok in prompt_tokens if tok in model.index]
    new_tokens, orders = [], []
    for _ in range(n):
        tok, k = next_token(model, history, rng, T, min_ch, min_seen)
        history.append(tok)
        new_tokens.append(model.vocab[tok])
        orders.append(k)
    out = list(prompt_tokens) + new_tokens
    return (out, orders) if return_orders else out


# --------------------------------------------------------------------------
# Inspecting the output
# --------------------------------------------------------------------------

def longest_copy(corpus_ids, ids):
    """Longest run of ``ids`` that appears verbatim in ``corpus_ids``.

    Returns (start in ids, length, corpus positions where the run occurs):
    one position means a passage copied from one place, many mean a formula.
    Brute force: for each start, keep the corpus positions that still match.
    """
    best = (0, 0, np.array([], dtype=np.intp))
    for start in range(len(ids)):
        positions = np.flatnonzero(corpus_ids == ids[start])
        length = 1 if len(positions) else 0
        while len(positions) and start + length < len(ids):
            ahead = positions[positions + length < len(corpus_ids)]
            ahead = ahead[corpus_ids[ahead + length] == ids[start + length]]
            if len(ahead) == 0:
                break
            positions, length = ahead, length + 1
        if length > best[1]:
            best = (start, length, positions)
    return best


def first_function(code):
    """Cut generated code after its first function.

    The function ends before the next def, class or decorator, or before the
    first line back at column 0. If it never ends, drop its last, probably
    unfinished, line.
    """
    lines = code.split("\n")
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() and (not line[0].isspace() or line.lstrip().startswith(("def ", "class ", "@"))):
            return "\n".join(lines[:i]).rstrip() + "\n"
    return "\n".join(lines[:-1]).rstrip() + "\n"
