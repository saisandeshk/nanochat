"""
Simple tokenizer implementations extracted from bpe/src/bpe.ipynb for testing
"""

import regex as re
from collections import defaultdict
import pytest 
import os 

# GPT4_SPLIT_REGEX = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]|\s[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPLIT_REGEX = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]?|\s*[\r\n]|\s+(?!\S)|\s+"""


class SimpleTokenizer:
    """Simple word-level tokenizer with regex-based preprocessing"""

    def __init__(self, vocab):
        self.str_to_id = vocab
        self.id_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.findall(GPT4_SPLIT_REGEX, text)
        preprocessed = [token for token in preprocessed if token]
        # replaces unknown words by <unk> token
        preprocessed = [token if token in self.str_to_id else '<|unk|>' for token in preprocessed]
        return [self.str_to_id[token] for token in preprocessed]

    def decode(self, token_ids):
        txt = " ".join([self.id_to_str[token_id] for token_id in token_ids])
        txt = re.sub(r'\s+([,.:;?!"()\'])', r'\1', txt)
        return txt


def get_stats(token_ids, counts=None):
    """
    Given a list of token_ids, return a dictionary of pair frequencies,
    which are then used to merge based on highest frequency.
    """
    counts = {} if counts is None else counts
    for pair in zip(token_ids, token_ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_token_ids(ids, pair, idx):
    """
    In the list of token_ids, merge the given pair and assign it idx
    """
    i = 0
    new_ids = []
    while i < len(ids):
        # if not at the very end of the list and found the pair
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# fast version of the merge_token_ids function - just better python code 
def fast_merge_inplace_token_ids(ids, pair, idx):
    """
    In the list of integers (ids), replace all the concurrent occurences of pair, with the new 
    integer token idx in place.
    """
    # find all the positions where the pair occurs 
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            ids[i] = idx 
            ids.pop(i+1)
        else:
            i += 1
    return ids 


class RegexTokenizer:
    """BPE tokenizer with regex-based preprocessing"""

    def __init__(self, pattern=None):
        # pattern - optional string to override the default GPT4 regex pattern
        # special_tokens - str -> int dict of special tokens used during tokenization
        self.pattern = GPT4_SPLIT_REGEX if pattern is None else pattern
        self.merges = {}
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {} # like "<|unk|>" or ""
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is deterministic and is derived from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)} # initialize the vocab
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # keep track of whether at any point, during the training, the merge is ambiguous - counts of pairs are not unique
        ambiguous = False

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # iteratively merge the most common pairs to create new tokens until we reach the desired vocab_size or num_merges which is vocab_size - 256
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        for i in range(num_merges):
            # count the no of times every consecutive pair appears
            stats = {}
            # chunk_ids = token_ids
            for chunk_ids in ids:
                # passing in stats will update in place, adding up counts - we are doing chunk_wise?
                get_stats(chunk_ids, stats)

            if not stats:
                break  # no more pairs to merge

            # find the pair with the highest count
            pair = max(stats, key=stats.get) # type: ignore 
            # check if the merge is ambiguous - i.e max value is not unique
            pair_count = stats[pair]
            pairs_with_max_count = [pair for pair, count in stats.items() if count == pair_count]
            if len(pairs_with_max_count) > 1:
                ambiguous = True

            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurences of pair with idx - done by the merge function
            ids = [merge_token_ids(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode() - updating from {}
        self.vocab = vocab # used in decode() - updating from the initial 256 dict

        return ambiguous

    def _encode_chunk(self, text_bytes):
        # return the token_ids
        # convert all the bytes into integers in range 0...255
        ids = list(text_bytes)
        while len(ids) >=2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily we can detect this terminating case by a membership check.
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair
            idx = self.merges[pair]
            ids = merge_token_ids(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        # Encoding that ignores any special tokens
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def decode(self, token_ids):
        # given token_ids, return Python string
        text_bytes = b''.join(self.vocab[idx] for idx in token_ids)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

@pytest.fixture 
def verdict_text():
    """Fixture to load the training text."""
    # Assuming 'the-verdict.txt' is in the same directory as the notebook
    file_path = 'bpe/src/the-verdict.txt'
    if not os.path.exists(file_path):
        # Fallback for different execution contexts, e.g. running from root
        file_path = os.path.join(os.path.dirname(__file__), 'src/the-verdict.txt')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def test_tokenizer_workflow(verdict_text):
    """
    Tests the full workflow:
    1. Initialize tokenizer.
    2. Train on 'the-verdict.txt'.
    3. Encode a sample text.
    4. Decode the tokens.
    5. Verify the decoded text matches the original.
    """
    # 1. Initialize tokenizer
    tokenizer = RegexTokenizer()
    
    # 2. Train the tokenizer
    vocab_size = 300 # A small vocab size for a quick test
    tokenizer.train(verdict_text, vocab_size, verbose=False)
    
    # Verify training has occurred
    assert len(tokenizer.merges) == vocab_size - 256
    assert len(tokenizer.vocab) == vocab_size
    
    # 3. Encode a sample text
    prompt = "Hello world, this is a test."
    encoded_ids = tokenizer.encode_ordinary(prompt)
    
    # Verify encoding output
    assert isinstance(encoded_ids, list)
    assert all(isinstance(i, int) for i in encoded_ids)
    
    # 4. Decode the tokens
    decoded_text = tokenizer.decode(encoded_ids)
    
    # 5. Verify the decoded text
    assert decoded_text == prompt
    
    print("\n--- Test Tokenizer Workflow ---")
    print(f"Original prompt: {prompt}")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Decoded text: {decoded_text}")
    print("✅ Test Passed: Decoded text matches original prompt.")

def test_empty_and_special_chars():
    """Tests tokenizer behavior with empty strings and special characters."""
    tokenizer = RegexTokenizer()
    tokenizer.train("", 256) # No training text, no merges

    # Test empty string
    assert tokenizer.encode_ordinary("") == []
    assert tokenizer.decode([]) == ""

    # Test string with only special characters
    prompt = "!@#$%^&*()"
    encoded = tokenizer.encode_ordinary(prompt)
    decoded = tokenizer.decode(encoded)
    assert decoded == prompt

    print("\n--- Test Empty and Special Chars ---")
    print("✅ Test Passed: Handles empty and special character strings correctly.")