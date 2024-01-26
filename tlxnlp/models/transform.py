import os
import re
import unicodedata
from collections import OrderedDict
from typing import List

import numpy as np
import sentencepiece as spm
from tensorlayerx import logging


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTransform(object):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        max_length=128,
        task="text",
        **kwargs,
    ):
        self.do_basic_tokenize = do_basic_tokenize
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.max_length = max_length
        self.task = task
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}
        self.unique_no_split_tokens: List[str] = []
        super().__init__(**kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token
        )
        self.all_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]

        self.is_train = True

    def set_train(self):
        self.is_train = True

    def set_eval(self):
        self.is_train = False

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens
            ):
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def string_to_ids(self, text, max_length=None):
        tokens = self._tokenize(text)

        if max_length is None:
            tokens = [self.cls_token] + tokens + [self.sep_token]
            attention_mask = [1] * len(tokens)
        else:
            if not isinstance(max_length, int):
                raise ValueError(f"{max_length} is not int.")
            else:
                tokens_length = len(tokens)
                if tokens_length >= (max_length - 2):
                    tokens = (
                        [self.cls_token] + tokens[: max_length - 2] + [self.sep_token]
                    )
                    attention_mask = [1] * len(tokens)
                else:
                    attention_mask = [1] * (len(tokens) + 2) + [0] * (
                        max_length - tokens_length - 2
                    )
                    tokens = (
                        [self.cls_token]
                        + tokens
                        + [self.sep_token]
                        + [self.pad_token] * (max_length - tokens_length - 2)
                    )

        ids = self.convert_tokens_to_ids(tokens)
        return {"inputs": np.array(ids), "attention_mask": np.array(attention_mask)}

    def process_token(self, text, label, max_length=None):
        ids = []
        labels = []
        for token, l in zip(text, label):
            token = self._tokenize(token)
            id = self.convert_tokens_to_ids(token)
            l = [l] * len(id)
            ids += id
            labels += l

        if max_length is None:
            ids = (
                self.convert_tokens_to_ids([self.cls_token])
                + ids
                + self.convert_tokens_to_ids([self.sep_token])
            )
            attention_mask = [1] * len(ids)
            labels = [-100] + labels + [-100]
        else:
            if not isinstance(max_length, int):
                raise ValueError(f"{max_length} is not int.")
            else:
                ids_length = len(ids)
                if ids_length >= (max_length - 2):
                    ids = (
                        self.convert_tokens_to_ids([self.cls_token])
                        + ids[: max_length - 2]
                        + self.convert_tokens_to_ids([self.sep_token])
                    )
                    attention_mask = [1] * len(ids)
                    labels = [-100] + labels[: max_length - 2] + [-100]
                else:
                    attention_mask = [1] * (len(ids) + 2) + [0] * (
                        max_length - ids_length - 2
                    )
                    ids = (
                        self.convert_tokens_to_ids([self.cls_token])
                        + ids
                        + self.convert_tokens_to_ids([self.sep_token])
                        + self.convert_tokens_to_ids([self.pad_token])
                        * (max_length - ids_length - 2)
                    )
                    labels = [-100] + labels + [-100] * (max_length - ids_length - 1)

        return {
            "inputs": np.array(ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.zeros_like(np.array(ids)),
        }, np.array(labels)

    def __call__(self, text, label, max_length=None):
        if self.task == "token":
            return self.process_token(
                text, label, max_length if max_length else self.max_length
            )
        max_length = max_length if max_length else self.max_length
        inputs = self.string_to_ids(text, max_length=max_length)
        inputs["token_type_ids"] = np.zeros_like(inputs["inputs"])
        return inputs, label


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = (
            self.never_split.union(set(never_split))
            if never_split
            else self.never_split
        )
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example::

            >>> trie = Trie()
            >>> trie.add("Hello 友達")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}
            >>> trie.add("Hello")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        """
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example::

            >>> trie = Trie()
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS] This is a extra_id_100"]
            >>> trie.add("[CLS]")
            >>> trie.add("extra_id_1")
            >>> trie.add("extra_id_100")
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS]", " This is a ", "extra_id_100"]
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = None
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = (
                            text[lookahead_index]
                            if lookahead_index < len(text)
                            else None
                        )
                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logging.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


class T5Transform(object):
    def __init__(
        self,
        vocab_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        prefix="translate English to French: ",
        next_prefix=None,
        task="text",
        source_max_length=512,
        label_max_length=512,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.extra_ids = extra_ids
        self.source_max_length = source_max_length
        self.prefix = prefix
        self.task = task
        self.next_prefix = next_prefix
        self.label_max_length = label_max_length
        if self.extra_ids > 0:
            self.additional_special_tokens = [
                f"<extra_id_{i}>" for i in range(self.extra_ids)
            ]

        if self.eos_token is not None:
            self.additional_special_tokens.append(self.eos_token)

        if self.unk_token is not None:
            self.additional_special_tokens.append(self.unk_token)

        if self.pad_token is not None:
            self.additional_special_tokens.append(self.pad_token)

        self._create_trie(self.additional_special_tokens)

        if self.vocab_file is None:
            raise ValueError(f"vocab file is None.")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

        self.is_train = True
        super(T5Transform, self).__init__(**kwargs)

    def set_train(self):
        self.is_train = True

    def set_eval(self):
        self.is_train = False

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self.extra_ids

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if (
                hasattr(self, "do_lower_case")
                and self.do_lower_case
                and token not in self.all_special_tokens
            ):
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def _tokenize(self, token):
        return self.sp_model.encode(token, out_type=str)

    def tokenize(self, text):
        no_split_token = set(self.additional_special_tokens)
        tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                # We strip left and right by default
                if right:
                    tokens[i + 1] = right.lstrip()
                if left:
                    tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            token = f"<extra_id_{self.vocab_size - 1 - index}>"
        return token

    def convert_tokens_to_string(self, tokens, remove_special_token=False):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.additional_special_tokens:
                if not remove_special_token:
                    out_string += (
                        self.sp_model.decode_pieces(current_sub_tokens) + token + " "
                    )
                    current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()

    def ids_to_string(self, ids, remove_special_token=True):
        tokens = [self._convert_id_to_token(int(index)) for index in ids if index >= 0]
        return self.convert_tokens_to_string(
            tokens, remove_special_token=remove_special_token
        )

    def string_to_ids(self, text, max_length=None):
        if isinstance(text, list):
            input_ids = []
            attention_masks = []
            for i in text:
                input_id, attention_mask = self.string_to_ids(i, max_length=max_length)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
            return {
                "inputs": np.array(input_ids),
                "attention_mask": np.array(attention_masks),
            }
        tokens = self.tokenize(text)

        if max_length is None:
            tokens = tokens + [self.eos_token]
            attention_mask = [1] * len(tokens)
        else:
            if not isinstance(max_length, int):
                raise ValueError(f"{max_length} is not int.")
            else:
                tokens_length = len(tokens)
                if tokens_length >= (max_length - 1):
                    tokens = tokens[: max_length - 1] + [self.eos_token]
                    attention_mask = [1] * len(tokens)
                else:
                    attention_mask = [1] * (len(tokens) + 1) + [0] * (
                        max_length - tokens_length - 1
                    )
                    tokens = (
                        tokens
                        + [self.eos_token]
                        + [self.pad_token] * (max_length - tokens_length - 1)
                    )

        ids = self.convert_tokens_to_ids(tokens)
        return {"inputs": np.array(ids), "attention_mask": np.array(attention_mask)}

    def process_token(self, text, label, source_max_length=None):
        ids = []
        labels = []
        for token, l in zip(text, label):
            token = self.tokenize(token)
            id = self.convert_tokens_to_ids(token)
            l = [l] * len(id)
            ids += id
            labels += l

        if source_max_length is None:
            ids = ids + self.convert_tokens_to_ids([self.eos_token])
            attention_mask = [1] * len(ids)
            labels = labels + [-100]
        else:
            if not isinstance(source_max_length, int):
                raise ValueError(f"{source_max_length} is not int.")
            else:
                ids_length = len(ids)
                if ids_length >= (source_max_length - 1):
                    ids = ids[: source_max_length - 1] + self.convert_tokens_to_ids(
                        [self.eos_token]
                    )
                    attention_mask = [1] * len(ids)
                    labels = labels[: source_max_length - 1] + [-100]
                else:
                    attention_mask = [1] * (len(ids) + 1) + [0] * (
                        source_max_length - ids_length - 1
                    )
                    ids = (
                        ids
                        + self.convert_tokens_to_ids([self.eos_token])
                        + self.convert_tokens_to_ids([self.pad_token])
                        * (source_max_length - ids_length - 1)
                    )
                    labels = labels + [-100] * (source_max_length - ids_length)
        return {
            "inputs": np.array(ids),
            "attention_mask": np.array(attention_mask),
            "labels": np.array(labels),
        }, np.array(labels)

    def __call__(self, text, label, source_max_length=None, label_max_length=None):
        if self.task == "token":
            return self.process_token(
                text,
                label,
                source_max_length if source_max_length else self.source_max_length,
            )
        if isinstance(text, str):
            if self.prefix:
                text = self.prefix + text
            inputs = self.string_to_ids(
                text,
                max_length=source_max_length
                if source_max_length
                else self.source_max_length,
            )
        elif (isinstance(text, tuple) or isinstance(text, list)) and len(text) == 2:
            first_text = text[0]
            if self.prefix:
                first_text = self.prefix + first_text
            first_inputs = self.string_to_ids(
                first_text,
                max_length=source_max_length
                if source_max_length
                else self.source_max_length,
            )

            second_text = text[1]
            if self.next_prefix:
                second_text = self.next_prefix + second_text
            second_inputs = self.string_to_ids(
                second_text,
                max_length=source_max_length
                if source_max_length
                else self.source_max_length,
            )
            inputs = {
                "inputs": np.concatenate(
                    [first_inputs["inputs"], second_inputs["inputs"]]
                ),
                "attention_mask": np.concatenate(
                    [first_inputs["attention_mask"], second_inputs["attention_mask"]]
                ),
            }
        else:
            raise ValueError(f"{text} is wrong.")

        if isinstance(label, str):
            labels = self.string_to_ids(
                label,
                max_length=label_max_length
                if label_max_length
                else self.label_max_length,
            )
            labels = np.where(labels["attention_mask"], labels["inputs"], -100)
        else:
            labels = label
        inputs["labels"] = labels
        return inputs, labels
