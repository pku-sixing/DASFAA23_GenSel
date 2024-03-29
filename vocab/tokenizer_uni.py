# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""
from transformers import XLNetTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/spiece.model",
    }
}


class UniTokenizer(XLNetTokenizer):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""

    def __init__(self, *args, **kwargs):
        """
        Construct a CPM tokenizer. Based on `Jieba <https://pypi.org/project/jieba/>` and `SentencePiece
        <https://github.com/google/sentencepiece>`__.

        This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main
        methods. Users should refer to this superclass for more information regarding those methods.

        Args:
            vocab_file (:obj:`str`):
                `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
                contains the vocabulary necessary to instantiate a tokenizer.
            do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to lowercase the input when tokenizing.
            remove_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to strip the text when tokenizing (removing excess spaces before and after the string).
            keep_accents (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to keep accents when tokenizing.
            bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier
                token.

                .. note::

                    When building a sequence using special tokens, this is not the token that is used for the beginning
                    of sequence. The token used is the :obj:`cls_token`.
            eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
                The end of sequence token.

                .. note::

                    When building a sequence using special tokens, this is not the token that is used for the end of
                    sequence. The token used is the :obj:`sep_token`.
            unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
                this token instead.
            sep_token (:obj:`str`, `optional`, defaults to :obj:`"<sep>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
                for sequence classification or for a text and a question for question answering. It is also used as the
                last token of a sequence built with special tokens.
            pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (:obj:`str`, `optional`, defaults to :obj:`"<cls>"`):
                The classifier token which is used when doing sequence classification (classification of the whole
                sequence instead of per-token classification). It is the first token of the sequence when built with
                special tokens.
            mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<eop>", "<eod>"]`):
                Additional special tokens used by the tokenizer.

        Attributes:
            sp_model (:obj:`SentencePieceProcessor`):
                The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
        """
        super().__init__(*args, **kwargs)
        try:
            import jieba
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install jieba to use CpmTokenizer."
                "See https://pypi.org/project/jieba/ for installation."
            )
        self.jieba = jieba
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    def _tokenize(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in self.jieba.cut(text, cut_all=False)]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)

    def word_align(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in self.jieba.cut(text, cut_all=False)]
        sub_text = super()._tokenize(" ".join(text), *args, **kwargs)
        aligns = [1]
        aligns_fw = [1]
        for token in sub_text[1:]:
            if token[0] == '▁':
                aligns.append(aligns[-1] + 1)
                aligns_fw.append(1)
            else:
                aligns.append(aligns[-1])
                aligns_fw.append(aligns_fw[-1]+1)
        aligns.reverse()
        aligns_bw = [1]
        for pos in range(1, len(aligns)):
            if aligns[pos] == aligns[pos-1]:
                aligns_bw.append(aligns_bw[-1] + 1)
            else:
                aligns_bw.append(1)
        aligns.reverse()
        aligns_bw.reverse()
        return aligns, aligns_fw, aligns_bw

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")
        return text
