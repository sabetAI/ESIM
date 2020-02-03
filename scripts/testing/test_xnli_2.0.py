# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import concat_batches, truncate, to_cuda
from ..data.dataset import ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters


XNLI_LANGS = [
    "fr",
]


logger = getLogger()


class XNLI:
    def __init__(self, embedder, scores, params):
        """
        Initialize XNLI trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self.scores = scores

    def get_iterator(self, splt, lang):
        """
        Get a monolingual data iterator.
        """
        assert splt in ["valid", "test"] or splt == "train" and lang == "en"
        return self.data[lang][splt]["x"].get_iterator(
            shuffle=(splt == "train"),
            group_by_size=self.params.group_by_size,
            return_indices=True,
        )

    def run(self):
        """
        Run XNLI training / evaluation.
        """

        # load data
        self.data = self.load_data()
        if not self.data["dico"] == self._embedder.dico:
            raise Exception(
                (
                    "Dictionary in evaluation data (%i words) seems different than the one "
                    + "in the pretrained model (%i words). Please verify you used the same dictionary, "
                    + "and the same values for max_vocab and min_count."
                )
                % (len(self.data["dico"]), len(self._embedder.dico))
            )

        # embedder

        # evaluation
        logger.info("XNLI - Evaluating epoch %i ..." % epoch)
        with torch.no_grad():
            scores = self.eval()
            self.scores.update(scores)

    def eval(self):
        """
        Evaluate on XNLI validation and test sets, for all languages.
        """
        params = self.params
        self.embedder.eval()
        self.proj.eval()

        scores = OrderedDict({"epoch": self.epoch})

        for splt in ["test"]:
            for lang in XNLI_LANGS:
                if lang not in params.lang2id:
                    continue

                lang_id = params.lang2id[lang]
                valid = 0
                total = 0

                for batch in self.get_iterator(splt, lang):

                    # batch
                    (sent1, len1), (sent2, len2), idx = batch
                    x, lengths, positions, langs = concat_batches(
                        sent1,
                        len1,
                        lang_id,
                        sent2,
                        len2,
                        lang_id,
                        params.pad_index,
                        params.eos_index,
                        reset_positions=False,
                    )
                    y = self.data[lang][splt]["y"][idx]

                    # cuda
                    x, y, lengths, positions, langs = to_cuda(
                        x, y, lengths, positions, langs
                    )

                    # forward
                    output = self.proj(
                        self.embedder.get_embeddings(x, lengths, positions, langs)
                    )
                    predictions = output.data.max(1)[1]

                    # update statistics
                    valid += predictions.eq(y).sum().item()
                    total += len(len1)

                # compute accuracy
                acc = 100.0 * valid / total
                scores["xnli_%s_%s_acc" % (splt, lang)] = acc
                logger.info(
                    "XNLI - %s - %s - Epoch %i - Acc: %.1f%%"
                    % (splt, lang, self.epoch, acc)
                )

        logger.info("__log__:%s" % json.dumps(scores))
        return scores

    def load_data(self):
        """
        Load XNLI cross-lingual classification data.
        """
        params = self.params
        data = {lang: {splt: {} for splt in ["test"]} for lang in XNLI_LANGS}
        label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        dpath = os.path.join(params.data_path, "eval", "XNLI")

        for splt in ["test"]:
            for lang in XNLI_LANGS:
                # only English has a training set
                if splt == "train" and lang != "en":
                    del data[lang]["train"]
                    continue

                # load data and dictionary
                data1 = load_binarized(
                    os.path.join(dpath, "%s.s1.%s.pth" % (splt, lang)), params
                )
                data2 = load_binarized(
                    os.path.join(dpath, "%s.s2.%s.pth" % (splt, lang)), params
                )
                data["dico"] = data.get("dico", data1["dico"])

                # set dictionary parameters
                set_dico_parameters(params, data, data1["dico"])
                set_dico_parameters(params, data, data2["dico"])

                # create dataset
                data[lang][splt]["x"] = ParallelDataset(
                    data1["sentences"],
                    data1["positions"],
                    data2["sentences"],
                    data2["positions"],
                    params,
                )

                # load labels
                with open(os.path.join(dpath, "%s.label.%s" % (splt, lang)), "r") as f:
                    labels = [label2id[l.rstrip()] for l in f]
                data[lang][splt]["y"] = torch.LongTensor(labels)
                assert len(data[lang][splt]["x"]) == len(data[lang][splt]["y"])

        return data
