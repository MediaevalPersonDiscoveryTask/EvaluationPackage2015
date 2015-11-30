#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


"""
MediaEval Person Discovery Task evaluation -- MAP

Usage:
  evaluation [options] <reference.shot> <reference.ref> <hypothesis.label>

Options:
  -h --help                     Show this screen.
  --version                     Show version.
  --queries=<queries.lst>       Query list.
  --levenshtein=<threshold>     Levenshtein ratio threshold [default: 0.95]
  --consensus=<consensus.shot>  Label-annotated subset of <reference.shot>
"""

from __future__ import print_function

from docopt import docopt
from Levenshtein import ratio
import numpy as np

from common import loadShot, loadLabel
from common import loadLabelReference


def loadFiles(shot, reference, label, consensus=None):

    shot = loadShot(shot)
    label = loadLabel(label)

    # check that labels are only provided for selected shots
    labelShots = set(
        tuple(s) for _, s in label[['videoID', 'shotNumber']].iterrows())
    if not labelShots.issubset(set(shot.index)):
        msg = ('Labels should only be computed for provided shots.')
        raise ValueError(msg)

    # only keep labels for shot with consensus
    if consensus:
        consensus = loadShot(consensus)
        mask = label.apply(
            lambda x: (x['videoID'], x['shotNumber']) in set(consensus.index),
            axis=1)
        label = label[mask]

    reference = loadLabelReference(reference)

    return reference, label


def closeEnough(personName, query, threshold):
    return ratio(query, personName) >= threshold


def computeAveragePrecision(vReturned, vRelevant):

    nReturned = len(vReturned)
    nRelevant = len(vRelevant)

    if nRelevant == 0:
        return 1.

    if nReturned == 0:
        return 0.

    returnedIsRelevant = np.array([item in vRelevant for item in vReturned])
    precision = np.cumsum(returnedIsRelevant) / (1. + np.arange(nReturned))

    return np.sum(precision * returnedIsRelevant) / nRelevant


if __name__ == '__main__':

    arguments = docopt(__doc__, version='0.1')

    shot = arguments['<reference.shot>']
    reference = arguments['<reference.ref>']
    label = arguments['<hypothesis.label>']
    threshold = float(arguments['--levenshtein'])
    consensus = arguments['--consensus']

    reference, label = loadFiles(
        shot, reference, label, consensus=consensus)

    if arguments['--queries']:
        with open(arguments['--queries'], 'r') as f:
            queries = [line.strip() for line in f]

    else:
        # build list of queries from reference
        queries = sorted(set(reference['personName'].unique()))

    # query --> averagePrecision dictionary
    averagePrecision = {}
    correctness = {}

    labels = set([])
    for p in label['personName']:
        labels.add(p)

    for query in queries:

        # find most similar personName
        ratios = [(personName, ratio(query, personName))
                  for personName in labels]
        best = sorted(ratios, key=lambda x: x[1], reverse=True)[0]

        personName = best[0] if best[1] > threshold else None

        # =====================================================================
        # Evaluation of LABELS
        # =====================================================================

        # get relevant shots for this query, according to reference
        qRelevant = reference[reference.personName == query]
        qRelevant = qRelevant[['videoID', 'shotNumber']]
        qRelevant = set((videoID, shotNumber)
                        for _, videoID, shotNumber in qRelevant.itertuples())

        # get returned shots for this query
        # (i.e. shots containing closest personName)
        qReturned = label[label.personName == personName]

        # this can only happen with --consensus option
        # when hypothesis contains shot in the out of consensus part
        # hack to solve this corner case
        if len(qReturned) == 0:
            averagePrecision[query] = 0. if len(qRelevant) > 0. else 1.

        else:
            # sort shots by decreasing confidence
            # (in case of shots returned twice for this query, keep maximum)
            qReturned = (qReturned.groupby(['videoID', 'shotNumber'])
                                  .aggregate(np.max)
                                  .sort_values(by=['confidence'], ascending=False))

            # get list of returned shots in decreasing confidence
            qReturned = list(qReturned.index)

            # compute average precision for this query
            averagePrecision[query] = computeAveragePrecision(qReturned,
                                                              qRelevant)

    MAP = np.mean([averagePrecision[query] for query in queries])

    print('%.2f %%' % (100 * MAP))
