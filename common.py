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

import pandas as pd


def loadShot(shot):
    names = ['videoID', 'shotNumber', 'startTime', 'endTime',
             'startFrame', 'endFrame']
    return pd.read_table(shot, sep=' ', names=names, index_col=[0, 1])


def loadLabel(label):
    names = ['videoID', 'shotNumber', 'personName', 'confidence']
    return pd.read_table(label, sep=' ', names=names)


def loadEvidence(evidence):
    names = ['personName', 'videoID', 'shotNumber', 'source']
    return pd.read_table(evidence, sep=' ', names=names)


def loadLabelReference(reference):
    names = ['videoID', 'shotNumber', 'personName']
    return pd.read_table(reference, sep=' ', names=names)


def loadEvidenceReference(evireference):
    names = ['videoID', 'shotNumber', 'personName', 'source']
    return pd.read_table(evireference, sep=' ', names=names)
