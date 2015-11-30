# MediaEval 2015 "Person Discovery" evaluation package

This package provides all the necessary data and tools to score submissions at the MediaEval Person Discovery 2015 task (see `MediaEvalPersonDiscovery2015.pdf` for a detailed description of the task).

## Installation

```bash
$ git clone https://github.com/MediaevalPersonDiscoveryTask/EvaluationPackage2015.git
$ pip install -r requirements.txt
```

The test set can be obtained from INA using the online form available [here](http://dataset.ina.fr). It contains the entirety of the TV broadcast news of the *"20 heures de France 2"* from the January 1st to June 30th, 2007 (see `all.lst`).

## MAP | Mean average precision

The test set has been segmented automatically into shots.
However, participants only had to process a subset made of shots longer than 2 seconds (see `2015.shot`).

Only half of the test set was annotated for 2015 edition, including videos from January 1st to March 31st, 2007 (see `2015.lst`).

To achieve high annotation quality, we asked for at least two annotations per shot from at least two annotators. In case no consensus was reached for a particular shot, we asked for additional annotations until reaching a consensus (see ``2015.consensus`` for the list of shots for which a consensus was reached).

For each shot in ``2015.consensus``, [``2015.ref``](https://github.com/MediaevalPersonDiscoveryTask/evaluation/wiki/File-format#reference-ref) provides the list of identities of talking faces.

Mean average precision can be computed using ``map.py``.
For instance, let's evaluate the performance of the [`baseline`](https://github.com/MediaevalPersonDiscoveryTask/baseline) system:

```bash
$ python map.py --queries=2015.queries \
                --consensus=2015.consensus \
                2015.shot 2015.ref \
                baseline.label
74.89 %
```

## EwMAP | Evidence-weighted mean average precision

For various reasons, we were not able to fully annotate the test set in terms of *evidences*. Only those submitted by participants in 2015 were manually checked. [`2015.eviref`](https://github.com/MediaevalPersonDiscoveryTask/evaluation/wiki/File-format#evidence-reference-eviref) therefore provides incomplete evidence groundtruth. To evaluate performance of new runs, you need to manually edit `2015.eviref` to add missing evidences your runs might have discovered. If you do so, please consider sharing these new annotations.

Evidence-weighted average precision can then be computed using ``ewmap.py``. For instance, let's evaluate the performance of the [`baseline`](https://github.com/MediaevalPersonDiscoveryTask/baseline) system:

```bash
$ python ewmap.py --queries=2015.queries \
                  --consensus=2015.consensus \
                  2015.shot 2015.ref 2015.eviref \
                  baseline.label baseline.evidence
74.63 %
```

## CHANGELOG

#### 2015-11-30 / Version 0.1
   - First public release
   - Annotation dump: 2015-09-09, 15:04
