from collections import OrderedDict
from typing import Any, Dict, OrderedDict as OrderedDictType

from .cifar10 import cifar_10_models
from .cifar100 import cifar_100_models
from .enums import BenchmarkDataset, ThreatModel

ModelsDict = OrderedDictType[str, Dict[str, Any]]
ThreatModelsDict = OrderedDictType[ThreatModel, ModelsDict]
BenchmarkDict = OrderedDictType[BenchmarkDataset, ThreatModelsDict]

model_dicts: BenchmarkDict = OrderedDict([
    (BenchmarkDataset.cifar_10, cifar_10_models),
    (BenchmarkDataset.cifar_100, cifar_100_models)
])
