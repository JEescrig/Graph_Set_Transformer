from .models import (
    SetTransformerGraphClassifier,
    DeepSetGraphClassifier,
    SetGraphClassifier,
    GCNGraphClassifier,
    SetTransformerGraphSetElementClassifier,
    DeepSetGraphSetElementClassifier,
    SetGraphSetElementClassifier,
)
from .model_dropout import (
    SetTransformerGraphClassifier as SetTransformerDropout,
    DeepSetGraphClassifier as DeepSetDropout,
    SetGraphClassifier as SetGraphDropout,
    GCNGraphClassifier as GCNDropout,
)

GraphSetTransformerGraphClassifier = SetGraphClassifier
GraphSetTransformerGraphSetElementClassifier = SetGraphSetElementClassifier
