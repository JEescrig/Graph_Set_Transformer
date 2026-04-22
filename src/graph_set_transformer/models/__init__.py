from .models import (
    SetTransformerGraphClassifier,
    DeepSetGraphClassifier,
    SetGraphClassifier,
    GCNGraphClassifier,
    SetTransformerGraphMultiTask,
    DeepSetGraphMultiTask,
    SetGraphMultiTask,
)
from .model_dropout import (
    SetTransformerGraphClassifier as SetTransformerDropout,
    DeepSetGraphClassifier as DeepSetDropout,
    SetGraphClassifier as SetGraphDropout,
    GCNGraphClassifier as GCNDropout,
)

GraphSetTransformerGraphClassifier = SetGraphClassifier
GraphSetTransformerGraphMultiTask = SetGraphMultiTask
