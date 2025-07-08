# deep_learning/NN_models/__init__.py
from deep_learning.registry import MODEL_REGISTRY

# Import all models to register them
from .resnet18 import ResNet18
from .resnet50 import ResNet50
from .transformer import VisionTransformer
from .efficient_net_B0 import EfficientNetB0
from .sparse_resnet50 import SparseResNet50