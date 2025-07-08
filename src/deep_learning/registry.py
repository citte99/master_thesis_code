# deep_learning/registry.py
class Registry:
    """A registry that maintains mappings from names to classes."""
    
    def __init__(self, name):
        self.name = name
        self._registry = {}
        
    def register(self, name=None):
        """Class decorator for registering components."""
        def decorator(cls):
            key = name if name is not None else cls.__name__
            self._registry[key] = cls
            return cls
        return decorator
        
    def get(self, name):
        """Get a registered class by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown {self.name}: '{name}'. Available options: {available}")
        return self._registry[name]
    
    def list_available(self):
        """List all registered components."""
        return list(self._registry.keys())


# Create registries
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")


