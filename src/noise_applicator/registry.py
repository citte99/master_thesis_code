
class Registry:
    """A registry that maintains mappings from names to classes."""
    
    def __init__(self, name):
        self.name = name
        self._registry = {}
        self.registry_mode = None
    def register(self, name=None):
        """Class decorator for registering components."""
        if self.registry_mode is None:
            self.registry_mode = "cls"
        elif self.registry_mode == "instance":
            raise TypeError (f"The register {self.name} is stroring istances, not classes")
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

    def add_instance(self, name, instance):
        if self.registry_mode is None:
            self.registry_mode = "instance"
        elif self.registry_mode == "cls":
            raise TypeError (f"The register {self.name} is stroring classes, not istances")
        self._registry[name] = instance


NOISERS_REGISTRY= Registry('noiser')
INSTRUMENTS_REGISTRY = Registry('instrumet')