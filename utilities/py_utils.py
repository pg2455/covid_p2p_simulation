import weakref


class InstanceRegistry(type):
    def __init__(cls, *args, **kwargs):
        super(InstanceRegistry, cls).__init__(*args, **kwargs)
        cls._instances = weakref.WeakSet()

    def __call__(cls, *args, **kwargs):
        instance = super(InstanceRegistry, cls).__call__(*args, **kwargs)
        cls._instances.add(instance)
        return instance


def instances_of(cls):
    for obj in cls._instances:
        yield obj

