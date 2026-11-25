
class BaseModel:
    def __init__(self, **data):
        ann = {}
        for klass in reversed(type(self).__mro__):
            ann.update(getattr(klass, "__annotations__", {}) or {})
        for name in ann:
            setattr(self, name, data.get(name, None))
        for k, v in data.items():
            setattr(self, k, v)
