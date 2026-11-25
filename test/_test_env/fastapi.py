
class HTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FastAPI:
    def __init__(self, *a, **k):
        pass

    def _deco(self, *a, **k):
        def wrap(f):
            return f
        return wrap

    get = _deco
    post = _deco
    delete = _deco
