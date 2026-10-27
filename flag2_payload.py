import hashlib as _h
import base64 as _b

_s1 = _b.b64decode('PT09PSBZb3UgZm91bmQgRmxhZyAyISA9PT09').decode()
_s2 = _b.b64decode('RW50ZXIgeW91ciBHVElEOiA=').decode()
_s3 = _b.b64decode('RXJyb3I6IEdUSUQgY2Fubm90IGJlIGVtcHR5LiBQbGVhc2UgdHJ5IGFnYWluLg==').decode()
_s4 = _b.b64decode('CllvdXIgaGFzaCBpczog').decode()
_k = b'X0NTNjAzNS1GTDIwMjYtTlZNLQ=='

def _f1(_x):
    _a = _h.sha256(_x.encode()).hexdigest()
    _c = _h.sha256(_b.b64decode(_k)).hexdigest()
    return _a + _c

def _f2():
    print(_s1)
    _i = input(_s2).strip()
    if not _i:
        print(_s3)
        return
    _r = _f1(_i)
    print(_s4, _r)

if __name__ == '__main__':
    _f2()