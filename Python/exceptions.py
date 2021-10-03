"""

Authors: Oleksandr Perederii, Anatolii Kashchuk
"""

class BaseParticleException(Exception):
    pass

class ParticleException(Exception):
    pass

class FilenameError(BaseParticleException):
    pass

class ThrtypeError(BaseParticleException):
    pass

class ThrError(BaseParticleException):
    pass

class RError(BaseParticleException):
    pass

class RoiError(BaseParticleException):
    pass

class PosAngleError(BaseParticleException):
    pass

class DzError(BaseParticleException):
    pass

class ZError(BaseParticleException):
    pass