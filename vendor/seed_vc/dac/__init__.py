__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

try:
    import audiotools
except ModuleNotFoundError:
    audiotools = None

if audiotools is not None:
    audiotools.ml.BaseModel.INTERN += ["dac.**"]
    audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
