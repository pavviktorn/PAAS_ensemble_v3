"""PAAS_ensemble_v3 -- 5-detector face-liveness ensemble:
FFAA (MLLM+MIDS) + 9-class members A1/A2 + GSD + SeLop, fused by mean.

Runs on the global python3.12 / transformers==4.37.2 interpreter. Typical use:

    from paas.config import PaasConfig
    from paas.pipeline import PaasPipeline
    pipe = PaasPipeline(PaasConfig.from_file("config/experiments/paas5_mean.json"))
    results = pipe.predict_images(["a.jpg", "b.jpg"])
"""
from .config import PaasConfig            # noqa: F401
