"""Doctor: measurement, diagnosis, prescription.

Doctor is not a second copy of the HAWKING control plane. The instruments
live as ``tools/doctor_seal.py`` and ``tools/gravity_doctor_*.py``
(nos_pipeline stage 3). This package is the HAWKING-side ownership marker
so those files are not imported under a second dotted identity.
"""

OWNED_PATHS = (
    "tools/doctor_seal.py",
    "tools/gravity_doctor_capability.py",
    "tools/gravity_doctor_dimensions.py",
    "tools/gravity_doctor_gate.py",
)
