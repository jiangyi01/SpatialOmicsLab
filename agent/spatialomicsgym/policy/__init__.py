"""The centrally managed security policy: what this platform may reach, install and write.

``egress.yaml`` is the policy; :mod:`spatialomicsgym.policy.egress` loads it and answers questions
about it. Every boundary that can enforce something reads the same answers, so the proxy, the
broker, ``utils/http_client.py`` and the recipe lens cannot disagree.
"""

from __future__ import annotations
