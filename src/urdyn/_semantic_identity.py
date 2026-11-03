"""The engine's versioned normalization identity.

Defined once, in a module with no third-party imports, because two modules
must agree on it: `_semantic` (NumPy; it implements the contract and persists
the identity with every index) and the public, NumPy-free `semantic_backend`
(an external backend's declared identity defaults to it). See
`_semantic.SEMANTIC_NORMALIZATION` for what the identity means and why the
legacy value is refused.
"""

SEMANTIC_NORMALIZATION = "l2-scaled-f64-v1"
LEGACY_SEMANTIC_NORMALIZATION = "l2"
