"""The contract major, and the version the wheel metadata reads.

``CONTRACT_MAJOR`` is the one hand-set number: raise it by one for a breaking
change. ``__version__`` stays the placeholder ``"<CONTRACT_MAJOR>.0.0"`` in Git;
publication computes the released version from the change class
(``tools/contract-version``) and stamps it here in its own workspace only.
"""

CONTRACT_MAJOR = 5
__version__ = "5.0.0"
