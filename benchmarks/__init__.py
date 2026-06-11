"""TrueMemory benchmark suites (regular package, not a PEP 420 namespace).

Issue #654 (M-81): keeping ``benchmarks`` a *regular* package (with this
``__init__.py``) prevents an unrelated editable-installed ``benchmarks``
distribution on a contributor's machine from shadowing the repo's package and
breaking ``from benchmarks... import ...`` in the test suite.
"""
