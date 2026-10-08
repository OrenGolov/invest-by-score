"""Repository scripts, importable as a package.

An explicit `__init__.py` rather than a namespace package: MEASURED, without
it `scripts` resolved to a namespace path that also contained
`site-packages/win32/scripts`, so a module name added there could shadow one
of ours. A regular package binds the name to this directory only.
"""
