"""Pytest configuration.

Force a headless matplotlib backend so the test suite never depends on a
working Tk/GUI installation (the Windows Store Python ships a broken Tcl/Tk).
"""

import matplotlib

matplotlib.use("Agg")
