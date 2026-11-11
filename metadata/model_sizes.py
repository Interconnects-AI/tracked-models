"""Deprecated: parameter counts live only in model_parameters.csv.

This file is kept because the Interconnects API reads a ``MANUAL_SIZES`` mapping
from this path on every cache refresh and fails without one. The mapping is
empty on purpose: the API takes every reviewed count from model_parameters.csv
first, so nothing here was ever used.

Delete this file once the API no longer reads it.
"""

MANUAL_SIZES = {}
