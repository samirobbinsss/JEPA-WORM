"""Per-dataset loader implementations.

Each loader module exports a class returning :class:`wormjepa.data.DatasetSample`
instances via the iterator protocol. Loaders are implemented in Stories 2.4-2.8;
this package currently holds skeletons that raise ``NotImplementedError`` with
clear story references.
"""
