"""
fragment.py
Molecule Builder

Fragment class
"""

from MolBuilder.utils import parse_pdb, get_base_fname


class Fragment:
    def __init__(self, pdb_fname, fragment_type=None):
        self.pdb_fname = pdb_fname
        self.fragment_type = fragment_type
        self.G = self._parse_pdb()

    def _parse_pdb(self):
        if self.fragment_type is not None:
            base_fname = get_base_fname(self.fragment_type)
        else:
            base_fname = None
        return parse_pdb(self.pdb_fname, base_fname)


class Residue(Fragment):
    def _parse_pdb(self):
        self.fragment_type = "residue"
        base_fname = get_base_fname(self.fragment_type)
        return parse_pdb(self.pdb_fname, base_fname)
