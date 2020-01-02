"""
fragment.py
Molecule Builder

Fragment class
"""

from abc import ABC, abstractclassmethod

from MolBuilder.utils import get_base_fname, parse_pdb


class AbstractFragment(ABC):
    @abstractclassmethod
    def _parse_pdb(self):
        pass

    @abstractclassmethod
    def connect(self, cpoint, frag, frag_cpoint):
        pass


class Fragment(AbstractFragment):
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

    def connect(self, cpoint, frag, frag_cpoint):
        pass


class Residue(Fragment):
    def _parse_pdb(self):
        self.fragment_type = "residue"
        base_fname = get_base_fname(self.fragment_type)
        return parse_pdb(self.pdb_fname, base_fname)


if __name__ == "__main__":
    import MolBuilder as MB

    A = MB.Residue("data/residues/Ala.pdb")
