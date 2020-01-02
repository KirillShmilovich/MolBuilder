"""
utils.py
Molecule Builder

Some utility functions
"""

import os

import mdtraj as md
import networkx as nx

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    return os.path.join(_ROOT, "data", path)


def parse_pdb(pdb_fname, base_fname=None):
    """Returns a graph object with available connectivity information"""
    trj = md.load(pdb_fname)
    G = make_bondgraph(trj.top)
    connect_dict = get_pdb_connect_dict(pdb_fname)

    if len(connect_dict) != trj.top.n_atoms:
        raise ValueError(
            f"Error reading {pdb_fname}. Connect dict ({len(connect_dict)}) != {trj.top.n_atoms} atoms"
        )

    connect_dict = {trj.top.atom(k): v for k, v in connect_dict.items()}
    nx.set_node_attributes(G, connect_dict, "connect")

    if base_fname is not None:
        G_base = parse_pdb(pdb_fname=base_fname, base_fname=None)
        mapping = subgraph_match(G, G_base)[0]
        base_connect = {k: G_base.nodes[v]["connect"] for k, v in mapping.items()}
        nx.set_node_attributes(G, base_connect, "connect")
    return G


def make_bondgraph(top):
    """Returns a bond graph from topology"""
    G = nx.Graph()
    G.add_nodes_from(top.atoms)
    G.add_edges_from(top.bonds)
    element_dict = {atom: atom.element.symbol for atom in G.nodes}
    nx.set_node_attributes(G, element_dict, "element")
    return G


def subgraph_match(G1, G2):
    def element_match(n1, n2):
        if n1["element"] == n2["element"]:
            return True
        return False

    GM = nx.algorithms.isomorphism.GraphMatcher(G1, G2, element_match)

    if GM.subgraph_is_isomorphic():
        return list(GM.subgraph_isomorphisms_iter())
    else:
        raise ValueError("No matching subgraphs")


def get_base_fname(fragment_type):
    if fragment_type == "residue":
        base_fname = get_data("residues/base_residue.pdb")
    else:
        raise ValueError(f"fragment type {fragment_type} not recognized")
    return base_fname


def get_pdb_connect_dict(pdb_fname):
    """Parses PDB file"""
    connect = dict()
    idx_counter = 0
    with open(pdb_fname) as f:
        for line in f:
            split_line = line.split()
            if (split_line[0] == "HETATM") or (split_line[0] == "ATOM"):
                if "C:" in split_line[-1]:
                    connection = split_line[-1].split("C:")[-1]
                else:
                    connection = None
                connect[idx_counter] = connection
                idx_counter += 1
    return connect


if __name__ == "__main__":
    g = parse_pdb("data/residues/Ala.pdb", base_fname="data/residues/base_residue.pdb")
