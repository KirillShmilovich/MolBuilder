"""
fragment.py
Molecule Builder

Fragment class
"""

from abc import ABC, abstractclassmethod
from copy import deepcopy

import mdtraj as md
import networkx as nx
import numpy as np

from MolBuilder.utils import (
    correct_xyz,
    dict_compare,
    flatten_list,
    get_base_fname,
    parse_pdb,
    write_optim_input,
)


class AbstractFragment(ABC):
    @abstractclassmethod
    def _parse_pdb(self):
        pass

    @abstractclassmethod
    def connect(self, cpoint, frag, frag_cpoint):
        pass


class Fragment(AbstractFragment):
    def __init__(
        self, pdb_fname, fragment_type=None, name=None, fragment_number=0, G=None
    ):
        self.pdb_fname = pdb_fname
        self.top = md.load(pdb_fname).top
        self.fragment_type = fragment_type
        self.G = self._parse_pdb()
        self._rename(name)
        self._renumber(fragment_number)

        self.G = nx.convert_node_labels_to_integers(self.G)
        self.fragments = [self.G]

    @property
    def n_fragments(self):
        return len(self.fragments)

    @property
    def trj(self):
        xyz = [node["xyz"] for i, node in self.G.nodes(data=True)]
        xyz = np.vstack(xyz)
        trj = md.Trajectory(xyz, self.top)
        return trj

    def save_pdb(self, fname):
        trj = self.trj
        trj.save_pdb(fname)

    def save_xyz(self, fname):
        f = open(fname, "w")
        f.write(str(self.G.number_of_nodes()) + "\n")
        f.write("\n")
        for i, node in self.G.nodes(data=True):
            line = (
                str(node["element"])
                + " "
                + str(10 * node["xyz"][0])
                + " "
                + str(10 * node["xyz"][1])
                + " "
                + str(10 * node["xyz"][2])
            )
            f.write(line)
            f.write("\n")
        f.close()

    def _parse_pdb(self):
        if self.fragment_type is not None:
            base_fname = get_base_fname(self.fragment_type)
        else:
            base_fname = None
        return parse_pdb(self.pdb_fname, base_fname)

    def _rename(self, name):
        if name is None:
            if self.fragment_type is None:
                name = "fragment"
            else:
                name = str(self.fragment_type)

        name_mapping = {node: name for node in self.G.nodes}
        nx.set_node_attributes(self.G, name_mapping, "fragment")

    def _renumber(self, fragment_number):
        number_mapping = {node: fragment_number for node in self.G.nodes}
        nx.set_node_attributes(self.G, number_mapping, "fragment_number")

    def connect(self, Frag, connect_dict):
        Frag_G = deepcopy(Frag.G)
        Frag_top = deepcopy(Frag.top)

        if len(connect_dict) != 2:
            raise ValueError(f"size of connect dict != 2 ({len(connect_dict)})")
        # Get conditions for connection
        condition_0, condition_1 = connect_dict

        # Find nodes meeting condition
        nodes_0 = [
            node
            for node, node_dict in self.G.nodes(data=True)
            if dict_compare(condition_0, node_dict)
        ]
        nodes_1 = [
            node
            for node, node_dict in Frag_G.nodes(data=True)
            if dict_compare(condition_1, node_dict)
        ]

        # Find connecting nodes
        connect_node_0 = [
            [n for n in self.G[node] if n not in nodes_0] for node in nodes_0
        ]
        connect_node_0 = flatten_list(connect_node_0)
        if len(connect_node_0) != 1:
            raise ValueError(f"Number of connect nodes not 1 ({len(connect_node_0)})")
        connect_node_0 = connect_node_0[0]

        connect_node_1 = [
            [n for n in Frag_G[node] if n not in nodes_1] for node in nodes_1
        ]
        connect_node_1 = flatten_list(connect_node_1)
        if len(connect_node_1) != 1:
            raise ValueError(f"Number of connect nodes not 1 ({len(connect_node_1)})")
        connect_node_1 = connect_node_1[0]

        # Find connecting edges
        connect_edges_0 = [[edge for edge in self.G.edges(node)] for node in nodes_0]
        connect_edges_0 = flatten_list(connect_edges_0)
        connect_edges_0 = [
            edge
            for edge in connect_edges_0
            if ((edge[0] not in nodes_0) or (edge[1] not in nodes_0))
        ]
        connect_dist_0 = [self.G[e0][e1]["distance"] for e0, e1 in connect_edges_0]
        connect_dist_0 = sum(connect_dist_0) / len(connect_dist_0)

        connect_edges_1 = [[edge for edge in Frag_G.edges(node)] for node in nodes_1]
        connect_edges_1 = flatten_list(connect_edges_1)
        connect_edges_1 = [
            edge
            for edge in connect_edges_1
            if ((edge[0] not in nodes_1) or (edge[1] not in nodes_1))
        ]
        connect_dist_1 = [Frag_G[e0][e1]["distance"] for e0, e1 in connect_edges_1]
        connect_dist_1 = sum(connect_dist_1) / len(connect_dist_1)

        connect_dist = (connect_dist_0 + connect_dist_1) / 2.0

        A = [node["xyz"] for i, node in self.G.nodes(data=True)]
        A = np.vstack(A)

        B = [node["xyz"] for i, node in Frag_G.nodes(data=True)]
        B = np.vstack(B)

        Bprime = correct_xyz(A, B, (connect_node_0, connect_node_1), connect_dist)
        xyz_mapping = {node: xyz for node, xyz in zip(Frag_G.nodes, Bprime)}
        nx.set_node_attributes(Frag_G, xyz_mapping, "xyz")

        number_mapping = {
            i: node["fragment_number"] + self.n_fragments
            for i, node in Frag_G.nodes(data=True)
        }
        nx.set_node_attributes(Frag_G, number_mapping, "fragment_number")
        self.fragments.append(Frag_G)

        n_G_nodes = self.G.number_of_nodes()
        self.G = nx.disjoint_union(self.G, Frag_G)
        self.G.add_edge(
            connect_node_0,
            connect_node_1 + n_G_nodes,
            distance=np.linalg.norm(A[connect_node_0] - Bprime[connect_node_1]),
        )

        at0 = [atom for atom in self.top.atoms if atom.index == connect_node_0][0]
        self.top = self.top.subset(
            [atom.index for atom in self.top.atoms if atom.index not in nodes_0]
        )

        at1 = [atom for atom in Frag_top.atoms if atom.index == connect_node_1][0]
        Frag_top = Frag_top.subset(
            [atom.index for atom in Frag_top.atoms if atom.index not in nodes_1]
        )

        self.top = self.top.join(Frag_top)
        self.top.add_bond(at0, at1)

        # Remove nodes
        remove_nodes = nodes_0 + [n + n_G_nodes for n in nodes_1]
        self.G.remove_nodes_from(remove_nodes)
        self.G = nx.convert_node_labels_to_integers(self.G)

        return self


class Residue(Fragment):
    def _parse_pdb(self):
        self.fragment_type = "residue"
        base_fname = get_base_fname(self.fragment_type)
        return parse_pdb(self.pdb_fname, base_fname)


if __name__ == "__main__":
    import MolBuilder as MB

    A = MB.Residue("/home/kirills/Projects/MolBuilder/MolBuilder/data/residues/Ala.pdb")
    V = MB.Residue("/home/kirills/Projects/MolBuilder/MolBuilder/data/residues/Val.pdb")
    C = MB.Residue("/home/kirills/Projects/MolBuilder/MolBuilder/data/residues/Cys.pdb")
    G = MB.Residue("/home/kirills/Projects/MolBuilder/MolBuilder/data/residues/Gly.pdb")
    A.connect(V, ({"connect": "N"}, {"connect": "C"}))
    A.connect(C, ({"connect": "N", "fragment_number": 1}, {"connect": "C"}))
    A.connect(G, ({"connect": "N", "fragment_number": 2}, {"connect": "C"}))
    A.save_xyz("test.xyz")
    A.save_pdb("test.pdb")
    write_optim_input(A)
    print(A.G.nodes(data=True))
