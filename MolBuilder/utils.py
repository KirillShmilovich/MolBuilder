"""
utils.py
Molecule Builder

Some utility functions
"""

import os

import mdtraj as md
import networkx as nx
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    return os.path.join(_ROOT, "data", path)


def parse_pdb(pdb_fname, base_fname=None):
    """Returns a graph object with available connectivity information"""
    trj = md.load(pdb_fname).center_coordinates()
    G = make_bondgraph(trj.top)
    connect_dict = get_pdb_connect_dict(pdb_fname)

    if len(connect_dict) != trj.top.n_atoms:
        raise ValueError(
            f"Error reading {pdb_fname}. Connect dict ({len(connect_dict)}) != {trj.top.n_atoms} atoms"
        )

    connect_dict = {trj.top.atom(k): v for k, v in connect_dict.items()}
    nx.set_node_attributes(G, connect_dict, "connect")

    xyz_dict = {atom: xyz for atom, xyz in zip(trj.top.atoms, trj.xyz[0])}
    nx.set_node_attributes(G, xyz_dict, "xyz")

    if base_fname is not None:
        G_base = parse_pdb(pdb_fname=base_fname, base_fname=None)
        mapping = subgraph_match(G, G_base)[0]
        base_connect = {k: G_base.nodes[v]["connect"] for k, v in mapping.items()}
        nx.set_node_attributes(G, base_connect, "connect")

    edge_idxs = np.empty(shape=(G.number_of_edges(), 2), dtype=np.int)
    for i, edge in enumerate(G.edges):
        edge_idxs[i] = edge[0].index, edge[1].index
    edge_distances = md.compute_distances(trj, edge_idxs)[0]
    distance_dict = {edge: dis for edge, dis in zip(G.edges, edge_distances)}
    nx.set_edge_attributes(G, distance_dict, "distance")
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


def dict_compare(ref_dict, compare_dict):
    """Returns True if key-value pairs in ref_dict are same as compare_dict"""
    for k, v in ref_dict.items():
        if compare_dict[k] != v:
            return False
    return True


def flatten_list(a):
    return sum(a, [])


def rotation_matrix(alpha, beta, gamma):
    """ Returns rotation matrix for transofmorming N x 3 coords (A @ R)"""
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    R = Rz @ Ry @ Rx
    Rt = R.T
    return Rt


def rotation_matrix_batch(alpha, beta, gamma):
    """ Returns rotation matrix for transofmorming N x 3 coords (A @ R)"""
    n = len(alpha)

    def ef(x):
        a = np.empty(n)
        a.fill(x)
        return a

    Rx = np.array(
        [
            [ef(1), ef(0), ef(0)],
            [ef(0), np.cos(alpha), -np.sin(alpha)],
            [ef(0), np.sin(alpha), np.cos(alpha)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(beta), ef(0), np.sin(beta)],
            [ef(0), ef(1), ef(0)],
            [-np.sin(beta), ef(0), np.cos(beta)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), ef(0)],
            [np.sin(gamma), np.cos(gamma), ef(0)],
            [ef(0), ef(0), ef(1)],
        ]
    )
    # 'li' not 'il' b/c transpose
    Rt = np.einsum("ij...,jk...,kl...->li...", Rz, Ry, Rx)
    return Rt


def get_rand_unit_vecs(N):
    x = np.random.randn(N, 3)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rotation_matrix_axis(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    R = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    return R.T


def correct_xyz(A, B, eq_idxs, d_max, n_restarts=15):
    for restart_i in range(n_restarts):
        N = 100
        node_0_i, node_1_j = eq_idxs

        thetas = 2 * np.pi * np.random.rand(N, 3)
        Rs = rotation_matrix_batch(thetas[:, 0], thetas[:, 1], thetas[:, 2])

        Bprime = np.einsum("ij,jk...->ik...", B, Rs)
        t = (A[node_0_i].reshape(3, 1) - Bprime[node_1_j]).reshape(1, 3, N)
        Bprime += t

        u = get_rand_unit_vecs(N)
        T = Bprime.reshape(*Bprime.shape, 1) + d_max * u.reshape(1, 3, 1, N)
        T = T.reshape(Bprime.shape[0], Bprime.shape[1], -1)

        diffmat = np.expand_dims(T, axis=1) - np.expand_dims(
            A.reshape(*A.shape, 1), axis=0
        )
        D = np.sqrt(np.sum(np.square(diffmat), axis=2))
        C = np.linalg.norm(D, axis=(0, 1))
        # C = np.linalg.norm(T.mean(axis=0) - A.mean(axis=0).reshape(3, 1), axis=0)
        best_idx = C.argmax()

        T_best = T[..., best_idx]

        T_best_com = T_best.mean(axis=0, keepdims=True)
        u = A[node_0_i] - T_best[node_1_j]
        T_best -= T_best_com

        Rs = [rotation_matrix_axis(u, theta) for theta in np.linspace(0, 2 * np.pi, N)]
        Rs = np.stack(Rs, axis=-1)
        T = np.einsum("ij,jk...->ik...", T_best, Rs)
        T += T_best_com.reshape(1, 3, 1)

        diffmat = np.expand_dims(T, axis=1) - np.expand_dims(
            A.reshape(*A.shape, 1), axis=0
        )
        D = np.sqrt(np.sum(np.square(diffmat), axis=2))
        C = np.linalg.norm(D, axis=(0, 1))
        # C = np.linalg.norm(T.mean(axis=0) - A.mean(axis=0).reshape(3, 1), axis=0)
        best_idx = C.argmax()

        val = C[best_idx]
        best = T[..., best_idx]

        if restart_i == 0:
            best_final = best
            val_final = val
        else:
            if val < val_final:
                best_final = best
                val_final = val

    return best_final


def correct_xyz_notWorking(A, B, eq_idxs, d_max, n_restarts=9):
    node_0_i, node_1_j = eq_idxs

    def obj_fun(x):
        alpha, beta, gamma, t0, t1, t2 = x
        R = rotation_matrix(alpha, beta, gamma)
        t = np.array([t0, t1, t2]).reshape(1, 3)
        Bprime = B @ R + t
        Y = cdist(A, Bprime)
        return -1 * np.linalg.norm(Y)

    def cons_f(x):
        alpha, beta, gamma, t0, t1, t2 = x
        R = rotation_matrix(alpha, beta, gamma)
        t = np.array([t0, t1, t2]).reshape(1, 3)
        Bprime = B @ R + t
        # return np.linalg.norm(A[i] - Bprime[j])
        return np.abs(np.linalg.norm(A[node_0_i] - Bprime[node_1_j]) - d_max)

    # cons = NonlinearConstraint(cons_f, 0, d_max)
    cons = {"type": "eq", "fun": cons_f}
    bounds = Bounds(
        [0, 0, 0, 0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi, np.inf, np.inf, np.inf]
    )
    for i in range(n_restarts):
        x0 = np.concatenate([np.random.rand(3), np.random.randn(3) + 2])
        res = minimize(obj_fun, x0, method="SLSQP", constraints=cons, bounds=bounds)
        if i == 0:
            res_best = res
        if res.fun < res_best.fun:
            res_best = res

    alpha, beta, gamma, t0, t1, t2 = res.x
    R = rotation_matrix(alpha, beta, gamma)
    t = np.array([t0, t1, t2]).reshape(1, 3)
    Bprime = B @ R + t
    return Bprime


def write_optim_input(frag, name="fragment.com", n=40, mem="155GB"):
    f = open(name, "w")
    f.write(f"%CPU=0-{n-1}\n")
    f.write(f"%mem={mem}\n")
    f.write("#p opt=(NoLinear,MaxStep=5) guess=indo b3lyp/6-31G(d)\n")
    f.write("\n")
    f.write("Fragment\n")
    f.write("\n")
    f.write("0 1\n")
    for i, node in frag.G.nodes(data=True):
        line = (
            str(node["element"])
            + " "
            + str(round(10 * node["xyz"][0], 4))
            + " "
            + str(round(10 * node["xyz"][1], 4))
            + " "
            + str(round(10 * node["xyz"][2], 4))
        )
        f.write(line)
        f.write("\n")
    f.write("\n")
    f.close()


if __name__ == "__main__":
    A = np.random.rand(10, 3)
    B = np.random.rand(11, 3)
    print(correct_xyz(A, B, (1, 2), 1.0))
