"""
Target-Guided Adaptive Vertex Repair (TGAVR).

Optional, training-free pre-processing of a diffusion-generated base-pair point
cloud before the *unchanged* Generative SNUPI scaffold routing
(``automated_design.auto_design.automated_DnaOrigami``).

Pipeline
    target graph (V, E)  ──┐
    generated points X  ───┴─> similarity alignment (target -> X; X is never moved)
                              -> chain extraction (same 0.4 nm rule as the router)
                              -> target-edge assignment of chains / points
                              -> vertex-local defect detection
                              -> vertex-local repair (trim / insert / resample)
                              -> [optional] bounded per-helix-end length search
                              -> existing routing

Conventions verified in the routing code (see docs/TGAVR_AUDIT.md):
  * coordinates are base-pair centres in nm;
  * consecutive base pairs of one helix are linked iff distance < 0.4 nm
    (``regular_BP_len`` in auto_design.py -> ``radius_graph(r=0.4)``);
  * ``pcl_filter`` deletes every point taking part in a link whose two end
    points have a summed degree > 4 (i.e. local over-density);
  * connected chains with <= 10 points are discarded (``segmentation(min_nodes=10)``);
  * each chain must be a simple path (``edge2route``) - a branch corrupts indexing;
  * every target edge is realised as a two-helix bundle (2HB), helix axes about
    2.05 nm apart (measured on generated samples), axial rise 0.32-0.34 nm.

Only numpy / scipy / torch are used (all already project dependencies).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TGAVRConfig:
    # --- alignment -----------------------------------------------------------
    align_max_iter: int = 80
    align_tol: float = 1e-7            # relative change of the objective
    align_trim: float = 0.90           # fraction of correspondences kept (robust)
    align_interior_u: float = 0.2      # curve->points term uses u in [0.2, 0.8] of each edge
    align_centroid_radius: float = 2.5 # nm, local centroid radius for curve->points
    align_coarse_points: int = 600     # subsample size for the multi-start stage
    align_coarse_iter: int = 8
    align_refine_top: int = 3
    align_target_spacing: float = 0.5  # nm, sampling of target edges (aligned frame)
    align_max_rmse: float = 2.5        # nm, X -> target-curve RMSE acceptance limit
    align_min_coverage: float = 0.85   # fraction of target curve near generated points
    align_cover_tol: float = 3.0       # nm
    # --- routing conventions (do not change unless the router changes) -------
    bp_rise: float = 0.34              # nm, nominal axial rise used for insertion
    link_radius: float = 0.4           # nm, regular_BP_len of the router
    min_chain_points: int = 11         # router keeps chains with > 10 points
    helices_per_edge: int = 2          # 2HB wireframe
    # --- vertex repair -------------------------------------------------------
    vertex_radius_factor: float = 0.3  # r_v = factor * shortest incident edge
    vertex_radius_min: float = 3.0     # nm
    vertex_radius_max: float = 8.0     # nm
    edge_assign_tol: float = 3.0       # nm, max point-to-edge distance for assignment
    end_clearance: float = 0.55        # nm, min distance between ends of distinct chains
    merge_bend_deg: float = 30.0       # merged chain is a defect only if the edges bend more
    axis_fit_points: int = 8           # interior points used to extrapolate an axis
    max_insert_per_end: int = 12       # bps inserted per helix end at most
    max_edit_fraction: float = 0.08    # reject the repair if more points are edited
    repair_interior_gaps: bool = True  # bridge collinear duplex breaks inside a target edge
    # --- topology recovery for point-cloud (.pt) targets (opt-in) ------------
    recover_pointcloud_topology: bool = False
    recover_link_factor: float = 1.5   # x median spacing (targets are sampled at 1.0 nm)
    recover_dp_tol: float = 0.5        # nm, Douglas-Peucker tolerance
    recover_junction_radius: float = 5.0  # nm, ends cut open by remove_intersection
    recover_contact_radius: float = 3.0   # nm, two paths touching away from their ends
    recover_min_path_points: int = 3
    # --- duplex length search (M2) -------------------------------------------
    length_search: bool = False
    length_search_steps: Tuple[int, ...] = (1, -1, 2, -2)   # bps per helix end
    max_routing_attempts: int = 5      # routing calls per sample including the first
    total_bp_budget: Optional[int] = None  # hard cap on total bps after repair
    # --- router-guided junction repair (M3) ----------------------------------
    # The router pairs helix ends by nearest neighbour (assign_SSedges_wsubstr) and places
    # staple nicks away from helix ends and crossovers.  These steps edit helix-end
    # lengths so that the unchanged router succeeds; they need no target graph.
    pairing_repair: bool = False       # make the single-strand pairing at junctions one-to-one
    pairing_max_trim: int = 6          # bp removed per helix end at most
    pairing_max_ext: int = 2           # bp added per helix end at most
    pairing_cluster_link: float = 5.0  # nm, helix ends closer than this belong to one junction
    pairing_context: float = 12.0      # nm, neighbouring ends included when simulating a junction
    pairing_restarts: int = 12         # random restarts of the local search per junction
    pairing_rounds: int = 4            # re-run with the router's real stages 1-3 as acceptance test
    pairing_short_helix: int = 30      # bp; a helix this short with both ends in one junction
                                       # (spurious fragment at a vertex) may be removed
    guard_m1_pairing: bool = False     # undo M1 edits that add single-strand pairing failures
    staple_search: bool = False        # per-helix-end search on staples longer than 50 bp
    staple_steps: Tuple[int, ...] = (1, -1, 2, -2, 3, -3)   # >0 trim, <0 extend (bp per end)
    # --- round 3 (M4): router stages that M3 cannot reach ---------------------
    crossover_repair: bool = False     # fix gen_align_CrossOvers exceptions caused by one-way
                                       # helix pairs (radius_graph keeps <= 32 neighbours)
    crossover_max_trim: int = 4        # bp trimmed per helix end at most for that fix
    crossover_rounds: int = 4
    closed_staple_search: bool = False # staple search also targets staples whose two ends are
                                       # the two ends of one single-strand edge
                                       # (assign_nucleidx_to_route raises on them)
    pairing_short_bridge: bool = False # a short helix may also be removed when only one of its
                                       # ends lies in the junction (it bridges two junctions)
    pairing_escalate: bool = False     # if pairing failures remain, re-plan with wider ranges
    pairing_escalate_trim: int = 10
    pairing_escalate_ext: int = 3
    pairing_escalate_restarts: int = 40
    pairing_max_edit_fraction: float = 1.0  # reject a pairing round whose cumulative edits
                                            # (deleted + inserted points) exceed this fraction
    pairing_exhaustive: bool = False   # enumerate all end positions of a still-unsolved junction
    pairing_exhaustive_ends: int = 4   # ... when it has at most this many free ends
    pairing_alternatives: int = 0      # if a round is rejected by the router's real stages
                                       # (replica and router disagree once crossovers move),
                                       # verify up to this many other replica solutions per junction
    pairing_alt_checks: int = 24       # real stage 1-3 checks per round for those alternatives
    staple_helix_targets: bool = False # staple search also moves both ends of helices holding
                                       # conflicting staple crossovers (mid-helix defects)
    staple_defect_guided: bool = False # while the router raises, accept moves that reduce the
                                       # staple-graph defects (each check counted as a call)
    loop_split: bool = False           # single-loop design (scaffold_routing raises on it):
                                       # re-pair two single strands so that two loops form
    loop_split_candidates: int = 8     # pair swaps tried, shortest new single strands first
    loop_split_checks: int = 6         # end-length combinations verified per swap
    # --- round 5 (M6): staple segmentation ------------------------------------
    staple_resplit: bool = False       # if the router leaves a staple > 50 bp, re-split that
                                       # staple-graph component by dynamic programming at the
                                       # router's own nick positions (geometry unchanged)
    staple_resplit_min: int = 20       # shortest staple segment the re-split may create
    staple_xover_cleanup: bool = False # remove the fewest staple double crossovers that put two
                                       # crossovers on one base pair (staple graph degree 3/4,
                                       # staples_routing raises); geometry unchanged
    drop_isolated_fragments: bool = False  # delete short components the scaffold cannot reach
    fragment_max_bp: int = 30
    # --- round 7 (M8): single-strand matching --------------------------------
    ss_rematch: bool = False           # where the router's nearest-neighbour single-strand
                                       # pairing leaves a helix end with 0 or >= 2 partners,
                                       # re-pair the ends of that junction by a minimum-length
                                       # perfect matching (geometry unchanged)
    ss_rematch_link: float = 5.0       # nm, helix ends closer than this form one junction
    ss_rematch_max_nm: float = 6.0     # nm, longest new single strand
    ss_rematch_max_ends: int = 16      # larger junctions are left unchanged

    def to_dict(self):
        return asdict(self)

    @classmethod
    def recommended(cls, **overrides) -> "TGAVRConfig":
        """The frozen routing configuration: method M8 of benchmarks/tgavr_benchmark.py
        with a budget of 15 routing calls (see docs/TGAVR_EXPERIMENT.md).  Keyword
        arguments override single fields."""
        return cls(**{**RECOMMENDED_OVERRIDES, **overrides})


# Frozen configuration (benchmark method M8, budget 15).  Every other field keeps its
# TGAVRConfig default.  recover_pointcloud_topology only affects .pt targets.
RECOMMENDED_OVERRIDES = dict(
    max_routing_attempts=15, recover_pointcloud_topology=True,
    # M2: duplex length search
    length_search=True,
    # M3: router-guided junction repair
    pairing_repair=True, staple_search=True, guard_m1_pairing=True,
    # M4
    crossover_repair=True, closed_staple_search=True, pairing_short_bridge=True,
    pairing_escalate=True, loop_split=True, pairing_max_edit_fraction=0.03,
    # M5
    pairing_exhaustive=True, pairing_alternatives=12, staple_helix_targets=True,
    staple_defect_guided=True,
    # M6
    staple_resplit=True,
    # M7
    staple_xover_cleanup=True, drop_isolated_fragments=True,
    # M8
    ss_rematch=True,
)


# =============================================================================
# TARGET GRAPH
# =============================================================================

@dataclass
class TargetGraph:
    """Wireframe target: vertices [V,3] and undirected edges [E,2] (no duplicates)."""
    vertices: np.ndarray
    edges: np.ndarray
    source: str = ""
    repairable: Optional[np.ndarray] = None   # [V] bool; None -> every vertex may be repaired
    recovered: bool = False                   # True: topology reconstructed from a point cloud
    diagnostics: dict = field(default_factory=dict)
    edge_group: Optional[np.ndarray] = None   # [E] continuous-path id per edge (None -> own id)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        e = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        grp = (np.arange(len(e)) if self.edge_group is None
               else np.asarray(self.edge_group, dtype=np.int64).reshape(-1))
        if grp.shape[0] != e.shape[0]:
            raise ValueError("edge_group must have one entry per edge")
        e = np.sort(e, axis=1)
        keep = e[:, 0] != e[:, 1]
        e, grp = e[keep], grp[keep]
        self.edges, idx = np.unique(e, axis=0, return_index=True)
        self.edge_group = (np.arange(self.edges.shape[0]) if self.edge_group is None else grp[idx])
        if self.vertices.shape[0] < 2 or self.edges.shape[0] < 1:
            raise ValueError("TargetGraph needs at least two vertices and one edge")
        if self.edges.max() >= self.vertices.shape[0]:
            raise ValueError("TargetGraph edge refers to a missing vertex")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("TargetGraph vertices must be finite")
        if self.repairable is None:
            self.repairable = np.ones(self.vertices.shape[0], dtype=bool)
        self.repairable = np.asarray(self.repairable, dtype=bool).reshape(-1)
        if self.repairable.shape[0] != self.vertices.shape[0]:
            raise ValueError("repairable mask must have one entry per vertex")

    @property
    def degree(self) -> np.ndarray:
        return np.bincount(self.edges.flatten(), minlength=self.vertices.shape[0])

    def edge_lengths(self, verts: Optional[np.ndarray] = None) -> np.ndarray:
        v = self.vertices if verts is None else verts
        return np.linalg.norm(v[self.edges[:, 1]] - v[self.edges[:, 0]], axis=1)

    def transformed(self, s: float, R: np.ndarray, t: np.ndarray) -> "TargetGraph":
        return TargetGraph(s * self.vertices @ R.T + t, self.edges.copy(), self.source,
                           self.repairable.copy(), self.recovered, dict(self.diagnostics),
                           self.edge_group.copy())


def load_target_graph(path: str, recover_pointcloud: bool = False,
                      cfg: Optional["TGAVRConfig"] = None) -> Optional[TargetGraph]:
    """
    Load target topology.

    * .ply / .stl: verified topology (edges from mesh faces, using the same edge
      extraction as shape/mesh_to_shape.py).
    * .pt: a plain centre-line point cloud without connectivity.  Returns None unless
      ``recover_pointcloud=True``, in which case the topology is *reconstructed*
      (``recover_pointcloud_graph``) and flagged ``recovered=True`` with diagnostics.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".ply", ".stl"):
        import meshio
        from shape.mesh_to_shape import _extract_edges_from_mesh
        msh = meshio.read(str(p))
        edges = _extract_edges_from_mesh(msh).T
        return TargetGraph(np.asarray(msh.points)[:, :3], edges, source=str(p))
    if ext == ".pt" and recover_pointcloud:
        P = torch.load(str(p), map_location="cpu")
        if not isinstance(P, torch.Tensor) or P.dim() != 2 or P.shape[1] != 3:
            return None
        g = recover_pointcloud_graph(P.double().numpy(), cfg or TGAVRConfig())
        if g is not None:
            g.source = str(p)
        return g
    return None


# -----------------------------------------------------------------------------
# Topology recovery for centre-line point-cloud targets (target_shapes/pt)
# -----------------------------------------------------------------------------
#
# How .pt targets are made (shape/shape_ults.py, shape/svg_to_shape.py):
#   * every path is sampled at a regular spacing of 1.0 nm (cut(line, 1.0) /
#     interpolate_edge(Le=1.0));
#   * remove_intersection() deletes the samples around junctions of > 2 paths
#     (radius_shape, default 2 nm) and samples closer than 0.5 nm to another path.
# Consequently a .pt target is a set of 1-nm-sampled polylines that are *cut open*
# at junctions.  Recovery inverts exactly this construction:
#   1. link mutual 2-nearest neighbours closer than link_factor x spacing  -> polylines
#   2. simplify each polyline (Douglas-Peucker, tolerance dp_tol)          -> straight edges
#   3. path end points facing each other within junction_radius are merged into a
#      junction vertex; an end point facing the *interior* of another path within
#      junction_radius is attached to it (T-junction)
# Junction and free-end vertices are repairable; Douglas-Peucker bend vertices are
# not (a duplex may legitimately run around a bend of one continuous path).


def _douglas_peucker(P: np.ndarray, tol: float) -> np.ndarray:
    keep = np.zeros(len(P), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = P[b] - P[a]
        L2 = float(seg @ seg)
        Q = P[a + 1:b] - P[a]
        if L2 < 1e-12:
            d = np.linalg.norm(Q, axis=1)
        else:
            u = np.clip(Q @ seg / L2, 0, 1)
            d = np.linalg.norm(Q - u[:, None] * seg[None], axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def recover_pointcloud_graph(P: np.ndarray, cfg: "TGAVRConfig") -> Optional[TargetGraph]:
    n = P.shape[0]
    if n < 4:
        return None
    tree = cKDTree(P)
    dd, ii = tree.query(P, k=3)
    h = float(np.median(dd[:, 1]))
    r = cfg.recover_link_factor * h
    # 1. mutual 2-NN links within r
    links = set()
    for i in range(n):
        for k in (1, 2):
            j = int(ii[i, k])
            if dd[i, k] < r and i in ii[j, 1:3]:
                links.add((min(i, j), max(i, j)))
    pairs = np.array(sorted(links), dtype=np.int64).reshape(-1, 2)
    deg = np.bincount(pairs.flatten(), minlength=n) if len(pairs) else np.zeros(n, int)
    lab, _ = _components(n, pairs)
    paths = []          # (ordered indices, is_cycle)
    n_branched = 0
    for c in np.unique(lab):
        nodes = np.nonzero(lab == c)[0]
        if nodes.size < cfg.recover_min_path_points:
            continue
        order, simple, cycle = _order_path(nodes, pairs)
        if not (simple or cycle):
            n_branched += 1
        paths.append((order, cycle))
    if not paths:
        return None
    # 2. end points and junction clustering
    ends = []           # (path id, which end 0/-1, point index)
    for pid, (order, cycle) in enumerate(paths):
        if not cycle:
            ends += [(pid, 0, int(order[0])), (pid, -1, int(order[-1]))]
    R = cfg.recover_junction_radius
    E_pts = np.array([P[e[2]] for e in ends]).reshape(-1, 3)
    parent = list(range(len(ends)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    if len(ends) > 1:
        for a, b in cKDTree(E_pts).query_pairs(R):
            if ends[a][0] == ends[b][0] and len(paths[ends[a][0]][0]) * h < 2 * R:
                continue            # both ends of one tiny path: not a junction
            parent[find(a)] = find(b)
    clusters: Dict[int, List[int]] = {}
    for k in range(len(ends)):
        clusters.setdefault(find(k), []).append(k)
    # T-junctions: single end facing the interior of another path
    path_of_point = np.full(n, -1)
    for pid, (order, _) in enumerate(paths):
        path_of_point[order] = pid
    attach = {}         # end id -> interior point index of another path
    for root, members in clusters.items():
        if len(members) != 1:
            continue
        k = members[0]
        pid, _, pidx = ends[k]
        cand = [j for j in tree.query_ball_point(P[pidx], R)
                if path_of_point[j] >= 0 and path_of_point[j] != pid]
        if cand:
            j = min(cand, key=lambda j: (np.linalg.norm(P[j] - P[pidx]), j))
            attach[k] = j
    # contact junctions: two different paths touching (closer than contact_radius) away
    # from their own end points, e.g. a circle tangent to another one.  Both closest
    # points become repairable vertices; no edge is added between the paths.
    contact_pts = set()
    end_pts = np.array([P[e[2]] for e in ends]).reshape(-1, 3)
    end_path = np.array([e[0] for e in ends])
    def near_own_end(i):
        if not len(end_pts):
            return False
        m = end_path == path_of_point[i]
        return bool(m.any() and np.linalg.norm(end_pts[m] - P[i], axis=1).min() < R)
    cpairs = [(a, b) for a, b in sorted(tree.query_pairs(cfg.recover_contact_radius))
              if path_of_point[a] >= 0 and path_of_point[b] >= 0
              and path_of_point[a] != path_of_point[b]
              and not near_own_end(a) and not near_own_end(b)]
    if cpairs:
        cp = np.array(cpairs)
        nodes = np.unique(cp.flatten())
        loc = {int(v): k for k, v in enumerate(nodes)}
        lab_c, _ = _components(len(nodes), np.array([[loc[a], loc[b]] for a, b in cpairs]))
        for c in np.unique(lab_c):
            members = nodes[lab_c == c]
            for pid in np.unique(path_of_point[members]):
                mine = members[path_of_point[members] == pid]
                other = members[path_of_point[members] != pid]
                d = np.linalg.norm(P[mine][:, None] - P[other][None], axis=-1).min(1)
                contact_pts.add(int(mine[np.argmin(d)]))
    # 3. vertices: DP-kept points (+ forced T attachment / contact points), junction centroids
    forced = {}
    for k, j in attach.items():
        forced.setdefault(int(path_of_point[j]), set()).add(j)
    for j in contact_pts:
        forced.setdefault(int(path_of_point[j]), set()).add(j)
    V, rep, vid_of_point = [], [], {}
    end_vertex = {}
    for root, members in sorted(clusters.items()):
        if len(members) >= 2:
            vid = len(V)
            V.append(np.mean([E_pts[m] for m in members], axis=0)); rep.append(True)
            for m in members:
                end_vertex[m] = vid
    # T-junction: the attachment point becomes the junction vertex and the attached
    # path end is merged into it (one degree-3 vertex instead of two nearby ones)
    for k, j in sorted(attach.items()):
        if j not in vid_of_point:
            vid_of_point[j] = len(V)
            V.append(P[j]); rep.append(True)
        end_vertex[k] = vid_of_point[j]
    edges, groups = [], []
    for pid, (order, cycle) in enumerate(paths):
        pts = P[order]
        pos = {int(o): q for q, o in enumerate(order)}
        if cycle:
            if forced.get(pid):          # start the loop at an attachment point
                q0 = min(pos[j] for j in forced[pid])
                order = np.roll(order, -q0); pts = P[order]
                pos = {int(o): q for q, o in enumerate(order)}
            far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
            keep = (set(_douglas_peucker(pts[:far + 1], cfg.recover_dp_tol).tolist()) |
                    set((far + _douglas_peucker(np.vstack((pts[far:], pts[:1])),
                                                cfg.recover_dp_tol)).tolist())) - {len(pts)}
        else:
            keep = set(_douglas_peucker(pts, cfg.recover_dp_tol).tolist())
        for j in forced.get(pid, ()):
            keep.add(pos[j])
        keep = sorted(keep)
        ids = []
        for q in keep:
            pi = int(order[q])
            k_end = None
            if not cycle and q in (0, len(order) - 1):
                k_end = next(k for k, e in enumerate(ends) if e[0] == pid and e[2] == pi)
            if k_end is not None and k_end in end_vertex:
                ids.append(end_vertex[k_end])
                continue
            if pi not in vid_of_point:
                vid_of_point[pi] = len(V)
                V.append(P[pi])
                is_free_end = k_end is not None and k_end not in attach
                rep.append(bool(is_free_end or pi in attach.values() or pi in contact_pts))
            ids.append(vid_of_point[pi])
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                edges.append((a, b)); groups.append(pid)
        if cycle and len(ids) > 2:
            edges.append((ids[-1], ids[0])); groups.append(pid)
    if not edges:
        return None
    V = np.array(V)
    used = np.unique(np.array(edges).flatten())
    remap = -np.ones(len(V), int); remap[used] = np.arange(used.size)
    g = TargetGraph(V[used], remap[np.array(edges)], repairable=np.array(rep)[used], recovered=True,
                    edge_group=np.array(groups))
    g.diagnostics = dict(spacing_nm=round(h, 3), link_radius_nm=round(r, 3), n_points=int(n),
                         n_paths=len(paths), n_cycles=int(sum(c for _, c in paths)),
                         n_branched_paths=n_branched,
                         frac_points_deg_gt2=float((deg > 2).mean()),
                         frac_points_unlinked=float((deg == 0).mean()),
                         n_junctions=int(sum(len(m) >= 2 for m in clusters.values())),
                         n_t_junctions=len(attach),
                         n_contact_vertices=len(contact_pts),
                         n_free_ends=int(sum(1 for root, m in clusters.items()
                                             if len(m) == 1 and m[0] not in attach)),
                         n_vertices=int(g.vertices.shape[0]), n_edges=int(g.edges.shape[0]),
                         n_repairable=int(g.repairable.sum()))
    return g


def resolve_target_path(name: str) -> Optional[str]:
    """Find a target file by name in the repository's target_shapes folders."""
    p = Path(name)
    if p.exists():
        return str(p)
    for d in ("target_shapes/ply/2D", "target_shapes/ply/3D", "target_shapes/ply",
              "target_shapes/stl", "target_shapes/svg", "target_shapes/pt"):
        q = Path(d) / p.name
        if q.exists():
            return str(q)
    return None


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def sample_graph(verts: np.ndarray, edges: np.ndarray, spacing: float, return_u: bool = False):
    """Dense samples along edges. Returns points [M,3], edge id [M], weight [M] (, u [M])."""
    pts, eid, w, us = [], [], [], []
    for k, (a, b) in enumerate(edges):
        L = np.linalg.norm(verts[b] - verts[a])
        n = max(2, int(math.ceil(L / max(spacing, 1e-9))) + 1)
        u = np.linspace(0.0, 1.0, n)
        pts.append(verts[a][None] + u[:, None] * (verts[b] - verts[a])[None])
        eid.append(np.full(n, k))
        w.append(np.full(n, L / n))
        us.append(u)
    out = (np.vstack(pts), np.concatenate(eid), np.concatenate(w))
    return out + (np.concatenate(us),) if return_u else out


def point_segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """
    Distances of points P [N,3] to segments (A[k], B[k]) [K,3].
    Returns dist [N,K] and the clamped segment parameter u [N,K] in [0,1].
    """
    AB = B - A                                   # [K,3]
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)     # [K]
    AP = P[:, None, :] - A[None, :, :]           # [N,K,3]
    u = np.clip((AP * AB[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = A[None] + u[..., None] * AB[None]
    d = np.linalg.norm(P[:, None, :] - proj, axis=-1)
    return d, u


def _umeyama(src: np.ndarray, dst: np.ndarray, w: np.ndarray):
    """Weighted similarity transform minimising sum w |s R src + t - dst|^2, det R = +1."""
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    S = src - mu_s
    D = dst - mu_d
    cov = (w[:, None] * D).T @ S
    U, sig, Vt = np.linalg.svd(cov)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1.0                       # forbid reflections
    R = U @ E @ Vt
    var_s = (w * (S ** 2).sum(1)).sum()
    s = float(np.trace(np.diag(sig) @ E) / max(var_s, 1e-12))
    t = mu_d - s * R @ mu_s
    return s, R, t


def _rotation_starts() -> List[np.ndarray]:
    """
    Deterministic set of proper rotations used as ICP starts: the octahedral (24)
    and icosahedral (60) rotation groups (84 starts, identity first).  Two groups
    with different axes give a quasi-uniform cover of SO(3) (max gap < ~45 deg).
    """
    from scipy.spatial.transform import Rotation
    mats = list(Rotation.create_group('O').as_matrix()) + list(Rotation.create_group('I').as_matrix())
    mats = [np.asarray(M, dtype=np.float64) for M in mats]
    mats.sort(key=lambda M: (-round(np.trace(M), 6), tuple(np.round(M.flatten(), 6))))
    return mats


# =============================================================================
# PHASE 1 - SIMILARITY ALIGNMENT  (target -> generated)
# =============================================================================

@dataclass
class AlignmentResult:
    scale: float
    R: np.ndarray
    t: np.ndarray
    rmse: float               # generated points -> aligned target curve (trimmed) [nm]
    rmse_full: float          # untrimmed
    coverage: float           # fraction of target curve within cover_tol of X
    n_iter: int
    converged: bool
    reliable: bool
    start_index: int
    ambiguity: float          # best/second-best objective ratio over distinct starts
    message: str = ""

    def as_dict(self):
        return dict(scale=self.scale, R=self.R.tolist(), t=self.t.tolist(), rmse=self.rmse,
                    rmse_full=self.rmse_full, coverage=self.coverage, n_iter=self.n_iter,
                    converged=self.converged, reliable=self.reliable,
                    start_index=self.start_index, ambiguity=self.ambiguity,
                    message=self.message)


def _align_from_start(X, tree_X, Yv, edges, R0, cfg: TGAVRConfig, max_iter=None, init=None):
    """Trimmed symmetric point-to-curve similarity ICP from an initial rotation
    (or from a full initial pose ``init=(s, R, t)``)."""
    max_iter = cfg.align_max_iter if max_iter is None else max_iter
    cX = X.mean(0)
    # robust extents (RMS radius) for initial scale
    Ys, _, Yw, Yu = sample_graph(Yv, edges, spacing=_raw_spacing(Yv, edges, cfg), return_u=True)
    cY = (Yw[:, None] * Ys).sum(0) / Yw.sum()
    # The curve -> X term only uses edge interiors: generated helices stop short of
    # the vertices (single-strand clearance), so vertex samples would bias s down.
    inner = (Yu >= cfg.align_interior_u) & (Yu <= 1.0 - cfg.align_interior_u)
    Ys, Yw = Ys[inner], Yw[inner]
    rX = np.sqrt(((X - cX) ** 2).sum(1).mean())
    rY = np.sqrt((Yw * ((Ys - cY) ** 2).sum(1)).sum() / Yw.sum())
    s = rX / max(rY, 1e-12)
    R = R0.copy()
    t = cX - s * R @ cY
    if init is not None:
        s, R, t = init[0], init[1].copy(), init[2].copy()

    prev = np.inf
    converged = False
    A_raw = Yv[edges[:, 0]]
    B_raw = Yv[edges[:, 1]]
    obj = np.inf
    it = -1
    for it in range(max_iter):
        A = s * A_raw @ R.T + t
        B = s * B_raw @ R.T + t
        # X -> curve correspondences (closest point on closest segment)
        d, u = _closest_on_segments(X, A, B)
        k = d.argmin(1)
        uk = u[np.arange(len(X)), k]
        dk = d[np.arange(len(X)), k]
        # correspondence expressed in raw target coordinates
        Yc_raw = A_raw[k] + uk[:, None] * (B_raw[k] - A_raw[k])
        # curve -> X correspondences: local centroid of X around each curve sample.
        # (The nearest point would be biased towards one helix of a 2HB bundle;
        #  the centroid of both helices lies on the bundle centre line.)
        Ys_al = s * Ys @ R.T + t
        Xc, dY = _local_centroids(X, tree_X, Ys_al, cfg.align_centroid_radius)
        # trimming
        thr_x = np.quantile(dk, cfg.align_trim)
        thr_y = np.quantile(dY, cfg.align_trim)
        mx = dk <= thr_x
        my = dY <= thr_y
        src = np.vstack((Yc_raw[mx], Ys[my]))
        dst = np.vstack((X[mx], Xc[my]))
        wx = np.full(mx.sum(), 1.0 / max(mx.sum(), 1))
        wy = Yw[my] / max(Yw[my].sum(), 1e-12)
        w = np.concatenate((wx, wy))
        obj = (wx * dk[mx] ** 2).sum() + (wy * dY[my] ** 2).sum()
        s, R, t = _umeyama(src, dst, w)
        if not np.isfinite(s) or s <= 0:
            return None
        if np.isfinite(prev) and abs(prev - obj) <= cfg.align_tol * max(prev, 1e-12):
            converged = True
            break
        prev = obj
    return s, R, t, obj, it + 1, converged


def _local_centroids(X, tree_X, Q, radius):
    """Centroid of X within `radius` of each query (nearest point if none)."""
    dn, inn = tree_X.query(Q)
    C = X[inn].copy()
    for i, nb in enumerate(tree_X.query_ball_point(Q, radius)):
        if len(nb) >= 3:
            C[i] = X[nb].mean(0)
    return C, np.linalg.norm(C - Q, axis=1)


def _raw_spacing(Yv, edges, cfg):
    # spacing in raw target units that maps to ~align_target_spacing after scaling;
    # resolved per call from the edge-length distribution (scale-free).
    L = np.linalg.norm(Yv[edges[:, 1]] - Yv[edges[:, 0]], axis=1)
    return max(L.sum() / 400.0, 1e-9)


def _closest_on_segments(P, A, B, chunk=4096):
    out_d = np.empty((P.shape[0], A.shape[0]))
    out_u = np.empty_like(out_d)
    for i in range(0, P.shape[0], chunk):
        d, u = point_segment_distance(P[i:i + chunk], A, B)
        out_d[i:i + chunk] = d
        out_u[i:i + chunk] = u
    return out_d, out_u


def align_target_to_points(target: TargetGraph, X, cfg: Optional[TGAVRConfig] = None,
                           starts: Optional[List[np.ndarray]] = None) -> AlignmentResult:
    """
    Estimate s>0, R in SO(3), t such that  s R Y + t  matches the generated base-pair
    point cloud X (the generated points are never modified).

    Objective: trimmed symmetric point-to-curve distance between X and the target
    edges, minimised by alternating closest-point correspondences and a weighted
    Umeyama update (reflections forbidden).  Deterministic multi-start over the
    24 cube rotations; the best final objective wins.
    """
    cfg = cfg or TGAVRConfig()
    X = _to_numpy(X)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 4:
        raise ValueError("generated points must be an [N>=4, 3] array")
    if not np.all(np.isfinite(X)):
        raise ValueError("generated points contain non-finite values")
    tree_X = cKDTree(X)
    starts = starts if starts is not None else _rotation_starts()
    # stage 1: every start, few iterations, deterministic subsample of X
    stride = max(1, X.shape[0] // cfg.align_coarse_points)
    Xs = X[::stride]
    tree_s = cKDTree(Xs)
    coarse = []
    for si, R0 in enumerate(starts):
        r = _align_from_start(Xs, tree_s, target.vertices, target.edges, R0, cfg,
                              max_iter=cfg.align_coarse_iter)
        if r is not None:
            coarse.append((round(r[3], 10), si, r))
    coarse.sort(key=lambda z: (z[0], z[1]))
    # stage 2: refine the best distinct poses on all points
    results = []
    for _, si, r in coarse[:cfg.align_refine_top]:
        rr = _align_from_start(X, tree_X, target.vertices, target.edges, starts[si], cfg,
                               init=(r[0], r[1], r[2]))
        if rr is not None:
            results.append((rr[3], si, rr))
    if not results:
        return AlignmentResult(float("nan"), np.eye(3), np.zeros(3), float("inf"), float("inf"),
                               0.0, 0, False, False, -1, 1.0, "all starts degenerate")
    results.sort(key=lambda z: (round(z[0], 10), z[1]))
    best_obj, si, (s, R, t, obj, n_it, conv) = results[0]
    # ambiguity: compare with the best start that converged to a *different* pose
    amb = 1.0
    for o, sj, (s2, R2, t2, *_r) in results[1:]:
        ang = np.degrees(np.arccos(np.clip((np.trace(R2.T @ R) - 1) / 2, -1, 1)))
        if ang > 5.0 or abs(s2 / s - 1) > 0.02:
            amb = best_obj / max(o, 1e-12)
            break
    aligned = target.transformed(s, R, t)
    rmse, rmse_full, cov = alignment_quality(aligned, X, cfg, tree_X)
    msg = []
    reliable = True
    if not (np.isfinite(s) and s > 0):
        reliable = False; msg.append("non-positive scale")
    if rmse > cfg.align_max_rmse:
        reliable = False; msg.append(f"rmse {rmse:.2f} nm > {cfg.align_max_rmse}")
    if cov < cfg.align_min_coverage:
        reliable = False; msg.append(f"coverage {cov:.2f} < {cfg.align_min_coverage}")
    if np.linalg.det(R) < 0:
        reliable = False; msg.append("reflection")
    return AlignmentResult(float(s), R, t, rmse, rmse_full, cov, n_it, conv, reliable, si,
                           float(amb), "; ".join(msg) or "ok")


def alignment_quality(aligned: TargetGraph, X: np.ndarray, cfg: TGAVRConfig, tree_X=None):
    A = aligned.vertices[aligned.edges[:, 0]]
    B = aligned.vertices[aligned.edges[:, 1]]
    d, _ = _closest_on_segments(X, A, B)
    dk = d.min(1)
    thr = np.quantile(dk, cfg.align_trim)
    rmse = float(np.sqrt((dk[dk <= thr] ** 2).mean()))
    rmse_full = float(np.sqrt((dk ** 2).mean()))
    Ys, _, Yw = sample_graph(aligned.vertices, aligned.edges, cfg.align_target_spacing)
    tree_X = tree_X if tree_X is not None else cKDTree(X)
    dY, _ = tree_X.query(Ys)
    cov = float((Yw * (dY <= cfg.align_cover_tol)).sum() / Yw.sum())
    return rmse, rmse_full, cov


def _to_numpy(X) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().double().numpy()
    return np.asarray(X, dtype=np.float64)


# =============================================================================
# ROUTER-CONSISTENT CHAIN EXTRACTION (non-destructive, original indices kept)
# =============================================================================

@dataclass
class ChainAnalysis:
    n_points: int
    pairs: np.ndarray            # [P,2] links (distance < link_radius)
    degree: np.ndarray           # [N]
    overdense: np.ndarray        # [N] bool, points pcl_filter would delete
    chains: List[np.ndarray]     # ordered original indices of kept chains (> 10 pts)
    fragments: List[np.ndarray]  # components the router would drop (<= 10 pts)
    branched: List[int]          # chain ids that are not simple paths
    cyclic: List[int]            # chain ids that are closed loops
    point_chain: np.ndarray      # [N] chain id or -1


def _components(n, pairs, mask=None):
    if mask is None:
        mask = np.ones(n, dtype=bool)
    keep = mask[pairs[:, 0]] & mask[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    p = pairs[keep]
    A = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n)) if len(p) else coo_matrix((n, n))
    _, lab = connected_components(A, directed=False)
    lab = lab.copy()
    lab[~mask] = -1
    return lab, p


def _order_path(nodes: np.ndarray, pairs: np.ndarray):
    """Order the nodes of a component along its path. Returns (order, is_simple, is_cycle)."""
    s = set(nodes.tolist())
    adj = {i: [] for i in nodes.tolist()}
    for a, b in pairs:
        if a in s and b in s:
            adj[a].append(b); adj[b].append(a)
    deg = {i: len(v) for i, v in adj.items()}
    simple = all(d <= 2 for d in deg.values()) and sum(deg.values()) // 2 == len(nodes) - 1
    cycle = all(d == 2 for d in deg.values()) and sum(deg.values()) // 2 == len(nodes)
    ends = sorted(i for i, d in deg.items() if d == 1)
    start = ends[0] if ends else min(nodes.tolist())
    order, prev, cur = [start], None, start
    seen = {start}
    while True:
        nxt = [j for j in sorted(adj[cur]) if j != prev and j not in seen]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        order.append(cur); seen.add(cur)
    if len(order) < len(nodes):            # branched: append the rest deterministically
        order += sorted(s - seen)
    return np.asarray(order, dtype=np.int64), simple, cycle


def analyse_chains(X: np.ndarray, cfg: TGAVRConfig) -> ChainAnalysis:
    """Reproduce the router's stage-1 rules (pcl_filter + segmentation) on X."""
    n = X.shape[0]
    pairs = np.array(sorted(cKDTree(X).query_pairs(cfg.link_radius - 1e-9)), dtype=np.int64).reshape(-1, 2)
    deg = np.bincount(pairs.flatten(), minlength=n) if len(pairs) else np.zeros(n, int)
    over = np.zeros(n, dtype=bool)
    if len(pairs):
        bad = deg[pairs[:, 0]] + deg[pairs[:, 1]] > 4
        over[pairs[bad].flatten()] = True
    lab, kept_pairs = _components(n, pairs, ~over)
    chains, frags, branched, cyclic = [], [], [], []
    point_chain = np.full(n, -1)
    for c in np.unique(lab[lab >= 0]):
        nodes = np.nonzero(lab == c)[0]
        if nodes.size < cfg.min_chain_points:
            frags.append(nodes)
            continue
        order, simple, cycle = _order_path(nodes, kept_pairs)
        cid = len(chains)
        chains.append(order)
        point_chain[order] = cid
        if cycle:
            cyclic.append(cid)
        elif not simple:
            branched.append(cid)
    return ChainAnalysis(n, pairs, deg, over, chains, frags, branched, cyclic, point_chain)


# =============================================================================
# PHASE 2 - TARGET-GUIDED EDGE ASSIGNMENT
# =============================================================================

@dataclass
class EdgeAssignment:
    vertices: np.ndarray          # refined vertex positions (aligned frame)
    vertex_shift: np.ndarray      # |refined - aligned| per vertex
    vertex_radius: np.ndarray     # r_v
    point_edge: np.ndarray        # [N] nearest edge (or -1 if farther than tol)
    point_dist: np.ndarray        # [N] distance to that edge
    point_u: np.ndarray           # [N] parameter along it
    point_zone: np.ndarray        # [N] vertex id whose zone contains the point, -1 otherwise
    ambiguous: np.ndarray         # [N] bool
    chain_edge: np.ndarray        # [C] edge id per chain (-1 unassigned)
    chain_conf: np.ndarray        # [C] vote fraction
    chain_second: np.ndarray      # [C] second edge with substantial votes (-1 none)
    edge_chains: List[List[int]]  # chains per edge
    edge_bps: np.ndarray          # [E] points assigned per edge (all chains)
    edge_length: np.ndarray       # [E] refined vertex-to-vertex length [nm]
    edge_group: Optional[np.ndarray] = None   # [E] continuous-path id per edge


def _fit_line(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    return c, Vt[0]


def _closest_point_to_lines(cs, ds):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, d in zip(cs, ds):
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ c
    if np.linalg.cond(A) > 1e8:
        return None
    return np.linalg.solve(A, b)


def assign_edges(X: np.ndarray, graph: TargetGraph, chains: ChainAnalysis,
                 cfg: TGAVRConfig) -> EdgeAssignment:
    V0 = graph.vertices
    E = graph.edges
    nE, nV = E.shape[0], V0.shape[0]

    def nearest(Vv):
        d, u = _closest_on_segments(X, Vv[E[:, 0]], Vv[E[:, 1]])
        o = np.argsort(d, axis=1)
        k1 = o[:, 0]
        d1 = d[np.arange(len(X)), k1]
        d2 = d[np.arange(len(X)), o[:, 1]] if nE > 1 else np.full(len(X), np.inf)
        return k1, d1, d2, u[np.arange(len(X)), k1]

    # --- 1. coarse assignment with aligned vertices; refine vertices from edge lines
    k1, d1, d2, u1 = nearest(V0)
    Lk = graph.edge_lengths(V0)
    interior = (u1 > 0.25) & (u1 < 0.75) & (d1 < cfg.edge_assign_tol) & (d2 - d1 > 1.0)
    lines = {}
    for k in range(nE):
        P = X[interior & (k1 == k)]
        if P.shape[0] >= 10:
            c, d = _fit_line(P)
            lines[k] = (c, d)
    V = V0.copy()
    for v in range(nV):
        inc = [k for k in range(nE) if v in E[k] and k in lines]
        if len(inc) >= 2:
            p = _closest_point_to_lines([lines[k][0] for k in inc], [lines[k][1] for k in inc])
            if p is not None and np.linalg.norm(p - V0[v]) < 0.25 * Lk[[k for k in range(nE) if v in E[k]]].min():
                V[v] = p
    shift = np.linalg.norm(V - V0, axis=1)

    # --- 2. final point assignment with refined vertices
    k1, d1, d2, u1 = nearest(V)
    L = graph.edge_lengths(V)
    inc_min = np.array([L[[k for k in range(nE) if v in E[k]]].min() for v in range(nV)])
    rv = np.clip(cfg.vertex_radius_factor * inc_min, cfg.vertex_radius_min, cfg.vertex_radius_max)
    # vertex zones only around repairable vertices (junctions / free ends / mesh vertices)
    rep_mask = graph.repairable
    dv = np.full((X.shape[0], nV), np.inf)
    if rep_mask.any():
        dv[:, rep_mask] = np.linalg.norm(X[:, None, :] - V[rep_mask][None], axis=-1)
    zone = np.where((dv < rv[None]).any(1), np.argmin(dv / rv[None], axis=1), -1)
    far = d1 > cfg.edge_assign_tol
    point_edge = np.where(far, -1, k1)
    ambiguous = (zone >= 0) | far | (d2 - d1 < 1.0)

    # --- 3. chain-level votes (reassignment of ambiguous points through their chain)
    C = len(chains.chains)
    chain_edge = np.full(C, -1); chain_conf = np.zeros(C); chain_second = np.full(C, -1)
    edge_chains: List[List[int]] = [[] for _ in range(nE)]
    for c, nodes in enumerate(chains.chains):
        votes = point_edge[nodes][~ambiguous[nodes]]
        votes = votes[votes >= 0]
        if votes.size == 0:
            votes = point_edge[nodes][point_edge[nodes] >= 0]
        if votes.size == 0:
            continue
        cnt = np.bincount(votes, minlength=nE)
        o = np.argsort(-cnt, kind="stable")
        chain_edge[c] = o[0]
        chain_conf[c] = cnt[o[0]] / cnt.sum()
        if nE > 1 and cnt[o[1]] >= max(5, 0.15 * cnt.sum()):
            chain_second[c] = o[1]
        edge_chains[o[0]].append(c)
    edge_bps = np.array([sum(len(chains.chains[c]) for c in edge_chains[k]) for k in range(nE)])
    return EdgeAssignment(V, shift, rv, point_edge, d1, u1, zone, ambiguous, chain_edge,
                          chain_conf, chain_second, edge_chains, edge_bps, L, graph.edge_group)


# =============================================================================
# PHASE 3 - VERTEX DEFECT DETECTION
# =============================================================================

@dataclass
class VertexDefects:
    vertex: int
    overdense: List[int] = field(default_factory=list)       # point ids
    merged_chains: List[int] = field(default_factory=list)   # chain ids crossing the vertex
    branched_chains: List[int] = field(default_factory=list)
    gaps: List[Tuple[int, int, float]] = field(default_factory=list)  # (end a, end b, gap nm)
    fragments: List[int] = field(default_factory=list)       # fragment ids in the zone
    helix_count: Dict[int, int] = field(default_factory=dict)  # edge -> chains (if != expected)

    @property
    def score(self) -> float:
        return (2.0 * len(self.overdense) + 5.0 * len(self.merged_chains) +
                5.0 * len(self.branched_chains) + 3.0 * len(self.gaps) +
                1.0 * len(self.fragments) + 2.0 * len(self.helix_count))

    @property
    def defective(self) -> bool:
        return (len(self.overdense) + len(self.merged_chains) + len(self.branched_chains) +
                len(self.gaps)) > 0

    def as_dict(self):
        return dict(vertex=self.vertex, overdense=len(self.overdense),
                    merged_chains=len(self.merged_chains), branched=len(self.branched_chains),
                    gaps=[round(g[2], 3) for g in self.gaps], fragments=len(self.fragments),
                    helix_count={int(k): int(v) for k, v in self.helix_count.items()},
                    score=self.score)


def _pieces(chains: ChainAnalysis):
    """All ordered pieces (kept chains + dropped fragments) with ('c'|'f', id)."""
    out = [(("c", i), ch) for i, ch in enumerate(chains.chains)]
    for i, fr in enumerate(chains.fragments):
        out.append((("f", i), fr))
    return out


def _end_axis(X, piece: np.ndarray, at_start: bool, k: int):
    """Unit direction pointing *out of* the piece at one end, from its last k points."""
    if piece.size < 2:
        return None
    seg = piece[:k] if at_start else piece[::-1][:k]       # seg[0] is the end point
    P = X[seg]
    if P.shape[0] >= 3:
        _, d = _fit_line(P)
    else:
        d = P[0] - P[-1]
    out = P[0] - P[-1]
    if np.dot(d, out) < 0:
        d = -d
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def _order_fragment(X, nodes):
    if nodes.size <= 2:
        return nodes
    c, d = _fit_line(X[nodes])
    return nodes[np.argsort((X[nodes] - c) @ d, kind="stable")]


def find_gap_pairs(X, chains: ChainAnalysis, assign: EdgeAssignment, cfg: TGAVRConfig,
                   zone_vertex: Optional[int] = None):
    """
    Collinear end-to-end breaks of one duplex: (end point a, end point b, gap).
    Both ends must lie on the same target edge (chain vote or nearest edge for
    fragments), the gap must exceed the link radius but be bridgeable, and the gap
    vector must follow the local helix axis of both pieces.
    """
    pieces = _pieces(chains)
    ends = []   # (point id, outward axis, edge, piece key)
    for key, nodes in pieces:
        if key[0] == "f":
            nodes = _order_fragment(X, nodes)
            e_vote = assign.point_edge[nodes]
            e_vote = e_vote[e_vote >= 0]
            edge = int(np.bincount(e_vote).argmax()) if e_vote.size else -1
        else:
            edge = int(assign.chain_edge[key[1]])
        if edge < 0 or nodes.size == 0:
            continue
        for at_start in (True, False):
            p = int(nodes[0] if at_start else nodes[-1])
            ax = _end_axis(X, nodes, at_start, cfg.axis_fit_points) if nodes.size >= 2 else None
            ends.append((p, ax, edge, key))
            if nodes.size == 1:
                break
    max_gap = cfg.max_insert_per_end * cfg.bp_rise
    cands = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            pa, axa, ea, ka = ends[i]
            pb, axb, eb, kb = ends[j]
            if ka == kb or assign.edge_group[ea] != assign.edge_group[eb]:
                continue
            if ka[0] == "f" and kb[0] == "f":
                continue
            g = X[pb] - X[pa]
            gl = np.linalg.norm(g)
            if not (cfg.link_radius <= gl <= max_gap):
                continue
            ok = True
            for ax, sgn in ((axa, 1.0), (axb, -1.0)):
                if ax is None:
                    continue
                cosang = sgn * np.dot(ax, g) / gl
                lateral = gl * math.sqrt(max(0.0, 1 - cosang ** 2))
                if cosang < math.cos(math.radians(25)) or lateral > 0.6:
                    ok = False
            if not ok:
                continue
            if zone_vertex is not None and not (assign.point_zone[pa] == zone_vertex or
                                                assign.point_zone[pb] == zone_vertex):
                continue
            cands.append((gl, pa, pb))
    # one-to-one matching, shortest gaps first (deterministic)
    cands.sort()
    used, out = set(), []
    for gl, pa, pb in cands:
        if pa in used or pb in used:
            continue
        used.update((pa, pb))
        out.append((pa, pb, float(gl)))
    return out


def _bend_deg(V, ea, eb, v):
    """Deviation from a straight continuation (deg) of edges ea, eb at shared vertex v."""
    oa = ea[1] if ea[0] == v else ea[0]
    ob = eb[1] if eb[0] == v else eb[0]
    da = V[oa] - V[v]; db = V[ob] - V[v]
    cos = np.dot(da, db) / max(np.linalg.norm(da) * np.linalg.norm(db), 1e-12)
    return 180.0 - math.degrees(math.acos(float(np.clip(cos, -1, 1))))


def detect_vertex_defects(X, chains: ChainAnalysis, assign: EdgeAssignment, graph: TargetGraph,
                          cfg: TGAVRConfig) -> List[VertexDefects]:
    nV = graph.vertices.shape[0]
    E = graph.edges
    out = [VertexDefects(v) for v in range(nV)]
    zone = assign.point_zone
    for i in np.nonzero(chains.overdense)[0]:
        if zone[i] >= 0:
            out[zone[i]].overdense.append(int(i))
    for c, nodes in enumerate(chains.chains):
        zs = np.unique(zone[nodes][zone[nodes] >= 0])
        if c in chains.branched:
            for v in zs:
                out[v].branched_chains.append(c)
        e2 = assign.chain_second[c]
        if e2 >= 0:
            e1 = assign.chain_edge[c]
            shared = set(E[e1].tolist()) & set(E[e2].tolist())
            for v in shared:
                if v in zs and _bend_deg(assign.vertices, E[e1], E[e2], v) > cfg.merge_bend_deg:
                    # a duplex continuing straight through a vertex of two (nearly)
                    # collinear target edges is legitimate -> only bent merges count
                    out[v].merged_chains.append(c)
    for f, nodes in enumerate(chains.fragments):
        for v in np.unique(zone[nodes][zone[nodes] >= 0]):
            out[v].fragments.append(f)
    for pa, pb, gl in find_gap_pairs(X, chains, assign, cfg):
        v = zone[pa] if zone[pa] >= 0 else zone[pb]
        if v >= 0:
            out[v].gaps.append((pa, pb, gl))
    if not graph.recovered:     # recovered edges are polyline pieces: count is meaningless
        for k in range(E.shape[0]):
            n = len(assign.edge_chains[k])
            if n != cfg.helices_per_edge:
                for v in E[k]:
                    out[v].helix_count[k] = n
    for d in out:               # never edit around non-repairable (bend) vertices
        if not graph.repairable[d.vertex]:
            out[d.vertex] = VertexDefects(d.vertex)
    return out


# =============================================================================
# PHASE 3 - VERTEX-LOCAL REPAIR
# =============================================================================

@dataclass
class RepairLog:
    deleted: List[int] = field(default_factory=list)                 # original indices
    inserted: List[np.ndarray] = field(default_factory=list)         # coordinates
    inserted_vertex: List[int] = field(default_factory=list)
    operations: List[dict] = field(default_factory=list)

    def summary(self):
        ops = {}
        for o in self.operations:
            ops[o["op"]] = ops.get(o["op"], 0) + 1
        return dict(n_deleted=len(self.deleted),
                    n_inserted=int(sum(len(a) for a in self.inserted)), ops=ops)


def _trim_overdense(X, alive, pts, cfg, log: RepairLog, v, point_edge, overdense):
    """
    Minimal deletion until no link around `pts` violates the router's density rule
    (deg_i + deg_j <= 4).  Deletion priority (deterministic):
      1. near-duplicates (closer than 0.6 x rise to another live point),
      2. largest lateral deviation from the local helix axis, fitted on clean
         (non-over-dense) live points of the same target edge within 3 nm,
      3. highest degree, 4. highest index.
    Spatial proximity alone never decides: a point lying on the fitted axis with
    regular spacing is kept even if crowded.
    """
    r = cfg.link_radius - 1e-9
    region = set(int(p) for p in pts)
    for _ in range(4 * len(pts) + 4):
        idx = np.nonzero(alive)[0]
        tree = cKDTree(X[idx])
        loc = {int(g): l for l, g in enumerate(idx)}
        sub = [loc[p] for p in sorted(region) if p in loc]
        if not sub:
            return
        deg = {l: len(tree.query_ball_point(X[idx[l]], r)) - 1 for l in sub}
        bad = set()
        for l in sub:
            for m in tree.query_ball_point(X[idx[l]], r):
                if m == l:
                    continue
                dm = deg[m] if m in deg else len(tree.query_ball_point(X[idx[m]], r)) - 1
                if deg[l] + dm > 4:
                    bad.update((l, m))
        if not bad:
            return

        def axis_dev(l):
            g = int(idx[l])
            ctx = [int(idx[m]) for m in tree.query_ball_point(X[g], 3.0)
                   if not overdense[idx[m]] and point_edge[idx[m]] == point_edge[g] and idx[m] != g]
            if len(ctx) < 4:
                ctx = [int(idx[m]) for m in tree.query_ball_point(X[g], 1.5) if idx[m] != g]
            if len(ctx) < 2:
                return 0.0
            c, d = _fit_line(X[ctx])
            w = X[g] - c
            return float(np.linalg.norm(w - np.dot(w, d) * d))

        def key(l):
            dd, _ = tree.query(X[idx[l]], k=2)
            dup = dd[1] < 0.6 * cfg.bp_rise
            dg = deg[l] if l in deg else len(tree.query_ball_point(X[idx[l]], r)) - 1
            return (0 if dup else 1, -round(axis_dev(l), 6), -dg, -int(idx[l]))

        victim = min(sorted(bad), key=key)
        g = int(idx[victim])
        alive[g] = False
        region.discard(g)
        log.deleted.append(g)
        log.operations.append(dict(op="trim_overdense", vertex=int(v), point=g))
        region.update(int(idx[m]) for m in tree.query_ball_point(X[g], 2 * cfg.link_radius)
                      if alive[idx[m]])


def _cut_merged(X, alive, chain_nodes, Vpos, cfg, log: RepairLog, v):
    """Remove the fewest points at the bend of a chain that runs through vertex v."""
    nodes = [p for p in chain_nodes if alive[p]]
    if len(nodes) < 3:
        return
    d = np.linalg.norm(X[nodes] - Vpos, axis=1)
    j = int(np.argmin(d))
    lo, hi = j, j  # delete nodes[lo..hi]
    for _ in range(6):
        left = nodes[lo - 1] if lo - 1 >= 0 else None
        right = nodes[hi + 1] if hi + 1 < len(nodes) else None
        if left is None or right is None or np.linalg.norm(X[left] - X[right]) >= cfg.end_clearance:
            break
        # grow toward the side whose next point is closer to the vertex
        if lo - 2 >= 0 and (hi + 2 >= len(nodes) or
                            np.linalg.norm(X[nodes[lo - 1]] - Vpos) <= np.linalg.norm(X[nodes[hi + 1]] - Vpos)):
            lo -= 1
        else:
            hi += 1
    for p in nodes[lo:hi + 1]:
        alive[p] = False
        log.deleted.append(int(p))
    log.operations.append(dict(op="cut_merged", vertex=int(v), n=hi - lo + 1))


def _cut_branch(X, alive, chain_nodes, pairs_alive_fn, cfg, log, v, zone):
    """Delete branch nodes (degree >= 3 inside a kept chain) lying in the vertex zone."""
    deg = pairs_alive_fn()
    for p in chain_nodes:
        if alive[p] and deg[p] >= 3 and zone[p] == v:
            alive[p] = False
            log.deleted.append(int(p))
            log.operations.append(dict(op="cut_branch", vertex=int(v), point=int(p)))


def _bridge(X, pa, pb, cfg):
    g = X[pb] - X[pa]
    gl = np.linalg.norm(g)
    m = int(math.ceil(gl / cfg.bp_rise)) - 1
    if m < 1:
        return np.zeros((0, 3))
    u = np.arange(1, m + 1) / (m + 1)
    return X[pa][None] + u[:, None] * g[None]


def _bridge_is_clean(Xc, pa, pb, P, cfg):
    """Inserted points may only link to each other and to the two bridged ends,
    and must not create a link that violates the router's density rule."""
    r = cfg.link_radius - 1e-9
    Xn = np.vstack((Xc, P))
    tree = cKDTree(Xn)
    n0 = Xc.shape[0]
    new_ids = set(range(n0, n0 + len(P)))
    allowed = new_ids | {pa, pb}
    for i in sorted(new_ids) + [pa, pb]:
        nb = [j for j in tree.query_ball_point(Xn[i], r) if j != i]
        if i in new_ids and any(j not in allowed for j in nb):
            return False
        if len(nb) > 2:
            return False
    return True


def repair_vertices(X: np.ndarray, graph: TargetGraph, cfg: TGAVRConfig,
                    vertices: Optional[List[int]] = None):
    """
    One deterministic repair pass.  Returns (X_new, log, defects_before, assign).
    Only points inside defective vertex zones are deleted, and only gap-bridging
    points are inserted; every retained point keeps its exact coordinates.
    """
    ch = analyse_chains(X, cfg)
    asg = assign_edges(X, graph, ch, cfg)
    defects = detect_vertex_defects(X, ch, asg, graph, cfg)
    log = RepairLog()
    alive = np.ones(X.shape[0], dtype=bool)
    todo = [d for d in defects if d.defective and (vertices is None or d.vertex in vertices)]
    todo.sort(key=lambda d: (-d.score, d.vertex))

    def live_degree():
        idx = np.nonzero(alive)[0]
        pr = np.array(sorted(cKDTree(X[idx]).query_pairs(cfg.link_radius - 1e-9)), dtype=np.int64).reshape(-1, 2)
        deg = np.zeros(X.shape[0], int)
        if len(pr):
            np.add.at(deg, idx[pr.flatten()], 1)
        return deg

    for d in todo:
        v = d.vertex
        if d.overdense:
            _trim_overdense(X, alive, d.overdense, cfg, log, v, asg.point_edge, ch.overdense)
        for c in sorted(set(d.merged_chains)):
            _cut_merged(X, alive, ch.chains[c], asg.vertices[v], cfg, log, v)
        for c in sorted(set(d.branched_chains)):
            _cut_branch(X, alive, ch.chains[c], live_degree, cfg, log, v, asg.point_zone)

    X1 = X[alive]
    # bridge gaps (including those created by the trims above) inside the repaired zones
    ch1 = analyse_chains(X1, cfg)
    asg1 = assign_edges(X1, graph, ch1, cfg)
    zones = sorted({d.vertex for d in todo} | {d.vertex for d in defects if d.gaps and
                                               (vertices is None or d.vertex in vertices)})
    new_pts = []
    for v in zones:
        for pa, pb, gl in find_gap_pairs(X1, ch1, asg1, cfg, zone_vertex=v):
            P = _bridge(X1, pa, pb, cfg)
            if len(P) and not _bridge_is_clean(np.vstack([X1] + new_pts), pa, pb, P, cfg):
                log.operations.append(dict(op="bridge_rejected", vertex=int(v), gap=round(gl, 3)))
                continue
            if len(P):
                new_pts.append(P)
                log.inserted.append(P)
                log.inserted_vertex.append(int(v))
                log.operations.append(dict(op="bridge_gap", vertex=int(v), gap=round(gl, 3), n=len(P)))
    # collinear breaks of a duplex *inside* a target edge (outside every vertex zone):
    # the target says the path is continuous there, so the break is a missing-point
    # defect of an ordered duplex (e.g. a seam cutting both helices of a 2HB).
    if cfg.repair_interior_gaps and vertices is None:
        Xc = np.vstack([X1] + new_pts) if new_pts else X1
        for pa, pb, gl in find_gap_pairs(X1, ch1, asg1, cfg):
            if asg1.point_zone[pa] >= 0 or asg1.point_zone[pb] >= 0:
                continue
            P = _bridge(X1, pa, pb, cfg)
            if not len(P):
                continue
            if not _bridge_is_clean(Xc, pa, pb, P, cfg):
                log.operations.append(dict(op="bridge_rejected", vertex=-1, gap=round(gl, 3)))
                continue
            new_pts.append(P)
            Xc = np.vstack((Xc, P))
            log.inserted.append(P)
            log.inserted_vertex.append(-1)
            log.operations.append(dict(op="bridge_interior_gap", vertex=-1, edge=int(asg1.point_edge[pa]),
                                       gap=round(gl, 3), n=len(P)))
    X2 = np.vstack([X1] + new_pts) if new_pts else X1
    return X2, log, defects, asg


# =============================================================================
# ROUTING VALIDATION (independent of TGAVR; used for every method)
# =============================================================================

ROUTER_FAILURE_WARNINGS = {
    "FAILED to define single-strands": "single_strand_assignment",
    "not one-loop graph": "scaffold_not_single_loop",
    "No valid nick positions": "staple_no_nick",
}


def _split_order(seq_nicks: np.ndarray, lo: int, hi: int, start_ok: bool = True):
    """
    Optimal split of a linear sequence into segments of length [lo, hi]; a cut is allowed
    after position i iff seq_nicks[i] (the router's convention in design_staples).
    Minimises (longest segment, sum of squared deviations from 32).  Returns the list of
    segment end positions (exclusive) or None if no split exists.
    """
    n = len(seq_nicks)
    INF = (10 ** 9, 10 ** 9)
    best = [INF] * (n + 1)
    prev = [-1] * (n + 1)
    best[0] = (0, 0)
    ends = [k for k in range(1, n + 1) if k == n or seq_nicks[k - 1]]
    reach = [0]
    for k in ends:
        cand = INF
        for j in reach:
            L = k - j
            if L < lo or L > hi or best[j] == INF:
                continue
            c = (max(best[j][0], L), best[j][1] + (L - 32) ** 2)
            if c < cand:
                cand, prev[k] = c, j
        if cand != INF:
            best[k] = cand
            reach.append(k)
    if best[n] == INF:
        return None, INF
    cuts, k = [], n
    while k > 0:
        cuts.append(k)
        k = prev[k]
    return cuts[::-1], best[n]


def resplit_long_staples(routes, RB, scCO, stCO, SS, scaf, min_pairing_len: int = 4,
                         lo: int = 20, hi: int = 50):
    """
    Post-process the router's staple routes: every staple-graph component (the router's own
    graph: RB + stCO + SS minus internal helix edges) that still holds a route longer than
    ``hi`` is re-split at the router's own safe nick positions (``Nicks``) by dynamic
    programming.  Components that are not simple cycles/paths, or have no feasible split,
    keep the router's routes.  Returns (routes, report).
    """
    from torch_geometric.utils import degree as _deg
    from automated_design.ults import segmentation, Nicks
    from automated_design.routing import reconfig_STroutes
    n = scaf.size(0)
    E = torch.cat((RB, stCO, SS), dim=1)
    dg = _deg(E[0], n)
    mk = (dg[E[0]] == 3) & (dg[E[1]] == 3) & ((E[1] - E[0]).abs() < 2)
    E = E[:, ~mk]
    lab = segmentation(E).cpu().numpy()
    nicks = Nicks(scaf, scCO, stCO, RB, SS, min_pairing_len=min_pairing_len).cpu().numpy().astype(bool)
    En = E.t().cpu().numpy()
    rl = [r.detach().cpu().numpy() for r in routes]
    rep = dict(components=0, resplit=0, infeasible=0, skipped=0, new_min=None, new_max=None)
    bad = sorted({int(lab[r[0]]) for r in rl if len(r) > hi})
    if not bad:
        return routes, rep
    keep = [True] * len(rl)
    new = []
    for c in bad:
        rep["components"] += 1
        nodes = np.nonzero(lab == c)[0]
        mine = [i for i, r in enumerate(rl) if lab[r[0]] == c]
        cover = np.concatenate([rl[i] for i in mine])
        if cover.size != nodes.size or np.unique(cover).size != nodes.size:
            rep["skipped"] += 1
            continue
        sel = En[(lab[En[:, 0]] == c) & (lab[En[:, 1]] == c)]
        adj = {int(v): set() for v in nodes}
        for a, b in sel:
            if a != b:
                adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
        degs = np.array([len(adj[v]) for v in adj])
        if degs.max() > 2 or (degs < 1).any():
            rep["skipped"] += 1
            continue
        ends = [v for v in adj if len(adj[v]) == 1]
        cyc = len(ends) == 0
        if not cyc and len(ends) != 2:
            rep["skipped"] += 1
            continue
        start = int(nodes[0]) if cyc else ends[0]
        order, prev = [start], None
        while len(order) < nodes.size:
            nb = [w for w in adj[order[-1]] if w != prev]
            if not nb or (cyc and nb[0] == start):
                break
            prev = order[-1]
            order.append(nb[0])
        if len(order) != nodes.size:
            rep["skipped"] += 1
            continue
        order = np.array(order)
        best, cuts_best = None, None
        rots = [s for s in range(len(order)) if nicks[order[s - 1]]] if cyc else [0]
        for s in rots:
            seq = np.r_[order[s:], order[:s]] if cyc else order
            cuts, cost = _split_order(nicks[seq], lo, hi)
            if cuts is not None and (best is None or cost < best[0]):
                best, cuts_best = (cost, s), cuts
        if best is None:
            rep["infeasible"] += 1
            continue
        s = best[1]
        seq = np.r_[order[s:], order[:s]] if cyc else order
        segs, j = [], 0
        for k in cuts_best:
            segs.append(torch.as_tensor(seq[j:k], dtype=torch.long, device=routes[0].device))
            j = k
        for i in mine:
            keep[i] = False
        new.extend(reconfig_STroutes(segs, scaf))
        rep["resplit"] += 1
        L = [len(x) for x in segs]
        rep["new_min"] = min(L) if rep["new_min"] is None else min(rep["new_min"], min(L))
        rep["new_max"] = max(L) if rep["new_max"] is None else max(rep["new_max"], max(L))
    out = [r for r, k in zip(routes, keep) if k] + new
    return out, rep


def _staple_bad_nodes(RB, stCO, SS, n) -> np.ndarray:
    """Nodes whose degree in the router's staple graph (as built in staples_routing) is != 2."""
    from torch_geometric.utils import degree as _deg
    se = torch.cat((RB, stCO, SS), dim=1)
    d = _deg(se[0], n)
    m = (d[se[0]] == 3) & (d[se[1]] == 3) & ((se[1] - se[0]).abs() < 2)
    d2 = _deg(se[:, ~m][0], n)
    return (d2 != 2).nonzero().flatten().cpu().numpy()


def clean_staple_crossovers(RB, stCO, SS, n: int, max_removals: int = 12):
    """
    Resolve staple-crossover conflicts that make the router's staple graph non-simple (a base
    pair holding crossovers to two helices -> staple-graph degree 3/4, staples_routing raises).
    Staple crossovers are grouped into units (an edge a-b and its sibling a'-b' on the
    neighbouring base pairs of the same two helices, i.e. one double crossover).  Units next
    to defective nodes are removed greedily, each removal chosen to reduce the number of
    defective nodes most; nothing else is changed.  Returns (stCO, report); the input
    stCO is returned unchanged unless the defect count decreases.
    """
    bad0 = _staple_bad_nodes(RB, stCO, SS, n)
    rep = dict(bad_before=int(bad0.size), bad_after=int(bad0.size), removed_units=0,
               removed_edges=0, units_total=0)
    if bad0.size == 0:
        return stCO, rep
    st = stCO.t().cpu().numpy()
    und = sorted({(int(min(a, b)), int(max(a, b))) for a, b in st})
    rb = {(int(min(a, b)), int(max(a, b))) for a, b in RB.t().cpu().numpy()}
    eset = set(und)
    nb = {}
    for a, b in rb:
        nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    unit_of, units = {}, []
    for e in und:
        if e in unit_of:
            continue
        a, b = e
        sib = None
        for a2 in nb.get(a, []):
            for b2 in nb.get(b, []):
                f = (min(a2, b2), max(a2, b2))
                if f in eset and f not in unit_of and f != e:
                    sib = f
                    break
            if sib:
                break
        u = [e] + ([sib] if sib else [])
        for f in u:
            unit_of[f] = len(units)
        units.append(u)
    rep["units_total"] = len(units)
    alive = np.ones(len(units), bool)

    def build(al):
        E = [f for k, u in enumerate(units) if al[k] for f in u]
        if not E:
            return stCO[:, :0]
        E = np.array(E).T
        return torch.as_tensor(np.hstack((E, E[::-1])), dtype=stCO.dtype, device=stCO.device)

    cur, bad = stCO, bad0
    for _ in range(max_removals):
        if bad.size == 0:
            break
        near = set(bad.tolist())
        for q in list(near):
            near.update(nb.get(q, []))
        cands = sorted({unit_of[f] for f in und if alive[unit_of[f]] and (f[0] in near or f[1] in near)})
        best = None
        for k in cands:
            al = alive.copy(); al[k] = False
            t = build(al)
            b = _staple_bad_nodes(RB, t, SS, n)
            if best is None or b.size < best[0]:
                best = (b.size, k, t, b)
        if best is None or best[0] >= bad.size:
            break
        alive[best[1]] = False
        cur, bad = best[2], best[3]
        rep["removed_units"] += 1
        rep["removed_edges"] += len(units[best[1]])
    if bad.size >= bad0.size:
        rep.update(removed_units=0, removed_edges=0)
        return stCO, rep
    rep["bad_after"] = int(bad.size)
    return cur, rep


def _ss_bad_ends(n, ends, pairs):
    """Helix ends whose single-strand degree is not exactly one (a self pair counts as bad,
    as the router counts it twice)."""
    d = np.zeros(n, int)
    for a, b in pairs:
        d[a] += 1
        d[b] += 1
    return [int(e) for e in ends if d[e] != 1]


def rematch_single_strands(coords, RB, CO, SS, link: float = 5.0, max_nm: float = 6.0,
                           max_ends: int = 16):
    """
    Re-pair the helix ends of junctions where the router's nearest-neighbour single-strand
    pairing (assign_SSedges_wsubstr) gives an end 0 or >= 2 partners.  Such an end makes
    the scaffold / staple graph non-2-regular and staples_routing raises later on.

    A junction is a single-linkage cluster of helix ends (``link`` nm), closed under the
    existing single-strand pairs.  Its ends are re-paired by a maximum-cardinality matching
    of minimum total length over pairs of different router substructures (the router's own
    constraint) no longer than ``max_nm``; existing pairs get a 1 nm bonus so that the
    smallest change wins among near-equal matchings.  A junction is changed only if the
    matching is perfect.  The geometry, helices and crossovers are untouched.
    Returns (SS edge index in the router's undirected format, report).
    """
    import networkx as nx
    from torch_geometric.utils import degree
    from automated_design.ults import segmentation, Undirected_edge_index
    n = coords.size(0)
    dRB = degree(RB[0], num_nodes=n).numpy()
    ends = np.nonzero(dRB == 1)[0]
    pairs = {(min(a, b), max(a, b)) for a, b in SS.t().tolist()}
    bad = _ss_bad_ends(n, ends, pairs)
    rep = dict(bad_before=len(bad), bad_after=len(bad), junctions=0, unsolved=0,
               removed=[], added=[])
    if not bad:
        return SS, rep
    ei = torch.hstack((RB, CO))
    deg = degree(ei[0], num_nodes=n)
    m = torch.logical_and(deg[RB[0]] == 3, deg[RB[1]] == 3)
    seg = segmentation(torch.hstack((RB[:, ~m], CO)), min_nodes=0, return_extra=True)[0].numpy()
    P = coords.detach().cpu().numpy().astype(np.float64)
    pos = {int(e): i for i, e in enumerate(ends)}
    lab = _junction_clusters(P[ends], link)
    partner = {}
    for a, b in pairs:
        partner.setdefault(a, set()).add(b)
        partner.setdefault(b, set()).add(a)
    new_pairs, done = set(pairs), set()
    for e in bad:
        if e in done or e not in pos:
            continue
        K = {int(q) for q in ends[lab == lab[pos[e]]]}
        while True:                       # close under existing pairs
            grow = {q for k in K for q in partner.get(k, ()) if q != k and q in pos} - K
            if not grow:
                break
            K |= grow
        done |= K
        rep["junctions"] += 1
        if len(K) > max_ends or len(K) % 2:
            rep["unsolved"] += 1
            continue
        Kl = sorted(K)
        G = nx.Graph()
        G.add_nodes_from(Kl)
        for i, a in enumerate(Kl):
            for b in Kl[i + 1:]:
                dist = float(np.linalg.norm(P[a] - P[b]))
                old = (a, b) in pairs
                if old or (seg[a] != seg[b] and dist <= max_nm):
                    G.add_edge(a, b, weight=100.0 - dist + (1.0 if old else 0.0))
        M = nx.max_weight_matching(G, maxcardinality=True)
        if 2 * len(M) != len(K):
            rep["unsolved"] += 1
            continue
        M = {(min(a, b), max(a, b)) for a, b in M}
        old_k = {p for p in pairs if p[0] in K or p[1] in K}
        new_pairs = (new_pairs - old_k) | M
        rep["removed"] += sorted([list(map(int, p)) for p in old_k - M])
        rep["added"] += sorted([list(map(int, p)) for p in M - old_k])
    if not rep["added"] and not rep["removed"]:
        return SS, rep
    keep = [i for i, (a, b) in enumerate(SS.t().tolist()) if (min(a, b), max(a, b)) in new_pairs]
    add = sorted(new_pairs - pairs)
    cols = [SS[:, keep]]
    if add:
        cols.append(torch.tensor(add, dtype=SS.dtype, device=SS.device).t())
    SS2 = Undirected_edge_index(coords, torch.hstack(cols))
    rep["bad_after"] = len(_ss_bad_ends(n, ends, {(min(a, b), max(a, b))
                                                  for a, b in SS2.t().tolist()}))
    rep["added_nm"] = [round(float(np.linalg.norm(P[a] - P[b])), 2) for a, b in add]
    return SS2, rep


def _patched_ss_edges(orig, store: dict, link: float = 5.0, max_nm: float = 6.0,
                      max_ends: int = 16):
    """Wrap the router's assign_SingleStrand_edges: call it unchanged, then re-pair the
    junctions it leaves with badly paired helix ends (rematch_single_strands)."""
    def assign_SingleStrand_edges(coords, RBedge_index, alignCOs_index, bp_Hindex):
        SS, L = orig(coords, RBedge_index, alignCOs_index, bp_Hindex)
        try:
            SS2, rep = rematch_single_strands(coords, RBedge_index, alignCOs_index, SS,
                                              link=link, max_nm=max_nm, max_ends=max_ends)
        except Exception as e:      # never let the post-processing masquerade as a router error
            store["ss_rematch"] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
            return SS, L
        if rep["bad_before"]:
            store["ss_rematch"] = rep
        if SS2 is SS:
            return SS, L
        from automated_design.ults import segmentation
        return SS2, segmentation(torch.hstack((RBedge_index, SS2)))
    return assign_SingleStrand_edges


def _patched_staples_routing(orig, store: dict, resplit: bool = False, lo: int = 20,
                             cleanup: bool = False):
    """Wrap the router's staples_routing: optionally resolve staple-crossover conflicts
    first, call it unchanged, then optionally re-split long staples."""
    def staples_routing(RBedge_index, scCO_index, stCO_index, SS_index, scaf_route,
                        min_pairing_len=4, coords=None):
        if cleanup:
            try:
                stCO_index, crep = clean_staple_crossovers(RBedge_index, stCO_index, SS_index,
                                                           scaf_route.size(0))
                if crep["bad_before"]:
                    store["xover_cleanup"] = crep
            except Exception as e:
                store["xover_cleanup"] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
        routes, stap_len, bp_stap_index = orig(RBedge_index, scCO_index, stCO_index, SS_index,
                                               scaf_route, min_pairing_len=min_pairing_len,
                                               coords=coords)
        if not resplit or len(routes) == 0 or float(stap_len.max()) <= MAX_STAPLE_LEN:
            return routes, stap_len, bp_stap_index
        try:
            routes, rep = resplit_long_staples(routes, RBedge_index, scCO_index, stCO_index,
                                               SS_index, scaf_route,
                                               min_pairing_len=min_pairing_len, lo=lo)
        except Exception as e:      # never let the post-processing masquerade as a router error
            store.update(error=f"{type(e).__name__}: {str(e)[:120]}")
            return routes, stap_len, bp_stap_index
        store.update(rep)
        stap_len = torch.zeros(len(routes), device=stap_len.device)
        for i, r in enumerate(routes):
            stap_len[i] = len(r)
            bp_stap_index[r] = i
        return routes, stap_len, bp_stap_index
    return staples_routing


def _rs_kw(cfg) -> dict:
    return dict(staple_resplit=cfg.staple_resplit, staple_resplit_min=cfg.staple_resplit_min,
                staple_xover_cleanup=cfg.staple_xover_cleanup, ss_rematch=_ssr_kw(cfg))


def _ssr_kw(cfg):
    """rematch_single_strands options of a config, or None when the feature is off."""
    if not getattr(cfg, "ss_rematch", False):
        return None
    return dict(link=cfg.ss_rematch_link, max_nm=cfg.ss_rematch_max_nm,
                max_ends=cfg.ss_rematch_max_ends)


def run_router(coords: torch.Tensor, routing_kwargs: dict, seed: Optional[int] = 0,
               staple_resplit: bool = False, staple_resplit_min: int = 20,
               staple_xover_cleanup: bool = False, ss_rematch: Optional[dict] = None):
    """
    Call the unchanged ``automated_DnaOrigami`` and capture exceptions and warnings.
    Plot calls (``fig.show``) inside the router are suppressed for batch use.
    With ``staple_resplit`` the router's ``staples_routing`` result is post-processed by
    ``resplit_long_staples`` (opt-in; the router code itself is not modified).
    Returns (outputs or None, info dict).
    """
    import contextlib, io, random, time, traceback
    from automated_design import auto_design as _ad
    automated_DnaOrigami = _ad.automated_DnaOrigami
    _orig_sr = _ad.staples_routing
    _orig_ss = _ad.assign_SingleStrand_edges
    rs_store = {}
    if staple_resplit or staple_xover_cleanup:
        _ad.staples_routing = _patched_staples_routing(_orig_sr, rs_store, staple_resplit,
                                                       staple_resplit_min, staple_xover_cleanup)
    if ss_rematch is not None:
        _ad.assign_SingleStrand_edges = _patched_ss_edges(_orig_ss, rs_store, **ss_rematch)
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    info = dict(exception=None, stage=None, warnings=[], runtime_s=None)
    t0 = time.time()
    try:
        import plotly.graph_objects as go
        _show = go.Figure.show
        go.Figure.show = lambda *a, **k: None
    except Exception:          # plotly missing -> nothing to suppress
        go, _show = None, None
    out = None
    try:
        with warnings.catch_warnings(record=True) as W, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            try:
                out = automated_DnaOrigami(coords.clone(), show=False, **routing_kwargs)
            except Exception as e:     # routing failure
                tb = traceback.extract_tb(e.__traceback__)
                names = [f.name for f in tb]
                stage = next((n for n in names[::-1] if n in _STAGES), names[-1] if names else "?")
                info["exception"] = f"{type(e).__name__}: {str(e)[:120]}"
                info["stage"] = stage
                info["traceback"] = " <- ".join(names[::-1][:6])
        info["warnings"] = sorted({str(w.message)[:100] for w in W
                                   if any(k in str(w.message) for k in ROUTER_FAILURE_WARNINGS)})
    finally:
        if go is not None:
            go.Figure.show = _show
        _ad.staples_routing = _orig_sr
        _ad.assign_SingleStrand_edges = _orig_ss
    if "ss_rematch" in rs_store:
        info["ss_rematch"] = rs_store.pop("ss_rematch")
    if "xover_cleanup" in rs_store:
        info["xover_cleanup"] = rs_store.pop("xover_cleanup")
    if staple_resplit and rs_store:
        info["staple_resplit"] = dict(rs_store)
    info["runtime_s"] = round(time.time() - t0, 2)
    return out, info


_STAGES = ("assign_RegularBp_edges", "reindexing_base_pair", "edge2route", "gen_align_CrossOvers",
           "assign_SingleStrand_edges", "assign_SSedges_wsubstr", "loop_indexing",
           "scaffold_routing", "cycle_graph_to_route", "assign_triads", "gen_Staples_CrossOvers",
           "select_staple_COs", "staples_routing", "gen_dnatop", "assign_nucleidx_to_route")


MAX_STAPLE_LEN = 50      # router: design_staples retries until max segment < 51 and
                         # auto_design warns for staples > 50 bp
MAX_SCAFFOLD_NT = 7560   # longest scaffold sequence shipped (m13_7560); longer -> repeated seq


def validate_routing(out, info, max_staple_len: int = MAX_STAPLE_LEN,
                     max_scaffold_nt: int = MAX_SCAFFOLD_NT,
                     require_scaffold_fit: bool = False) -> dict:
    """
    Validated success = router returned AND
      * no router failure warning (single strands / one-loop scaffold / nick),
      * the scaffold route visits every base pair exactly once,
      * staple routes cover every base pair exactly once,
      * consecutive scaffold base pairs are < 10 nm apart (no bogus jumps),
      * every staple has <= 50 bound base pairs (the router's own acceptance limit).

    Whether the scaffold (incl. single-strand loops) fits the longest shipped M13 sequence
    is reported as ``scaffold_fits_M13``. It is a property of the design size (set by the
    generator's bp budget), not of routing, and the router itself accepts longer scaffolds
    by repeating the sequence; it only counts as a failure with ``require_scaffold_fit``.
    """
    res = dict(success=False, category=None, num_bps=None, scaffold_nt=None,
               max_staple=None, n_staples=None, scaffold_fits_M13=None)
    if out is None:
        res["category"] = "exception:" + str(info.get("stage"))
        return res
    (dnaGeom, dnaTop, coords, triads, bp_Hindex, coords_scaf, scaf_route,
     coords_stap, node_scaf_seq, stap_routes, unbound) = out
    n = coords.shape[0]
    res["num_bps"] = int(n)
    res["n_helices"] = int(bp_Hindex.max().item() + 1)
    try:   # scaffold entries come first in dnaTop; the last one has down == -1
        res["scaffold_nt"] = next(i + 1 for i, d in enumerate(dnaTop) if d.down == -1)
    except Exception:
        res["scaffold_nt"] = None
    fails = []
    for w in info.get("warnings", []):
        for k, cat in ROUTER_FAILURE_WARNINGS.items():
            if k in w:
                fails.append(cat)
    sr = scaf_route.detach().cpu().numpy()
    if sr.size != n or np.unique(sr).size != n:
        fails.append("scaffold_not_hamiltonian")
    else:
        step = np.linalg.norm(coords[sr[1:]].cpu().numpy() - coords[sr[:-1]].cpu().numpy(), axis=1)
        if step.max() > 10.0:
            fails.append("scaffold_long_jump")
    cov = np.zeros(n, int)
    lens = []
    for r in stap_routes:
        r = r.detach().cpu().numpy()
        np.add.at(cov, r, 1)
        lens.append(len(r))
    res["n_staples"] = len(lens)
    res["max_staple"] = int(max(lens)) if lens else None
    if not np.all(cov == 1):
        fails.append("staple_coverage")
    if lens and max(lens) > max_staple_len:
        fails.append("staple_too_long")
    if res["scaffold_nt"] is not None:
        res["scaffold_fits_M13"] = bool(res["scaffold_nt"] <= max_scaffold_nt)
        if require_scaffold_fit and not res["scaffold_fits_M13"]:
            fails.append("scaffold_exceeds_M13")
    res["success"] = len(fails) == 0
    res["category"] = "ok" if not fails else ",".join(sorted(set(fails)))
    return res


# =============================================================================
# FIDELITY / DISPLACEMENT METRICS
# =============================================================================

def target_fidelity(X: np.ndarray, aligned: TargetGraph, cfg: TGAVRConfig) -> dict:
    rmse, rmse_full, cov = alignment_quality(aligned, X, cfg)
    return dict(rmse_trim=rmse, rmse=rmse_full, coverage=cov)


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

def target_guided_vertex_repair(coords, target, cfg: Optional[TGAVRConfig] = None,
                                vertices: Optional[List[int]] = None):
    """
    Align the target to ``coords`` and repair defective vertex regions.

    Args:
        coords: generated base-pair centres [N,3] (torch or numpy, nm). Not modified.
        target: TargetGraph, or path/name of a .ply/.stl target.
        cfg: TGAVRConfig.
        vertices: restrict repair to these vertex ids (used by the retry loop).
    Returns:
        (repaired coords as torch.float32 [M,3], report dict).  If the target
        topology is unavailable or the alignment is unreliable, the original
        coordinates are returned unchanged with ``report['applied'] = False``.
    """
    cfg = cfg or TGAVRConfig()
    is_torch = isinstance(coords, torch.Tensor)
    X = _to_numpy(coords)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("coords must be [N,3]")
    report = dict(applied=False, reason=None, n_in=int(X.shape[0]))
    graph = target
    if not isinstance(target, TargetGraph):
        path = resolve_target_path(str(target)) if target is not None else None
        graph = load_target_graph(path, cfg.recover_pointcloud_topology, cfg) if path else None
    if graph is None:
        report["reason"] = ("target topology unavailable (only .ply/.stl targets carry edges; "
                            "set recover_pointcloud_topology=True for .pt targets)")
        return _as_out(X, is_torch, coords), report
    report["topology"] = "recovered" if graph.recovered else "verified"
    if graph.recovered:
        report["topology_diagnostics"] = graph.diagnostics
    al = align_target_to_points(graph, X, cfg)
    report["alignment"] = al.as_dict()
    if not al.reliable:
        report["reason"] = "alignment unreliable: " + al.message
        return _as_out(X, is_torch, coords), report
    aligned = graph.transformed(al.scale, al.R, al.t)
    X2, log, defects, asg = repair_vertices(X, aligned, cfg, vertices=vertices)
    edited = len(log.deleted) + sum(len(a) for a in log.inserted)
    report.update(defects=[d.as_dict() for d in defects if d.score > 0],
                  defective_vertices=[d.vertex for d in defects if d.defective],
                  vertex_scores={d.vertex: d.score for d in defects},
                  repair=log.summary(), operations=log.operations,
                  refined_vertex_shift=asg.vertex_shift.round(3).tolist(),
                  vertex_radius=asg.vertex_radius.round(3).tolist())
    if edited > cfg.max_edit_fraction * X.shape[0]:
        report["reason"] = f"repair rejected: {edited} edits > {cfg.max_edit_fraction:.0%} of points"
        return _as_out(X, is_torch, coords), report
    if cfg.total_bp_budget is not None and X2.shape[0] > cfg.total_bp_budget:
        report["reason"] = "repair rejected: total base-pair budget exceeded"
        return _as_out(X, is_torch, coords), report
    report["applied"] = edited > 0
    report["reason"] = "ok" if edited > 0 else "no vertex defects detected"
    report["n_out"] = int(X2.shape[0])
    report["_log"] = log
    report["_aligned"] = aligned
    return _as_out(X2, is_torch, coords), report


def _as_out(X, is_torch, ref):
    """Return the original tensor object when nothing changed, else a float32 tensor."""
    if is_torch and X.shape == tuple(ref.shape) and np.array_equal(X, _to_numpy(ref)):
        return ref
    return torch.as_tensor(X, dtype=torch.float32)


def _length_variants(X: np.ndarray, aligned: TargetGraph, cfg: TGAVRConfig, vertex: int,
                     delta: int):
    """
    Bounded duplex-length candidate at one vertex: change every helix end inside
    the vertex zone by ``delta`` base pairs (delta < 0 trims end points, delta > 0
    extends along the fitted end axis with the nominal rise).  Returns new X or None.
    """
    ch = analyse_chains(X, cfg)
    asg = assign_edges(X, aligned, ch, cfg)
    alive = np.ones(X.shape[0], bool)
    new = []
    for c, nodes in enumerate(ch.chains):
        for at_start in (True, False):
            end = nodes[0] if at_start else nodes[-1]
            if asg.point_zone[end] != vertex:
                continue
            if delta < 0:
                seq = nodes[:(-delta)] if at_start else nodes[::-1][:(-delta)]
                if len(nodes) - (-delta) < cfg.min_chain_points + 10:
                    continue
                alive[seq] = False
            else:
                ax = _end_axis(X, nodes, at_start, cfg.axis_fit_points)
                if ax is None:
                    continue
                step = np.linalg.norm(X[nodes[1]] - X[nodes[0]]) if len(nodes) > 1 else cfg.bp_rise
                step = float(np.clip(step, 0.3, 0.38))
                P = X[end][None] + (np.arange(1, delta + 1) * step)[:, None] * ax[None]
                new.append(P)
    Xn = np.vstack([X[alive]] + new) if new else X[alive]
    if Xn.shape[0] == X.shape[0] and np.allclose(Xn, X):
        return None
    # reject if the change created new over-density / merges in the zone
    ch2 = analyse_chains(Xn, cfg)
    if len(ch2.chains) != len(ch.chains) or ch2.overdense.sum() > ch.overdense.sum():
        return None
    return Xn


def single_strand_failure_points(coords: torch.Tensor, routing_kwargs: dict) -> Optional[np.ndarray]:
    """
    Re-run the router's own (unchanged) stages 1-3 - helix extraction, alignment
    crossovers, single-strand edges - exactly as automated_DnaOrigami calls them and
    return the positions of base pairs whose helix+single-strand degree is not 2.
    Such nodes break the scaffold cycle (scaffold 'not one-loop', duplicated route).
    """
    import contextlib, io
    from torch_geometric.utils import degree
    from automated_design.coords_to_edges import (assign_RegularBp_edges, gen_align_CrossOvers,
                                                  assign_SingleStrand_edges)
    from automated_design.ults import del_nodes
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            c, RB, H, endH, rem = assign_RegularBp_edges(coords.clone(), equi_len=0.4)
            if len(rem) > 0:
                c = del_nodes(c, rem)
                c, RB, H, endH, rem = assign_RegularBp_edges(c, equi_len=0.4)
            CO = gen_align_CrossOvers(c, H, RB, endH, effective_len=10,
                                      adjust_short_edges=routing_kwargs.get("adjust_short_edges", 4),
                                      adjust_long_edges=routing_kwargs.get("adjust_long_edges", 4))
            SS, _ = assign_SingleStrand_edges(c, RB, CO, H)
    except Exception:
        return None
    deg = degree(torch.hstack((RB, SS))[0], c.shape[0])
    return _to_numpy(c[deg != 2])


def failure_vertices(out, val: dict, aligned: TargetGraph, radius: float = 8.0,
                     coords: Optional[torch.Tensor] = None,
                     routing_kwargs: Optional[dict] = None) -> List[int]:
    """
    Vertices implicated by an observed routing failure.  The router only reports
    exceptions / warnings, so localisation uses what is observable:
      * over-long staples -> the vertices the longest offending staples pass through;
      * exceptions / scaffold-cycle / single-strand failures -> helix ends whose
        single-strand degree != 2 in the router's own stage-3 output.
    Only repairable vertices are returned; [] if nothing can be localised.
    """
    cat = str(val.get("category"))
    if (out is None or any(k in cat for k in ("single_strand", "scaffold_not", "exception"))) \
            and coords is not None:
        pts = single_strand_failure_points(coords, routing_kwargs or {})
        if pts is None or not len(pts):
            return []
        rep = np.nonzero(aligned.repairable)[0]
        if not rep.size:
            return []
        dv = np.linalg.norm(pts[:, None] - aligned.vertices[rep][None], axis=-1)
        near = dv.min(1) < 2 * radius
        cnt = np.bincount(dv.argmin(1)[near], minlength=rep.size)
        dmin = np.array([dv[:, k].min() for k in range(rep.size)])
        return [int(rep[k]) for k in sorted(np.nonzero(cnt)[0], key=lambda k: (-cnt[k], dmin[k]))]
    if out is None or "staple_too_long" not in cat:
        return []
    coords = _to_numpy(out[2])
    stap = [r.detach().cpu().numpy() for r in out[9]]
    L = np.array([len(r) for r in stap])
    hits: Dict[int, float] = {}
    for i in np.argsort(-L, kind="stable"):
        if L[i] <= MAX_STAPLE_LEN:
            break
        dv = np.linalg.norm(coords[stap[i]][:, None] - aligned.vertices[None], axis=-1)
        near = dv.min(1) < radius
        for v in np.unique(dv.argmin(1)[near]):
            hits[int(v)] = hits.get(int(v), 0.0) + float((dv.argmin(1)[near] == v).sum()) * L[i]
    return [v for v, _ in sorted(hits.items(), key=lambda z: (-z[1], z[0]))
            if aligned.repairable[v]]


# =============================================================================
# ROUTER-GUIDED JUNCTION REPAIR (M3): single-strand pairing + staple nicks
# =============================================================================
# Audit findings this section acts on (see docs/TGAVR_EXPERIMENT.md):
#  * assign_SSedges_wsubstr pairs every helix end with its nearest helix end of another
#    substructure.  When the nearest-neighbour relation is not mutual, one end receives two
#    single-strand edges; the router only warns and later raises inside staples_routing.
#  * Staple nicks are forbidden within min_pairing_len bp of single-strand ends and
#    crossovers.  A staple that wraps a junction through several short helix-end segments
#    then has no valid nick and stays > 50 bp.
# Both depend only on helix-end positions, so they are repaired by trimming / extending
# individual helix ends by a few bp.  Points are never moved.

@dataclass
class RouterGraph:
    """Router stage-1/2/3 graph of a point cloud (unchanged router functions)."""
    c: np.ndarray            # [n,3] router base-pair coordinates (exact copies of input rows)
    H: np.ndarray            # [n] helix id
    seg: np.ndarray          # [n] substructure id used by the single-strand pairing
    ends: np.ndarray         # RB-degree-1 nodes (helix ends)
    hend: np.ndarray         # [len(ends)] RB+alignment-CO degree 1 (first pairing pass)
    co_node: np.ndarray      # [n] node is part of an alignment crossover
    adj: list                # RB adjacency
    bad_nodes: np.ndarray    # nodes with RB+SS degree != 2 in the router's real stage 3
    n_loops: Optional[int] = None   # connected components of RB + SS (router's loop_indexing)
    island: Optional[np.ndarray] = None  # [n] component of RB + SS + alignment CO

    @property
    def n_bad(self) -> int:
        return int(self.bad_nodes.size)


def router_graph(coords, routing_kwargs: Optional[dict] = None) -> Optional[RouterGraph]:
    """Run the router's own stages 1-3 on ``coords``; None if the router raises."""
    import contextlib, io
    from torch_geometric.utils import degree
    from automated_design.coords_to_edges import (assign_RegularBp_edges, gen_align_CrossOvers,
                                                  assign_SingleStrand_edges)
    from automated_design.ults import del_nodes, segmentation
    kw = routing_kwargs or {}
    X = torch.as_tensor(_to_numpy(coords), dtype=torch.float32)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            c, RB, H, endH, rem = assign_RegularBp_edges(X.clone(), equi_len=0.4)
            if len(rem) > 0:
                c = del_nodes(c, rem)
                c, RB, H, endH, rem = assign_RegularBp_edges(c, equi_len=0.4)
            CO = gen_align_CrossOvers(c, H, RB, endH, effective_len=10,
                                      adjust_short_edges=kw.get("adjust_short_edges", 4),
                                      adjust_long_edges=kw.get("adjust_long_edges", 4))
            SS, _ = assign_SingleStrand_edges(c, RB, CO, H)
    except Exception:
        return None
    n = c.shape[0]
    ei = torch.hstack((RB, CO))
    deg = degree(ei[0], num_nodes=n)
    m = torch.logical_and(deg[RB[0]] == 3, deg[RB[1]] == 3)
    seg = segmentation(torch.hstack((RB[:, ~m], CO)), min_nodes=0, return_extra=True)[0]
    dRB = degree(RB[0], num_nodes=n)
    dSS = degree(SS[0], num_nodes=n)
    adj = [[] for _ in range(n)]
    for a, b in RB.t().tolist():
        adj[a].append(b)
    ends = (dRB == 1).nonzero().flatten().numpy()
    loops = _components(n, torch.hstack((RB, SS)).t().numpy())[0]
    island = _components(n, torch.hstack((RB, SS, CO)).t().numpy())[0]
    return RouterGraph(island=island, c=c.numpy().astype(np.float64), H=H.numpy(), seg=seg.numpy(), ends=ends,
                       hend=(deg == 1).numpy()[ends], co_node=(deg >= 3).numpy(), adj=adj,
                       bad_nodes=((dRB + dSS) != 2).nonzero().flatten().numpy(),
                       n_loops=int(loops.max()) + 1)


def _ss_pairing(P: np.ndarray, seg: np.ndarray, hend: np.ndarray):
    """
    Replica of the router's single-strand pairing (assign_SSedges_wsubstr after its
    segmentation step) restricted to a set of helix ends.  P [k,3] end positions,
    seg [k] substructure ids, hend [k] first-pass flags.  Returns (SS degree [k], pairs).
    Checked to reproduce the router exactly on all benchmark samples.
    """
    k = P.shape[0]
    if k < 2:
        return np.zeros(k, int), set()
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, 10.0)
    hi = np.nonzero(hend)[0]
    E = []
    if hi.size > 1:
        nnh = hi[D[np.ix_(hi, hi)].argmin(1)]
        E = [(int(i), int(j)) for i, j in zip(hi, nnh) if seg[i] != seg[j]]
    d = np.zeros(k, int)
    for a, b in E:
        d[a] += 1; d[b] += 1
    fix = [(a, b) for a, b in E if d[a] < 2 or d[b] < 2]
    E = E + [x for a, b in fix for x in ((a, b), (b, a))]
    for constrained in (True, False):
        out = np.zeros(k, int)
        for a, _ in E:
            out[a] += 1
        rem = np.nonzero(out < 1)[0]
        if rem.size:
            nn = rem[D[np.ix_(rem, rem)].argmin(1)]
            E += [(int(a), int(b)) for a, b in zip(rem, nn) if not constrained or seg[a] != seg[b]]
    pairs = {(min(a, b), max(a, b)) for a, b in E}
    d = np.zeros(k, int)
    for a, b in pairs:          # a self pair (no end within 10 nm) counts twice, as in the router
        d[a] += 1
        d[b] += 1
    return d, pairs


def _end_paths(rg: RouterGraph, depth: int):
    """Inward router-node path from every helix end (path[0] is the end)."""
    paths = []
    for e in rg.ends:
        path, prev = [int(e)], -1
        while len(path) < depth:
            nxt = [q for q in rg.adj[path[-1]] if q != prev]
            if not nxt:
                break
            prev = path[-1]
            path.append(nxt[0])
        paths.append(path)
    return paths


def _end_options(rg: RouterGraph, paths, tree, max_trim: int, max_ext: int, rise: float):
    """Admissible end positions: delta>0 trims delta bp (never into a crossover, helix
    keeps >= 22 bp), delta<0 extends along the end axis without approaching other points."""
    hlen = np.bincount(rg.H)
    opts = []
    for e, path in zip(rg.ends, paths):
        co_at = next((i for i, q in enumerate(path) if rg.co_node[q]), len(path))
        tmax = max(0, min(max_trim, co_at - 3, (hlen[rg.H[e]] - 22) // 2, len(path) - 1))
        o = {d: rg.c[path[d]] for d in range(tmax + 1)}
        if max_ext > 0 and len(path) >= 4:
            P = rg.c[path[:8]]
            ax = np.linalg.svd(P - P.mean(0))[2][0]
            if np.dot(ax, P[0] - P[-1]) < 0:
                ax = -ax
            for k in range(1, max_ext + 1):
                p = rg.c[e] + k * rise * ax
                if not set(tree.query_ball_point(p, 0.6)) <= set(path[:2]):
                    break
                o[-k] = p
        opts.append(o)
    return opts


def _junction_clusters(P: np.ndarray, link: float) -> np.ndarray:
    T = cKDTree(P)
    pairs = np.array(sorted(T.query_pairs(link)), dtype=int).reshape(-1, 2)
    return _components(P.shape[0], pairs)[0]


def _pairing_plan(rg: RouterGraph, cfg: TGAVRConfig, seed: int = 0):
    """Per junction with pairing failures: helix-end deltas (and optionally removal of a
    spurious short helix lying inside the junction) minimising, in this order, the number
    of badly paired ends, the single-strand lengths and the number of edited bp."""
    depth = max(cfg.pairing_max_trim + 12, 14)
    paths = _end_paths(rg, depth)
    tree = cKDTree(rg.c)
    opts = _end_options(rg, paths, tree, cfg.pairing_max_trim, cfg.pairing_max_ext, cfg.bp_rise)
    P0 = rg.c[rg.ends]
    seg_e, hend = rg.seg[rg.ends], rg.hend
    d0, _ = _ss_pairing(P0, seg_e, hend)
    lab = _junction_clusters(P0, cfg.pairing_cluster_link)
    T = cKDTree(P0)
    hlen = np.bincount(rg.H)
    rng = np.random.default_rng(seed)
    delta = np.zeros(len(rg.ends), int)
    removed = np.zeros(len(rg.ends), bool)
    plans = []
    for cid in sorted({int(lab[b]) for b in np.nonzero(d0 != 1)[0]}):
        mem = np.nonzero(lab == cid)[0]
        cen = P0[mem].mean(0)
        rad = cfg.pairing_context + np.linalg.norm(P0[mem] - cen, axis=1).max()
        ctx = np.array(sorted(set(T.query_ball_point(cen, rad)) | set(mem.tolist())))
        # short helices with both ends in this junction -> optional removal
        hs = {}
        for g in mem:
            hs.setdefault(int(rg.H[rg.ends[g]]), []).append(int(g))
        short = [h for h, gs in hs.items() if len(gs) == 2 and hlen[h] <= cfg.pairing_short_helix]
        if cfg.pairing_short_bridge:
            for h, gs in hs.items():
                if len(gs) == 1 and hlen[h] <= cfg.pairing_short_helix:
                    other = [int(g) for g in np.nonzero(rg.H[rg.ends] == h)[0] if g not in gs]
                    hs[h] = gs + other
                    short.append(h)
            ctx = np.array(sorted(set(ctx.tolist()) | {g for h in short for g in hs[h]}))

        def score(dl, drop):
            keep = np.array([not drop[g] for g in ctx])
            P = np.array([opts[g][dl[g]] for g in ctx])[keep]
            d, prs = _ss_pairing(P, seg_e[ctx][keep], hend[ctx][keep])
            idx = np.nonzero(keep)[0]
            inmem = np.isin(ctx[idx], mem)
            L = sum(np.linalg.norm(P[a] - P[b]) for a, b in prs if inmem[a] or inmem[b])
            edit = sum(abs(int(dl[g])) for g in mem if not drop[g]) + \
                sum(hlen[rg.H[rg.ends[g]]] for g in mem if drop[g]) / 2.0
            return (int((d != 1).sum()), int((d[inmem] != 1).sum()), round(L + 0.5 * edit, 3))

        best = None
        alts = {}
        drop_sets = [[]] + [[h] for h in short] + ([short] if len(short) > 1 else [])
        for drop_set in drop_sets:
            drop = removed.copy()
            for h in drop_set:
                drop[hs[h]] = True
            free = [g for g in mem if not drop[g]]
            for r in range(cfg.pairing_restarts + 1):
                dl = delta.copy()
                if r > 0:
                    for g in free:
                        dl[g] = rng.choice(list(opts[g].keys()))
                cur = score(dl, drop)
                improved = True
                while improved:
                    improved = False
                    for g in free:
                        for v in opts[g]:
                            if v == dl[g]:
                                continue
                            old, dl[g] = dl[g], v
                            s = score(dl, drop)
                            if s < cur:
                                cur, improved = s, True
                            else:
                                dl[g] = old
                if best is None or cur < best[0]:
                    best = (cur, dl.copy(), drop.copy())
                if cfg.pairing_alternatives and cur[1] == 0:
                    alts.setdefault((tuple(dl[mem]), tuple(drop[mem])), cur)
                if best[0][0] == 0 and r >= 2:
                    break
        if cfg.pairing_exhaustive and (best[0][1] > 0 or cfg.pairing_alternatives):
            # enumerate every combination of end positions of small junctions; it replaces the
            # local-search plan only if no drop set solved the junction, otherwise it only
            # supplies alternatives (the plan stays exactly as without this option)
            import itertools
            unsolved = best[0][1] > 0
            for drop_set in drop_sets:
                drop = removed.copy()
                for h in drop_set:
                    drop[hs[h]] = True
                free = [g for g in mem if not drop[g]]
                if not 0 < len(free) <= cfg.pairing_exhaustive_ends:
                    continue
                for combo in itertools.product(*[sorted(opts[g], key=abs) for g in free]):
                    dl = delta.copy()
                    dl[free] = combo
                    sc = score(dl, drop)
                    if unsolved and sc < best[0]:
                        best = (sc, dl.copy(), drop.copy())
                    if cfg.pairing_alternatives and sc[1] == 0:
                        alts.setdefault((tuple(dl[mem]), tuple(drop[mem])), sc)
        start = score(delta, removed)
        if best[0] < start:
            delta[mem] = best[1][mem]
            removed[mem] = best[2][mem]
            if cfg.pairing_short_bridge:      # both ends of a removed bridging helix
                for h in short:
                    if best[2][hs[h]].any():
                        removed[hs[h]] = True
            plans.append(dict(junction=int(cid), ends=[int(rg.ends[g]) for g in mem],
                              deltas={int(rg.ends[g]): int(delta[g]) for g in mem
                                      if delta[g] != 0 and not removed[g]},
                              removed_helices=sorted({int(rg.H[rg.ends[g]]) for g in mem if removed[g]}),
                              bad_before=int(start[1]), bad_after=int(best[0][1])))
            if cfg.pairing_alternatives:
                ranked = sorted(alts.items(), key=lambda kv: kv[1])[:cfg.pairing_alternatives]
                plans[-1]["_mem"] = mem
                plans[-1]["_alternatives"] = [(np.array(k[0]), np.array(k[1])) for k, _ in ranked]
    return plans, delta, removed, paths, opts


def _apply_router_edits(X: np.ndarray, rg: RouterGraph, plans, delta, removed, paths, opts,
                        only=None):
    """Turn router-node edits into deletions / insertions on the input point cloud."""
    key = {tuple(p): i for i, p in enumerate(X.astype(np.float32))}
    sel = None if only is None else {g for p in only for g in p["ends"]}
    drop, add = set(), []
    for g, e in enumerate(rg.ends):
        if sel is not None and int(e) not in sel:
            continue
        if removed[g]:
            for q in np.nonzero(rg.H == rg.H[e])[0]:
                drop.add(key[tuple(rg.c[q].astype(np.float32))])
        elif delta[g] > 0:
            for q in paths[g][:delta[g]]:
                drop.add(key[tuple(rg.c[q].astype(np.float32))])
        elif delta[g] < 0:
            add += [opts[g][-k] for k in range(1, -delta[g] + 1)]
    keep = np.ones(X.shape[0], bool)
    keep[list(drop)] = False
    Xn = np.vstack([X[keep]] + ([np.array(add)] if add else []))
    return Xn, len(drop), len(add)


def ss_pairing_repair(coords, routing_kwargs: dict, cfg: Optional[TGAVRConfig] = None):
    """
    Repair the router's single-strand pairing at junctions by bounded helix-end edits.
    Each round plans edits with the exact pairing replica, applies them and accepts the
    round only if the router's real stages 1-3 run and report fewer bad nodes.  If the
    router raises on the edited geometry, junctions are added one at a time instead.
    Returns (coords as torch.float32, report).
    """
    cfg = cfg or TGAVRConfig()
    X = _to_numpy(coords).astype(np.float64)
    rep = dict(rounds=[], bad_before=None, bad_after=None, deleted=0, inserted=0, junctions=[])
    rg = router_graph(X, routing_kwargs)
    if rg is None:
        rep["reason"] = "router stages 1-3 raise on the input"
        return torch.as_tensor(X, dtype=torch.float32), rep
    rep["bad_before"] = bad = rg.n_bad
    cur, cur_rg = X, rg
    k_in = {tuple(q) for q in X.astype(np.float32)}
    stages = [cfg]
    if cfg.pairing_escalate:
        from dataclasses import replace
        stages.append(replace(cfg, pairing_max_trim=cfg.pairing_escalate_trim,
                              pairing_max_ext=cfg.pairing_escalate_ext,
                              pairing_restarts=cfg.pairing_escalate_restarts))
    for si, r in ((si, r) for si, cf in enumerate(stages) for r in range(cf.pairing_rounds)):
        if bad == 0:
            break
        if rep["rounds"] and rep["rounds"][-1].get("stop") and rep["rounds"][-1]["stage"] == si:
            continue                           # this stage stopped improving
        plans, delta, removed, paths, opts = _pairing_plan(cur_rg, stages[si], seed=r + 100 * si)
        if not plans:
            rep["rounds"].append(dict(round=r, stage=si, mode="no plan", junctions=0, bad=bad,
                                      n_points=int(cur.shape[0]), stop=True))
            continue
        Xn, nd, ni = _apply_router_edits(cur, cur_rg, plans, delta, removed, paths, opts)
        rg2 = router_graph(Xn, routing_kwargs)
        mode = "all"
        if rg2 is None:                        # router raised: add junctions one by one
            mode = "incremental"
            Xn, rg2, used = cur, cur_rg, []
            for p in plans:
                Xt, _, _ = _apply_router_edits(cur, cur_rg, plans, delta, removed, paths, opts,
                                               only=used + [p])
                rt = router_graph(Xt, routing_kwargs)
                if rt is not None and rt.n_bad < rg2.n_bad:
                    Xn, rg2, used = Xt, rt, used + [p]
            plans = used
        nb = rg2.n_bad if rg2 is not None else None
        if nb is not None and cfg.pairing_max_edit_fraction < 1.0:
            k1 = {tuple(q) for q in Xn.astype(np.float32)}
            if len(k_in - k1) + len(k1 - k_in) > cfg.pairing_max_edit_fraction * X.shape[0]:
                nb = None                      # too large an edit: reject the round
        stop = nb is None or nb >= bad
        if stop and cfg.pairing_alternatives:
            # one junction at a time, other replica-valid solutions, real stages as the judge
            checks, found = 0, None
            for p in plans:
                for dlm, dpm in p.get("_alternatives", []):
                    if checks >= cfg.pairing_alt_checks:
                        break
                    d2 = np.zeros(len(cur_rg.ends), int); r2 = np.zeros(len(cur_rg.ends), bool)
                    d2[p["_mem"]] = dlm; r2[p["_mem"]] = dpm
                    Xt, _, _ = _apply_router_edits(cur, cur_rg, [p], d2, r2, paths, opts, only=[p])
                    checks += 1
                    rt = router_graph(Xt, routing_kwargs)
                    if rt is None or rt.n_bad >= bad:
                        continue
                    k1 = {tuple(q) for q in Xt.astype(np.float32)}
                    if len(k_in - k1) + len(k1 - k_in) > cfg.pairing_max_edit_fraction * X.shape[0]:
                        continue
                    found = (Xt, rt, p)
                    break
                if found is not None or checks >= cfg.pairing_alt_checks:
                    break
            rep["rounds"].append(dict(round=r, stage=si, mode=mode, junctions=len(plans), bad=nb,
                                      n_points=int(Xn.shape[0]), stop=found is None,
                                      alternative_checks=checks))
            if found is None:
                continue
            cur, cur_rg, bad = found[0], found[1], found[1].n_bad
            rep["junctions"].append({k: v for k, v in found[2].items() if not k.startswith("_")})
            continue
        rep["rounds"].append(dict(round=r, stage=si, mode=mode, junctions=len(plans), bad=nb,
                                  n_points=int(Xn.shape[0]), stop=stop))
        if stop:
            continue
        cur, cur_rg, bad = Xn, rg2, nb
        rep["junctions"] += [{k: v for k, v in p.items() if not k.startswith("_")} for p in plans]
    rep["bad_after"] = bad
    k0 = {tuple(p) for p in X.astype(np.float32)}
    k1 = {tuple(p) for p in cur.astype(np.float32)}
    rep["deleted"] = len(k0 - k1)
    rep["inserted"] = len(k1 - k0)
    rep["changed"] = bool(rep["deleted"] or rep["inserted"])
    if not rep["changed"]:
        return _as_out(X, isinstance(coords, torch.Tensor), coords), rep
    return torch.as_tensor(cur, dtype=torch.float32), rep


def loop_split_repair(coords, routing_kwargs: dict, cfg: Optional[TGAVRConfig] = None):
    """
    ``scaffold_routing`` raises (``amax`` of an empty tensor) when the regular base pairs and
    single strands form a single loop, because there is no crossover between loops.  Swap
    the partners of two single strands (a-b, c-e -> a-c, b-e or a-e, b-c) so that the loop
    splits into two, by trimming / extending only those four helix ends until the router's
    nearest-neighbour pairing produces the swapped pairs.  Candidates are ordered by the
    longest new single strand; each is checked with the router's real stages 1-3 (pairing
    valid and >= 2 loops).  Returns (coords as torch.float32, report).
    """
    import itertools
    from dataclasses import replace
    cfg = cfg or TGAVRConfig()
    X = _to_numpy(coords).astype(np.float64)
    rep = dict(loops_before=None, loops_after=None, swap=None, deleted=0, inserted=0, changed=False)
    rg = router_graph(X, routing_kwargs)
    if rg is None:
        return _as_out(X, isinstance(coords, torch.Tensor), coords), rep
    rep["loops_before"] = rep["loops_after"] = rg.n_loops
    if rg.n_loops != 1 or rg.n_bad > 0:
        return _as_out(X, isinstance(coords, torch.Tensor), coords), rep
    P0 = rg.c[rg.ends]
    seg, hend = rg.seg[rg.ends], rg.hend
    _, prs = _ss_pairing(P0, seg, hend)
    prs = sorted(prs)
    n = rg.c.shape[0]
    rb = np.array([(a, b) for a in range(n) for b in rg.adj[a] if a < b])

    def n_loops(pairs):
        E = np.vstack([rb, np.array([(rg.ends[a], rg.ends[b]) for a, b in pairs])])
        return int(_components(n, E)[0].max()) + 1

    cands = []
    for (a, b), (c, e) in itertools.combinations(prs, 2):
        for m in (((a, c), (b, e)), ((a, e), (b, c))):
            if any(seg[x] == seg[y] for x, y in m):
                continue
            m = tuple(tuple(sorted(x)) for x in m)
            other = [p for p in prs if p not in ((a, b), (c, e))]
            if n_loops(other + list(m)) >= 2:
                cands.append((max(np.linalg.norm(P0[x] - P0[y]) for x, y in m), m))
    cands.sort(key=lambda t: t[0])
    ecfg = replace(cfg, pairing_max_trim=cfg.pairing_escalate_trim, pairing_max_ext=cfg.pairing_escalate_ext)
    paths = _end_paths(rg, ecfg.pairing_max_trim + 12)
    opts = _end_options(rg, paths, cKDTree(rg.c), ecfg.pairing_max_trim, ecfg.pairing_max_ext, cfg.bp_rise)
    T = cKDTree(P0)
    rep["tried"] = []
    for _, m in cands[:cfg.loop_split_candidates]:
        Q = sorted({x for p in m for x in p})
        ctx = np.array(sorted({j for q in Q for j in T.query_ball_point(P0[q], cfg.pairing_context)}))
        loc = {g: i for i, g in enumerate(ctx)}
        want = {(loc[x], loc[y]) for x, y in m}

        def score(dl):
            P = np.array([opts[g][dl.get(g, 0)] for g in ctx])
            d, pp = _ss_pairing(P, seg[ctx], hend[ctx])
            return (len(want - set(pp)), int((d != 1).sum()), sum(abs(v) for v in dl.values()))

        # exhaustive over the four ends' admissible positions (small: <= 14^4); the local
        # replica (ends within pairing_context) only pre-selects, the global replica and the
        # router's real stages 1-3 decide
        good, best = [], None
        for combo in itertools.product(*[sorted(opts[q], key=abs) for q in Q]):
            dl = dict(zip(Q, combo))
            sc = score(dl)
            if best is None or sc < best:
                best = sc
            if sc[:2] == (0, 0):
                good.append((sc[2], dl))
        rec = dict(swap=[[int(rg.ends[x]) for x in p] for p in m], score=list(best), checked=[])
        rep["tried"].append(rec)
        good.sort(key=lambda t: t[0])
        for _, dl in good[:cfg.loop_split_checks]:
            P = P0.copy()
            for g, v in dl.items():
                P[g] = opts[g][v]
            d, pp = _ss_pairing(P, seg, hend)
            if int((d != 1).sum()) or not set(m) <= set(pp) or n_loops(sorted(pp)) < 2:
                rec["checked"].append("global replica rejects")
                continue
            delta = np.zeros(len(rg.ends), int)
            for g, v in dl.items():
                delta[g] = v
            Xn, nd, ni = _apply_router_edits(X, rg, [dict(ends=[int(rg.ends[g]) for g in Q])], delta,
                                             np.zeros(len(rg.ends), bool), paths, opts)
            rg2 = router_graph(Xn, routing_kwargs)
            rec["checked"].append(None if rg2 is None else [rg2.n_bad, rg2.n_loops])
            if rg2 is not None and rg2.n_bad == 0 and rg2.n_loops >= 2:
                rep.update(loops_after=rg2.n_loops, swap=rec["swap"], deleted=nd, inserted=ni,
                           changed=True, deltas={int(rg.ends[g]): int(v) for g, v in dl.items() if v})
                return torch.as_tensor(Xn, dtype=torch.float32), rep
    return _as_out(X, isinstance(coords, torch.Tensor), coords), rep


def isolated_fragment_removal(coords, routing_kwargs: dict, cfg: Optional[TGAVRConfig] = None):
    """
    Remove short spurious fragments that the scaffold can never visit: connected components
    of the router's own stage-1..3 graph (regular base pairs + single strands + alignment
    crossovers) other than the largest one, holding at most ``cfg.fragment_max_bp`` base
    pairs (e.g. an 11-bp helix whose two ends were paired with each other).  The router's
    scaffold_routing cannot reach them, so its scaffold route repeats nodes and
    staples_routing raises.  Accepted only if the router's single-strand pairing does not
    get worse.  Returns (coords, report).
    """
    cfg = cfg or TGAVRConfig()
    is_torch = isinstance(coords, torch.Tensor)
    X = _to_numpy(coords)
    rep = dict(changed=False, fragments=[], deleted=0)
    rg = router_graph(X, routing_kwargs)
    if rg is None or rg.island is None:
        return coords, rep
    lab = rg.island
    sizes = np.bincount(lab)
    if sizes.size < 2:
        return coords, rep
    main = int(sizes.argmax())
    small = [k for k in range(sizes.size) if k != main and sizes[k] <= cfg.fragment_max_bp]
    if not small:
        return coords, rep
    drop_c = rg.c[np.isin(lab, small)]
    key = {tuple(r): i for i, r in enumerate(X.astype(np.float32))}
    idx = [key.get(tuple(r)) for r in drop_c.astype(np.float32)]
    if any(i is None for i in idx):
        return coords, rep
    keep = np.ones(len(X), bool)
    keep[idx] = False
    Y = X[keep]
    g2 = router_graph(Y, routing_kwargs)
    if g2 is None or g2.n_bad > rg.n_bad:
        rep["rejected"] = True
        return coords, rep
    rep.update(changed=True, fragments=[int(sizes[k]) for k in small], deleted=int((~keep).sum()))
    return _as_out(Y, is_torch, coords), rep


def _router_stage1(X: torch.Tensor):
    """assign_RegularBp_edges exactly as automated_DnaOrigami calls it."""
    from automated_design.coords_to_edges import assign_RegularBp_edges
    from automated_design.ults import del_nodes
    c, RB, H, endH, rem = assign_RegularBp_edges(X.clone(), equi_len=0.4)
    if len(rem) > 0:
        c = del_nodes(c, rem)
        c, RB, H, endH, rem = assign_RegularBp_edges(c, equi_len=0.4)
    return c, RB, H, endH


def one_way_helix_pairs(coords, routing_kwargs: Optional[dict] = None, effective_len: int = 10):
    """
    Diagnose the ``gen_align_CrossOvers`` exception "expanded size of the tensor (k) must
    match the existing size (k+1)".  The router builds candidate crossovers with
    ``radius_graph(r=2.75)``, which keeps at most 32 neighbours per base pair; in dense
    regions the bp graph is therefore not symmetric.  After the helix-end mask
    (``EndHelix_bpmask``, ``effective_len`` hops) a helix pair can survive in one direction
    only, and ``To_undirected_edge_index`` (which assumes both directions) raises.
    Returns None if stage 2 runs, else dict(c, RB, H, pairs=[(a, b, supporting bp edges)]).
    """
    import contextlib, io
    from torch_geometric.nn import radius_graph
    from torch_geometric.utils import coalesce
    from automated_design.coords_to_edges import gen_align_CrossOvers, EndHelix_bpmask
    kw = routing_kwargs or {}
    X = torch.as_tensor(_to_numpy(coords), dtype=torch.float32)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        c, RB, H, endH = _router_stage1(X)
        try:
            gen_align_CrossOvers(c, H, RB, endH, effective_len=effective_len,
                                 adjust_short_edges=kw.get("adjust_short_edges", 4),
                                 adjust_long_edges=kw.get("adjust_long_edges", 4))
            return None
        except Exception:
            pass
        e = radius_graph(c, r=2.75, loop=False)            # same call as the router
        e = e[:, H[e[0]] != H[e[1]]]
        if endH.size(0) > 0:
            m = EndHelix_bpmask(c.size(0), RB, mask_seed_idx=endH.flatten(), k_hop=effective_len)
            e = e[:, m[e[0]] & m[e[1]]]
    hs = set(map(tuple, coalesce(H[e]).t().tolist()))
    pairs = []
    for a, b in sorted(hs):
        if (b, a) not in hs:
            sup = e[:, (H[e[0]] == a) & (H[e[1]] == b)]
            pairs.append((a, b, sup.t().tolist()))
    return dict(c=c.numpy().astype(np.float64), RB=RB, H=H.numpy(), pairs=pairs)


def crossover_symmetry_repair(coords, routing_kwargs: dict, cfg: Optional[TGAVRConfig] = None,
                              effective_len: int = 10):
    """
    Make ``gen_align_CrossOvers`` run by trimming a helix end so that the base pairs that
    support a one-way helix pair fall inside the router's own helix-end mask
    (a base pair d hops from an end is masked iff d <= effective_len, so a trim of
    d - effective_len bp suffices).  Candidates are tried smallest trim first and accepted
    when the router's real stage 2 succeeds, or the number of one-way pairs decreases.
    Returns (coords as torch.float32, report).
    """
    cfg = cfg or TGAVRConfig()
    X = _to_numpy(coords).astype(np.float64)
    rep = dict(one_way_before=None, one_way_after=None, trims=[], deleted=0, changed=False)
    diag = one_way_helix_pairs(X, routing_kwargs, effective_len)
    if diag is None:
        rep["one_way_before"] = rep["one_way_after"] = 0
        return _as_out(X, isinstance(coords, torch.Tensor), coords), rep
    rep["one_way_before"] = n_cur = len(diag["pairs"])
    cur = X
    for _ in range(cfg.crossover_rounds):
        if diag is None or not diag["pairs"]:
            break
        c, RB, H = diag["c"], diag["RB"], diag["H"]
        n = c.shape[0]
        adj = [[] for _ in range(n)]
        for u, v in RB.t().tolist():
            adj[u].append(v)
        hlen = np.bincount(H)
        key = {tuple(p): i for i, p in enumerate(cur.astype(np.float32))}

        def walk(q):                     # (end node, path end->q) for both ends of q's helix
            res = []
            for s in adj[q]:
                path, prev, cur_ = [q], q, s
                while True:
                    path.append(cur_)
                    nxt = [w for w in adj[cur_] if w != prev]
                    if not nxt or len(path) > effective_len + cfg.crossover_max_trim + 2:
                        break
                    prev, cur_ = cur_, nxt[0]
                if not [w for w in adj[path[-1]] if w != path[-2]]:
                    res.append(path[::-1])          # path[0] = helix end, path[-1] = q
            if not adj[q]:
                res.append([q])
            return res

        cands = []
        for a, b, sup in diag["pairs"]:
            for q in sorted({x for ed in sup for x in ed}):
                for path in walk(q):
                    d = len(path) - 1                 # hops from the end
                    t = d - effective_len
                    if 1 <= t <= cfg.crossover_max_trim and hlen[H[q]] - t >= 22:
                        cands.append((t, int(q), [int(x) for x in path[:t]]))
        cands = sorted({(t, q, tuple(p)) for t, q, p in cands})
        accepted = None
        for t, q, drop in cands:
            rows = [key.get(tuple(c[x].astype(np.float32))) for x in drop]
            if any(r is None for r in rows):
                continue
            keep = np.ones(cur.shape[0], bool)
            keep[rows] = False
            Xn = cur[keep]
            d2 = one_way_helix_pairs(Xn, routing_kwargs, effective_len)
            n2 = 0 if d2 is None else len(d2["pairs"])
            if d2 is None or n2 < n_cur:
                accepted = (Xn, d2, n2, dict(trim=t, node=q, deleted=len(rows)))
                break
        if accepted is None:
            break
        cur, diag, n_cur, info = accepted
        rep["trims"].append(info)
    rep["one_way_after"] = n_cur if diag is not None else 0
    rep["stage2_ok"] = diag is None
    k0 = {tuple(p) for p in X.astype(np.float32)}
    k1 = {tuple(p) for p in cur.astype(np.float32)}
    rep["deleted"] = len(k0 - k1)
    rep["changed"] = bool(rep["deleted"])
    if not rep["changed"]:
        return _as_out(X, isinstance(coords, torch.Tensor), coords), rep
    return torch.as_tensor(cur, dtype=torch.float32), rep


def staple_graph_defects(coords, routing_kwargs: dict, seed: Optional[int] = 0,
                         xover_cleanup: bool = False,
                         ss_rematch: Optional[dict] = None) -> Optional[dict]:
    """
    Re-run the router's own stages up to the staple graph of ``staples_routing`` (same
    calls, same order as automated_DnaOrigami) and report router coordinates of
      * ``broken``: base pairs whose staple-graph degree is not 2 (staples_routing then
        walks into a dead end and raises), and
      * ``no_nick``: base pairs of staple components without any valid nick position.
    Returns None if the router raises before the staple graph exists.
    """
    import contextlib, io, random
    from torch_geometric.utils import degree
    from automated_design.coords_to_edges import (assign_RegularBp_edges, gen_align_CrossOvers,
                                                  assign_SingleStrand_edges, gen_Staples_CrossOvers)
    from automated_design.ults import del_nodes, loop_indexing, bps_to_bases, Nicks, segmentation
    from automated_design.routing import scaffold_routing
    from automated_design.coords_to_triads import assign_triads
    kw = routing_kwargs
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    X = torch.as_tensor(_to_numpy(coords), dtype=torch.float32)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            c, RB, H, endH, rem = assign_RegularBp_edges(X.clone(), equi_len=0.4)
            if len(rem) > 0:
                c = del_nodes(c, rem)
                c, RB, H, endH, rem = assign_RegularBp_edges(c, equi_len=0.4)
            CO = gen_align_CrossOvers(c, H, RB, endH, effective_len=10,
                                      adjust_short_edges=kw.get("adjust_short_edges", 4),
                                      adjust_long_edges=kw.get("adjust_long_edges", 4))
            SS, L = assign_SingleStrand_edges(c, RB, CO, H)
            if ss_rematch is not None:   # same post-processing as run_router(ss_rematch=...)
                SS, _ = rematch_single_strands(c, RB, CO, SS, **ss_rematch)
            L, _ = loop_indexing(RB, SS)
            scaf, scCO, CO_noscaf = scaffold_routing(CO, RB, SS, H, L, c)
            triads, _, scaf, scCO = assign_triads(CO, RB, SS, scCO, c, H, L, scaf,
                                                  one_turn_angle=(-2) * math.pi / 10.5)
            _, cst = bps_to_bases(c, triads)
            stCO = gen_Staples_CrossOvers(cst, H, scaf, scCO, CO_noscaf, RB, SS,
                                          effective_len=kw.get("min_COs_distance", 10),
                                          min_pairing_len=kw.get("min_pairing_len", 4),
                                          adaptive=kw.get("adaptive_staples", False))
            if xover_cleanup:      # same pre-processing as run_router(staple_xover_cleanup=True)
                stCO, _ = clean_staple_crossovers(RB, stCO, SS, scaf.size(0))
            # staple graph exactly as in routing.staples_routing
            n = scaf.size(0)
            se = torch.cat((RB, stCO, SS), dim=1)
            deg = degree(se[0, :], n)
            m = torch.logical_and(deg[se[0]] == 3, deg[se[1]] == 3)
            m = torch.logical_and(m, torch.abs(se[1] - se[0]) < 2)
            se = se[:, ~m]
            comp = segmentation(se)
            nicks = Nicks(scaf, scCO, stCO, RB, SS, min_pairing_len=kw.get("min_pairing_len", 4))
    except Exception:
        return None
    d2 = degree(se[0], n)
    broken = (d2 != 2).nonzero().flatten().numpy()
    no_nick = []
    for s in range(int(comp.max()) + 1):
        idx = (comp == s).nonzero().flatten()
        if idx.numel() and not bool(nicks[idx].any()):
            no_nick += idx.tolist()
    # staples whose two ends are the two ends of one single-strand edge: the staple is a
    # closed loop opened at the single strand, and assign_SStrand_to_route (gen_dnatop)
    # raises because it splits routes at consecutive pairs of single-strand ends
    closed = []
    try:
        from automated_design.routing import staples_routing
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            routes, _, _ = staples_routing(RB, scCO, stCO, SS, scaf,
                                           min_pairing_len=kw.get("min_pairing_len", 4))
        ss_pairs = set(map(tuple, SS.t().tolist()))
        for r in routes:
            if len(r) > 1 and (int(r[0]), int(r[-1])) in ss_pairs:
                closed += [int(r[0]), int(r[-1])]
    except Exception:
        pass
    cn = c.numpy().astype(np.float64)
    partners = stCO[1, torch.isin(stCO[0], torch.as_tensor(broken, dtype=stCO.dtype))].numpy()
    return dict(broken=cn[broken], no_nick=cn[np.array(no_nick, dtype=int)],
                ss_closed=cn[np.array(closed, dtype=int)], broken_partners=cn[partners])


def _staple_score(out, val, info=None) -> Tuple[int, int, int, int]:
    """(failed, raised, staples > 50 bp + staples without a valid nick, total excess bp)."""
    if out is None:
        return (1, 1, 10 ** 6, 10 ** 6)
    lens = [len(r) for r in out[9]]
    over = [l for l in lens if l > MAX_STAPLE_LEN]
    ws = (info or {}).get("warnings", [])
    no_nick = sum("No valid nick" in w for w in ws) or int("staple_no_nick" in str(val.get("category")))
    return (0 if val["success"] else 1, 0, len(over) + no_nick,
            sum(l - MAX_STAPLE_LEN for l in over))


def _staple_moves(X: np.ndarray, out, rg: RouterGraph, cfg: TGAVRConfig, targets=None):
    """Single-helix-end moves, ordered by step size first (all +1 trims, then -1
    extensions, ...).  Ends are those traversed by the staples > 50 bp (longest first),
    then helix ends within 6 nm of ``targets`` (router coordinates of staple-graph defects)."""
    idx_rg = {tuple(p): i for i, p in enumerate(rg.c.astype(np.float32))}
    end_pos = {int(e): g for g, e in enumerate(rg.ends)}
    gs = []
    if out is not None:
        c_out = out[2].detach().cpu().numpy().astype(np.float32)
        routes = sorted((r.detach().cpu().numpy() for r in out[9] if len(r) > MAX_STAPLE_LEN),
                        key=len, reverse=True)
        for r in routes:
            for q in r:
                i = idx_rg.get(tuple(c_out[q]))
                if i is not None and i in end_pos and end_pos[i] not in gs:
                    gs.append(end_pos[i])
    if targets is not None and len(targets):
        P = rg.c[rg.ends]
        dist = cKDTree(np.asarray(targets)).query(P)[0]
        for g in np.argsort(dist):
            if dist[g] <= 6.0 and int(g) not in gs:
                gs.append(int(g))
    depth = max(abs(s) for s in cfg.staple_steps) + 12
    paths = _end_paths(rg, depth)
    tree = cKDTree(rg.c)
    opts = _end_options(rg, paths, tree, max(cfg.staple_steps), max(0, -min(cfg.staple_steps)),
                        cfg.bp_rise)
    moves = []
    for s in cfg.staple_steps:
        for g in gs:
            if s in opts[g]:
                moves.append((g, s))
    return moves, paths, opts


def staple_search(X, out, val, info, routing_kwargs: dict, cfg: TGAVRConfig, budget: int,
                  seed: Optional[int], attempts: list):
    """
    Greedy search over single helix-end trims / extensions with the routing call as
    evaluator, for staple-level failures once the single-strand pairing is valid:
      * staples > 50 bp       -> ends the over-long staples pass through;
      * staples_routing raised, or a staple has no valid nick
                              -> ends near the staple-graph defects reported by
                                 staple_graph_defects (one router run, counted as a call).
    A move is routed only if the pairing replica says it keeps the pairing valid.
    Accepts strictly better (failed, raised, #bad staples, excess) scores.
    Returns (out, X, val, router calls used).
    """
    import time
    Xc = _to_numpy(X).astype(np.float64)
    best = _staple_score(out, val, info)
    calls, tried = 0, set()
    pending = None                               # diagnosis already computed for Xc
    n_def = lambda d: sum(len(d[k]) for k in ("broken", "no_nick", "ss_closed"))
    while best[0] == 1 and calls < budget:
        rg = router_graph(Xc, routing_kwargs)
        if rg is None or rg.n_bad > 0:          # pairing problem: not handled here
            break
        cat = str(val.get("category"))
        targets, diag = None, None
        staple_cats = {"exception:staples_routing"}
        if cfg.closed_staple_search:
            staple_cats |= {"exception:assign_nucleidx_to_route", "exception:gen_dnatop"}
        if out is None or "staple_no_nick" in cat:
            if out is None and cat not in staple_cats:
                break                            # raised elsewhere: not a staple problem
            if pending is not None:
                diag, pending = pending, None
            else:
                t0 = time.time()
                diag = staple_graph_defects(Xc, routing_kwargs, seed, cfg.staple_xover_cleanup, _ssr_kw(cfg))
                calls += 1
                attempts.append(dict(kind="staple diagnosis", n_bps_in=int(Xc.shape[0]), success=False,
                                     category="diagnosis", max_staple=None, exception=None, stage=None,
                                     warnings=[], runtime_s=round(time.time() - t0, 2),
                                     broken=None if diag is None else int(len(diag["broken"])),
                                     no_nick=None if diag is None else int(len(diag["no_nick"])),
                                     ss_closed=None if diag is None else int(len(diag["ss_closed"]))))
            if diag is not None:
                parts = [diag["broken"], diag["no_nick"]]
                if cfg.closed_staple_search:
                    parts.append(diag["ss_closed"])
                targets = np.vstack([p.reshape(-1, 3) for p in parts])
            if (targets is None or not len(targets)) and out is None:
                break
        moves, paths, opts = _staple_moves(Xc, out, rg, cfg, targets)
        moves = [[m] for m in moves]
        if cfg.closed_staple_search and diag is not None and len(diag["ss_closed"]):
            # a staple closed by a single strand sits at a U-turn (two helix ends paired
            # with each other): move all of those ends together, same step
            _, gc = cKDTree(rg.c[rg.ends]).query(diag["ss_closed"])
            gc = list(dict.fromkeys(int(g) for g in gc))
            joint = []
            for s in cfg.staple_steps:
                mv = [(g, s) for g in gc if s in opts[g]]
                if len(mv) > 1:
                    joint.append(mv)
            moves = joint + moves
        if cfg.staple_helix_targets and diag is not None and len(diag["broken"]):
            # conflicting staple crossovers sit mid-helix, far from any end: a 1-3 bp change at
            # either end of the helices involved shifts their twist phase and the crossovers
            key = {tuple(q): i for i, q in enumerate(rg.c.astype(np.float32))}
            nodes = [key.get(tuple(q)) for q in np.vstack([diag["broken"], diag["broken_partners"]])
                     .astype(np.float32)]
            hs = {int(rg.H[q]) for q in nodes if q is not None}
            gh = [g for g in range(len(rg.ends)) if int(rg.H[rg.ends[g]]) in hs]
            extra = [[(g, st)] for st in cfg.staple_steps for g in gh if st in opts[g]]
            moves = extra + [m for m in moves if m not in extra]
        P0 = rg.c[rg.ends]
        seg_e = rg.seg[rg.ends]
        d0, _ = _ss_pairing(P0, seg_e, rg.hend)
        bad0 = int((d0 != 1).sum())
        improved = False
        for mv in moves:
            if calls >= budget:
                break
            k = tuple(sorted((tuple(np.round(P0[g], 3)), s) for g, s in mv))
            if k in tried:
                continue
            tried.add(k)
            P = P0.copy()
            dl = np.zeros(len(rg.ends), int)
            for g, s in mv:
                P[g] = opts[g][s]
                dl[g] = s
            if int((_ss_pairing(P, seg_e, rg.hend)[0] != 1).sum()) > bad0:
                continue
            Xn, _, _ = _apply_router_edits(Xc, rg, [dict(ends=[int(rg.ends[g]) for g, _ in mv])], dl,
                                           np.zeros(len(rg.ends), bool), paths, opts)
            Xt = torch.as_tensor(Xn, dtype=torch.float32)
            kind = (f"staple end{int(rg.ends[mv[0][0]])} d{mv[0][1]:+d}" if len(mv) == 1 else
                    f"staple U-turn x{len(mv)} d{mv[0][1]:+d}")
            if cfg.staple_defect_guided and out is None and diag is not None and n_def(diag) > 0:
                # the router raises: judge the move by the staple-graph defects (one replay,
                # counted as a call) and route only once a move removes defects
                t0 = time.time()
                d2 = staple_graph_defects(Xn, routing_kwargs, seed, cfg.staple_xover_cleanup, _ssr_kw(cfg))
                calls += 1
                attempts.append(dict(kind=kind + " (defects)", n_bps_in=int(Xt.shape[0]), success=False,
                                     category="diagnosis", max_staple=None, exception=None, stage=None,
                                     warnings=[], runtime_s=round(time.time() - t0, 2),
                                     defects=None if d2 is None else n_def(d2)))
                if d2 is None or n_def(d2) >= n_def(diag):
                    continue
                if n_def(d2) > 0:
                    Xc, pending, improved = Xn, d2, True      # progress without routing
                    break
                if calls >= budget:
                    Xc, improved = Xn, True
                    break
            o2, i2 = run_router(Xt, routing_kwargs, seed=seed, **_rs_kw(cfg))
            v2 = validate_routing(o2, i2)
            calls += 1
            attempts.append(dict(kind=kind, n_bps_in=int(Xt.shape[0]),
                                 **v2, **{kk: i2[kk] for kk in ("exception", "stage", "warnings", "runtime_s")}))
            sc = _staple_score(o2, v2, i2)
            if sc < best or (cfg.staple_defect_guided and out is None and diag is not None
                             and n_def(diag) > 0):
                # (defect-guided: this move removed every staple-graph defect; keep it)
                Xc, out, val, best, improved = Xn, o2, v2, min(sc, best), True
                break
        if not improved:
            break
    return out, torch.as_tensor(Xc, dtype=torch.float32), val, calls


def tgavr_route(coords: torch.Tensor, target, routing_kwargs: dict,
                cfg: Optional[TGAVRConfig] = None, seed: Optional[int] = 0):
    """
    TGAVR + unchanged routing with failure-aware bounded retry.

    M1 (cfg.length_search=False): repair -> route once (if repair changed nothing,
        the result is identical to the baseline).
    M2 (cfg.length_search=True): if the repaired geometry fails validation, try
        bounded per-vertex duplex-length variants (cfg.length_search_steps) at the
        most defective vertices, at most cfg.max_routing_attempts routing calls.
    M3 (cfg.pairing_repair / cfg.staple_search, optionally with length_search):
        M1 (undone if it adds single-strand pairing failures, cfg.guard_m1_pairing),
        then single-strand pairing repair at junctions (no routing call), route, then
        per-helix-end search on staples > 50 bp, then the M2 vertex search with the
        remaining budget.  All steps share cfg.max_routing_attempts.
    Returns (router outputs or None, coords actually routed, report).
    """
    cfg = cfg or TGAVRConfig()
    X_rep, rep = target_guided_vertex_repair(coords, target, cfg)
    m3 = cfg.pairing_repair or cfg.staple_search
    kinds = ["repair"] if rep.get("applied") else []
    if m3 and cfg.guard_m1_pairing and rep.get("applied"):
        def _graph(Y):
            g = router_graph(Y, routing_kwargs)
            if g is None and cfg.crossover_repair:   # judge M1 after the stage-2 fix
                g = router_graph(crossover_symmetry_repair(Y, routing_kwargs, cfg)[0], routing_kwargs)
            return g
        g0, g1 = _graph(coords), _graph(X_rep)
        b0 = g0.n_bad if g0 is not None else 10 ** 6
        b1 = g1.n_bad if g1 is not None else 10 ** 6
        rep["m1_pairing_bad"] = [b0, b1]
        if b1 > b0:
            rep["m1_reverted"] = rep.get("repair")
            rep["repair"], rep["applied"] = {}, False
            rep["reason"] = f"M1 undone: single-strand pairing failures {b0} -> {b1}"
            if "_log" in rep:
                rep["_log"] = RepairLog()
            X_rep, kinds = coords, []
    if cfg.crossover_repair:
        X_rep, crep = crossover_symmetry_repair(X_rep, routing_kwargs, cfg)
        rep["crossover"] = crep
        if crep.get("changed"):
            kinds.append("crossover")
    if cfg.pairing_repair:
        X_rep, prep = ss_pairing_repair(X_rep, routing_kwargs, cfg)
        rep["pairing"] = prep
        if prep.get("changed"):
            kinds.append("pairing")
    if cfg.drop_isolated_fragments:
        X_rep, frep = isolated_fragment_removal(X_rep, routing_kwargs, cfg)
        rep["fragments"] = frep
        if frep.get("changed"):
            kinds.append("fragments")
    if cfg.loop_split:
        X_rep, lrep = loop_split_repair(X_rep, routing_kwargs, cfg)
        rep["loop_split"] = lrep
        if lrep.get("changed"):
            kinds.append("loop split")
    attempts = []
    out, info = run_router(X_rep, routing_kwargs, seed=seed, **_rs_kw(cfg))
    val = validate_routing(out, info)
    attempts.append(dict(kind="+".join(kinds) if kinds else "unchanged",
                         n_bps_in=int(X_rep.shape[0]), **val,
                         **{k: info[k] for k in ("exception", "stage", "warnings", "runtime_s")},
                         **{k: info[k] for k in ("staple_resplit", "xover_cleanup", "ss_rematch") if k in info}))
    best = (out, X_rep, val)
    n_calls = 1
    base_X = X_rep
    if not val["success"] and cfg.staple_search and n_calls < cfg.max_routing_attempts:
        o2, X2, v2, used = staple_search(X_rep, out, val, info, routing_kwargs, cfg,
                                         cfg.max_routing_attempts - n_calls, seed, attempts)
        n_calls += used
        if used:
            base_X, out, val = X2, o2, v2      # continue from the best geometry found
        if v2["success"]:
            best = (o2, X2, v2)
    if not best[2]["success"] and cfg.length_search and "_aligned" in rep \
            and n_calls < cfg.max_routing_attempts:
        aligned = rep["_aligned"]
        X0 = _to_numpy(base_X)
        scores = rep.get("vertex_scores", {})
        # failure-located vertices first, then geometric defect score
        located = failure_vertices(out, val, aligned, coords=base_X, routing_kwargs=routing_kwargs)
        rest = sorted((v for v in range(aligned.vertices.shape[0])
                       if v not in located and aligned.repairable[v]),
                      key=lambda v: (-scores.get(v, 0), v))
        order = located + [v for v in rest if scores.get(v, 0) > 0]
        rep["retry_vertices"] = order
        rep["failure_located_vertices"] = located
        for v in order:
            for dlt in cfg.length_search_steps:
                if dlt == 0 or n_calls >= cfg.max_routing_attempts:
                    continue
                Xv = _length_variants(X0, aligned, cfg, v, dlt)
                if Xv is None:
                    continue
                if cfg.total_bp_budget is not None and Xv.shape[0] > cfg.total_bp_budget:
                    continue
                Xt = torch.as_tensor(Xv, dtype=torch.float32)
                o2, i2 = run_router(Xt, routing_kwargs, seed=seed, **_rs_kw(cfg))
                v2 = validate_routing(o2, i2)
                n_calls += 1
                attempts.append(dict(kind=f"length v{v} d{dlt:+d}", n_bps_in=int(Xt.shape[0]), **v2,
                                     **{k: i2[k] for k in ("exception", "stage", "warnings", "runtime_s")},
                                     **{k: i2[k] for k in ("staple_resplit", "xover_cleanup", "ss_rematch") if k in i2}))
                if v2["success"]:
                    best = (o2, Xt, v2)
                    break
            if best[2]["success"] or n_calls >= cfg.max_routing_attempts:
                break
    rep["attempts"] = attempts
    rep["routing_calls"] = len(attempts)
    rep["validation"] = best[2]
    # final routed geometry: which attempt produced it, and whether it differs from the input
    final = next((a["kind"] for a in attempts if a.get("success")), attempts[0]["kind"])
    rep["final_geometry"] = final
    rep["geometry_changed"] = bool(best[1].shape != tuple(coords.shape) or
                                   not np.array_equal(_to_numpy(best[1]), _to_numpy(coords)))
    return best[0], best[1], rep
