#!/usr/bin/env python3
"""Walking travel times carreau -> facility for one département (OSM + pandana).

* walking network from the département extract (pyrosm, network_type=walking),
  projected to EPSG:3035; nodes of tiny components (< 20 nodes reachable within
  1 km) are excluded from snapping;
* carreau centroids and facilities snap to the nearest kept node; snap distances
  are added on both ends (carreau dropped if snap > network.max_snap_m);
* candidate pairs = facilities within max(tau0) EUCLIDEAN distance (KD-tree),
  exact network distance by pandana shortest_path_lengths (contraction
  hierarchies) on unique node pairs, in chunks (bounded memory).
Output parquet: jid, fid, layer, tau_min (<= max tau0).
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree


def load_network(pbf):
    from pyrosm import OSM
    nodes, edges = OSM(pbf).get_network(network_type="walking", nodes=True)
    return nodes[["id", "lon", "lat"]], edges[["u", "v", "length"]]


# pandana warns for every pair of unconnected nodes (returned distance = 4294967 m,
# hence removed by the tau <= tau0 filter): expected warning, silenced.
warnings.filterwarnings("ignore", message="Unsigned integer: shortest path distance")


def travel_times(nodes, edges, pop, fac, cfg_net, chunk=2_000_000):
    """Exact network times for all pairs within max(tau0) Euclidean distance.

    Memory-light: candidate pairs come from a KD-tree prefilter (network
    distance >= Euclidean distance, so no pair is lost), then pandana
    shortest_path_lengths on unique node pairs, by chunks. (The former
    nearest_pois approach allocated nodes x 2K columns and was OOM-killed on
    dense départements.)
    """
    import pandana
    tr = Transformer.from_crs(4326, 3035, always_xy=True)
    nx, ny = tr.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nodes = nodes.assign(x=nx, y=ny).drop_duplicates("id").set_index("id")
    edges = edges[edges.u.isin(nodes.index) & edges.v.isin(nodes.index) & (edges.u != edges.v)]
    net = pandana.Network(nodes.x, nodes.y, edges.u, edges.v, edges[["length"]], twoway=True)
    low = set(net.low_connectivity_nodes(impedance=1000, count=20, imp_name="length"))
    keep = nodes[~nodes.index.isin(low)]
    tree = cKDTree(np.c_[keep.x.to_numpy(), keep.y.to_numpy()])
    keep_ids = keep.index.to_numpy()
    del nodes, edges

    speed = cfg_net["walk_speed_m_s"]
    tmax = max(cfg_net["tau0_min"])
    maxdist = speed * tmax * 60.0
    ds, ks = tree.query(np.c_[pop.x3035.to_numpy(), pop.y3035.to_numpy()])
    pop = pop.assign(node=keep_ids[ks], snap=ds)
    pop = pop[pop.snap <= cfg_net["max_snap_m"]].reset_index(drop=True)
    fx, fy = tr.transform(fac.lon.to_numpy(), fac.lat.to_numpy())
    df_, kf = tree.query(np.c_[fx, fy])
    fac = fac.assign(fx=fx, fy=fy, node=keep_ids[kf], fsnap=df_)
    fac = fac[fac.fsnap <= cfg_net["max_snap_m"]].reset_index(drop=True)

    out, stats = [], {}
    for layer, fl in fac.groupby("layer"):
        fl = fl.reset_index(drop=True)
        ftree = cKDTree(np.c_[fl.fx.to_numpy(), fl.fy.to_numpy()])
        cand = ftree.query_ball_point(np.c_[pop.x3035.to_numpy(), pop.y3035.to_numpy()], r=maxdist)
        nper = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(cand))
        if nper.sum() == 0:
            stats[layer] = {"pairs_prefilter": 0}
            continue
        jj = np.repeat(np.arange(len(pop)), nper)
        ff = np.concatenate([np.asarray(c, dtype=np.int64) for c in cand if len(c)])
        del cand
        a = pop.node.to_numpy()[jj]
        b = fl.node.to_numpy()[ff]
        pairs = pd.DataFrame({"a": a, "b": b})
        uniq = pairs.drop_duplicates().reset_index(drop=True)
        dist = np.empty(len(uniq))
        for c0 in range(0, len(uniq), chunk):
            u = uniq.iloc[c0:c0 + chunk]
            dist[c0:c0 + len(u)] = net.shortest_path_lengths(u.a.to_numpy(), u.b.to_numpy(), imp_name="length")
        uniq["d"] = dist
        d = pairs.merge(uniq, on=["a", "b"], how="left").d.to_numpy()
        tot = d + pop.snap.to_numpy()[jj] + fl.fsnap.to_numpy()[ff]
        tau = tot / speed / 60.0
        m = np.isfinite(tau) & (tau <= tmax)          # unreachable -> huge distance -> dropped
        out.append(pd.DataFrame({"jid": pop.jid.to_numpy()[jj][m].astype(np.int32),
                                 "fid": fl.fid.to_numpy()[ff][m].astype(np.int32),
                                 "layer": layer, "tau_min": tau[m].astype(np.float32)}))
        stats[layer] = {"pairs_prefilter": int(len(pairs)), "unique_node_pairs": int(len(uniq)),
                        "pairs_kept": int(m.sum())}
    od = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["jid", "fid", "layer", "tau_min"])
    stats["carreaux_snapped"] = int(len(pop))
    return od, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--dep", required=True)
    ap.add_argument("--access", default="all", help="facilities kept: all (filtering happens later)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    pop = pd.read_parquet(a.population)
    pop = pop[pop.dep == a.dep]
    fac = pd.read_parquet(a.facilities, columns=["fid", "layer", "lon", "lat"])
    buf = cfg["network"]["departement_buffer_m"] / 111_000
    fac = fac[fac.lon.between(pop.lon.min() - 1.5 * buf, pop.lon.max() + 1.5 * buf)
              & fac.lat.between(pop.lat.min() - buf, pop.lat.max() + buf)]
    nodes, edges = load_network(a.pbf)
    od, stats = travel_times(nodes, edges, pop, fac, cfg["network"])
    od.to_parquet(a.out, index=False)
    stats.update({"dep": a.dep, "carreaux": int(len(pop)), "pairs": int(len(od))})
    with open(a.log, "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    print(stats)


if __name__ == "__main__":
    main()
