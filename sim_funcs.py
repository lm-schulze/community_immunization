# library imports
import numpy as np   
import networkx as nx
import random
import EoN
import dataclasses as dc
import pyarrow as pa
import pyarrow.parquet as pq
import joblib as jl
import glob
import json
import os
import warnings
import time
from tqdm.auto import tqdm
import immunization_funcs as imf # this is where the immunization strategies are implemented, see immunization_funcs.py


immunization_funcs = {  # dict of immunization functions
    'None': imf.no_immunization,
    'Random': imf.random_immunization,
    'Degree': imf.degree_immunization,
    'ACQ': imf.ACQ,
    'CBF': imf.cbf_immunization,
    'BHD': imf.BHD,
    #'BNI-LI': imf.BNI_LI,
    #'BNI-LI-local': lambda G, coverage: imf.BNI_LI(G, coverage, doLocalProbing=True),
    #'BNI-LI-teleport': imf.BNI_LI_with_teleport,
    #'BNI-LI-teleport-local': lambda G, coverage: imf.BNI_LI_with_teleport(G, coverage, doLocalProbing=True),
    'BNI-LI-random-trial': imf.BNI_LI_with_random_restarts,
    'BNI-LI-random-trial-local': lambda G, coverage: imf.BNI_LI_with_random_restarts(G, coverage, doLocalProbing=True),
    } 


'''
immunization_funcs = {  # dict of immunization functions
    'None': imf.no_immunization,
    'Random': imf.random_immunization,
    'Degree': imf.degree_immunization,
    'ACQ': imf.ACQ,
    'CBF': imf.cbf_immunization,
    'BHD': imf.BHD,
    'BNI-LI': imf.BNI_LI
    } 
'''


'''
immunization_funcs = {  # dict of immunization functions
    'BNI-LI': imf.BNI_LI,
    'BNI-LI-local': lambda G, coverage: imf.BNI_LI(G, coverage, doLocalProbing=True),
    'BNI-LI-teleport': imf.BNI_LI_with_teleport,
    'BNI-LI-teleport-local': lambda G, coverage: imf.BNI_LI_with_teleport(G, coverage, doLocalProbing=True),
    'BNI-LI-random-trial': imf.BNI_LI_with_random_restarts,
    'BNI-LI-random-trial-local': lambda G, coverage: imf.BNI_LI_with_random_restarts(G, coverage, doLocalProbing=True),
    } 
'''

# schema for saving simulation results
schema = pa.schema([
    # simulation params
    ("rewire_steps",       pa.int32()),
    ("network_rep",        pa.int32()),
    ("modularity",         pa.float32()),
    ("algorithm",          pa.string()),
    ("coverage",           pa.float32()),
    ("imm_rep",            pa.int32()),
    ("beta",               pa.float32()),
    ("gamma",              pa.float32()),
    ("sir_rep",            pa.int32()),
    # result metrics
    ("final_attack_ratio", pa.float32()),
    ("peak_prevalence",    pa.float32()),
    ("duration",           pa.float32()),
])


# this might be useless but oh well
@dc.dataclass 
class SIRConfig: # all the SIR simulation parameters
    beta: float = 0.08 # transmission rate
    gamma: float = 0.2 # recovery rate
    tmax: float = 1000.0 # max. timesteps (for EoN)
    n_reps: int = 100 # number of repetitions for each simulation 


def generate_community_network(m = 50, n_sw=40, rewire_steps=0, verbose=False):
    """Generates network with community structure as described by Salathé-Jones 2010.

    Args:
        m (int, optional): Number of communities. Defaults to 50.
        n_sw (int, optional): Community size. Defaults to 40.
        rewire_steps (int, optional): Number of rewiring steps to increase modularity. Defaults to 0.
        verbose (bool, optional): Whether to print additional info for debugging. Defaults to False.

    Returns:
        Tuple(networkx.Graph, list[float], float: Tupel of output graph G, community partition of G, modularity of G.
    """
    n_tot = m*n_sw # total number of nodes in the graph
    k = int(0.2*n_sw) # number of inter-community edges per node
    sw_graphs = [nx.watts_strogatz_graph(n_sw, k=k, p=0.0) for _ in range(m)]    
    # create list of community labels for each node
    communities_partition = [{i*n_sw + j for j in range(n_sw)} for i in range(m)]  # sets of nodes
    G = nx.disjoint_union_all(sw_graphs)
    # add n_tot random inter-community edges
    added_edges = 0
    while added_edges < n_tot:
        c1, c2 = np.random.choice(m, size=2, replace=False) # get 2 random communities
        n1 = np.random.choice(n_sw) + c1*n_sw # get a random node from community 1
        n2 = np.random.choice(n_sw) + c2*n_sw # get a random node from community 2
        if not G.has_edge(n1, n2): # avoid multi-edges
            G.add_edge(n1, n2) # add an inter-community edge
            added_edges += 1
    # compute modularity
    if verbose:
        print("Modularity before rewiring:", nx.algorithms.community.quality.modularity(G, communities_partition))
        print("Average degree:", np.mean([d for n, d in G.degree()]))
    # rewire random between-community edges to within-community edges, to increase modularity
    betw_edges = set((u, v) for u, v in G.edges() if u // n_sw != v // n_sw)
    rewired = 0
    while rewired < rewire_steps:
        if not betw_edges:
            break
        # pick a random between-community edge
        edge = random.choice(list(betw_edges))    # pick a random endpoint to keep (u or v), and rewire the other one
        u, v = edge
        if np.random.rand() < 0.5:
            u, v = v, u # swap so that u is the one we keep
        # rewire v to a random node in the same community as u
        community = u // n_sw
        new_v = np.random.choice(n_sw) + community*n_sw
        # only add the new edge if it doesn't already exist, to avoid creating multi-edges
        # and also avoid self-loops
        if new_v != u and not G.has_edge(u, new_v):
            G.remove_edge(*edge)
            betw_edges.discard(edge)
            G.add_edge(u, new_v)
            rewired += 1

    modularity_final = nx.algorithms.community.quality.modularity(G, communities_partition)
    if verbose:
        print("Modularity after rewiring:", modularity_final)

    return G, communities_partition, modularity_final


# the SIR simulation repetitions are independent from each other
# and could therefore run in parallel
def run_sir_batch(G, immunized_nodes, sir_config):
    """Runs the n_rep repetitions of the specified SIR parameter configuration,
    as described in sir_config, and save the desired result metrics

    Args:
        G (networkx.Graph): input contact network
        immunized_nodes (set): set of nodes that are initially immunized
        sir_config (SIRConfig): Contains SIR simulation parameters
    """
    n_nodes = G.number_of_nodes()

    def _single_rep(rep):
        res = EoN.fast_SIR(G, sir_config.beta, sir_config.gamma,
                           tmax=sir_config.tmax,
                           initial_recovereds=list(immunized_nodes),
                           return_full_data=True)
        attack_ratio = (res.R()[-1] - len(immunized_nodes)) / n_nodes
        return {
            'sir_rep':            rep,
            'final_attack_ratio': attack_ratio,
            'peak_prevalence':    max(res.I()) / n_nodes,
            'duration':           res.t()[-1],
        }

    return jl.Parallel(n_jobs=-1)(
        jl.delayed(_single_rep)(rep) for rep in range(sir_config.n_reps)
    )

# --- Full simulations for parameter sweeps -------------------------
def run_modularity_sweep(
    rewire_steps_list: list[int],
    sir_cfg: SIRConfig=SIRConfig(beta=0.08, gamma=0.2),
    coverage: float=0.1,
    n_network_reps: int=1,
    output_path: str='results/modularity_sweep.parquet',
    ):
    """Runs network creation, immunization & SIR simulations for different numbers of rewire steps at network generation
    corresponding to different modularity settings for the contact network. Writes results to a parquet file.

    Args:
        rewire_steps_list (list[int]): Rewire steps settings to explore.
        sir_cfg (SIRConfig, optional): Configuration for SIR simulation. Defaults to SIRConfig(beta=0.08, gamma=0.2).
        coverage (float, optional): Immunization coverage. Defaults to 0.1.
        n_network_reps (int, optional): Number of network instances to test per network configuration (in terms of rewire steps). Defaults to 1.
        output_path (str, optional): Where to save the resulting .parquet file. Defaults to 'results/modularity_sweep.parquet'.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = pq.ParquetWriter(output_path, schema)
    
    for rewire_steps in tqdm(rewire_steps_list, desc='modularity'):
        for net_rep in tqdm(range(n_network_reps), desc='network rep', leave=False):
            G, partition, Q = generate_community_network(rewire_steps=rewire_steps)
            #adj = build_adj(G)
            N = G.number_of_nodes()
            
            # immunize with all algorithms
            immunized = {}
            for algo_name, algo_fn in immunization_funcs.items():
                immunized[algo_name] = algo_fn(G, coverage=coverage)
            
            # SIR batch (parallelised across algorithms)
            rows = []
            algo_results = jl.Parallel(n_jobs=-1)(
                jl.delayed(run_sir_batch)(G, immunized[algo], sir_cfg)
                for algo in immunization_funcs
            )
            for algo_name, batch in zip(immunization_funcs, algo_results):
                for r in batch:
                    rows.append({
                        'rewire_steps': rewire_steps,
                        'network_rep': net_rep,
                        'modularity': Q,
                        'algorithm': algo_name,
                        'coverage': coverage,
                        'beta': sir_cfg.beta,
                        'gamma': sir_cfg.gamma,
                        **r,
                    })
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))

    writer.close()


def run_coverage_sweep(
    coverage_list: list[float],
    sir_cfg: SIRConfig=SIRConfig(),
    rewire_steps=0,
    n_network_reps: int=1,
    output_path: str='results/coverage_sweep.parquet',):
    """Runs network creation, immunization & SIR simulations for different immunization coverages at the immunization stage.
    Writes results to a parquet file.

    Args:
        coverage_list (list[float]): List of immunization coverages to explore.
        sir_cfg (SIRConfig, optional):Configuration for SIR simulation. Defaults to SIRConfig().
        rewire_steps (int, optional): Number of rewire steps to perform at network generation; determines modularity. Defaults to 0.
        n_network_reps (int, optional): Number of network instances to test per network configuration (in terms of rewire steps). Defaults to 1.
        output_path (str, optional): Where to save the resulting .parquet file. Defaults to 'results/coverage_sweep.parquet'.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = pq.ParquetWriter(output_path, schema)
    
    for net_rep in tqdm(range(n_network_reps), desc='network rep'):
        G, partition, Q = generate_community_network(rewire_steps=rewire_steps)

        for coverage in tqdm(coverage_list, desc='immunization coverage', leave=False):            
            # immunize with all algorithms
            immunized = {}
            for algo_name, algo_fn in immunization_funcs.items():
                immunized[algo_name] = algo_fn(G, coverage=coverage)
            
            # SIR batch (parallelised across algorithms)
            rows = []
            algo_results = jl.Parallel(n_jobs=-1)(
                jl.delayed(run_sir_batch)(G, immunized[algo], sir_cfg)
                for algo in immunization_funcs
            )
            for algo_name, batch in zip(immunization_funcs, algo_results):
                for r in batch:
                    rows.append({
                        'network_rep': net_rep,
                        'rewire_steps': rewire_steps,
                        'modularity': Q,
                        'algorithm': algo_name,
                        'coverage': coverage,
                        'beta': sir_cfg.beta,
                        'gamma': sir_cfg.gamma,
                        **r,
                    })
            
            # checkpoint: flush after every network replicate
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    
    writer.close()


def run_sir_params_sweep(
    beta_list: list[float],
    gamma_list: list[float],
    rewire_steps=0,
    coverage: float=0.1,
    n_network_reps: int=1,
    output_path: str='results/sir_params_sweep.parquet',):
    """Runs network creation, immunization & SIR simulations for different transmission rates beta and recovery rates gamma
     at the SIR simulation stage. Writes results to a parquet file.

    Args:
        beta_list (list[float]): List of transmission rates to explore in the SIR simulation
        gamma_list (list[float]): List of recovery rates to explore in the SIR simmulation
        rewire_steps (int, optional): Number of rewire steps to perform at network generation; determines modularity. Defaults to 0.
        coverage (float, optional): Immunization coverage. Defaults to 0.1.
        n_network_reps (int, optional): Number of network instances to test per network configuration (in terms of rewire steps). Defaults to 1.
        output_path (str, optional): Where to save the resulting .parquet file. Defaults to 'results/sir_params_sweep.parquet'.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = pq.ParquetWriter(output_path, schema)
    
    for net_rep in tqdm(range(n_network_reps), desc='network rep'):
        G, partition, Q = generate_community_network(rewire_steps=rewire_steps)
        
        # immunize with all algorithms
        immunized = {}
        for algo_name, algo_fn in immunization_funcs.items():
            immunized[algo_name] = algo_fn(G, coverage=coverage)

        for gamma in tqdm(gamma_list, desc='transmission rate', leave=False):
            for beta in tqdm(beta_list, desc='recovery rate', leave=False):
                sir_cfg = SIRConfig(beta=beta, gamma=gamma)
        
                # SIR batch (parallelised across algorithms)
                rows = []
                algo_results = jl.Parallel(n_jobs=-1)(
                    jl.delayed(run_sir_batch)(G, immunized[algo], sir_cfg)
                    for algo in immunization_funcs
                )
                for algo_name, batch in zip(immunization_funcs, algo_results):
                    for r in batch:
                        rows.append({
                            'network_rep': net_rep,
                            'rewire_steps': rewire_steps,
                            'modularity': Q,
                            'algorithm': algo_name,
                            'coverage': coverage,
                            'beta': sir_cfg.beta,
                            'gamma': sir_cfg.gamma,
                            **r,
                        })
                
                # checkpoint: flush after every network replicate
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    
    writer.close()


def run_gen_sweep_no_checkpoints(
    rewire_steps_list: list[int] = [0],
    coverage_list: list[float] = [0.1],
    beta_list: list[float] = [0.8],
    gamma_list: list[float] = [0.2],
    n_network_reps: int=1,
    n_sir_reps = 100,
    output_path: str='results/full_sweep.parquet',):
    """Runs network creation, immunization & SIR simulations for different rewire steps at network generation (corresponding to different modularity settings),
    different immunization coverages at the immunization step, and different transmission rates beta and recovery rates gamma at the SIR simulation stage.
    Writes results to a parquet file.

    Args:
        rewire_steps_list (list[int]): Rewire steps settings to explore (controls network modularity). Defaults to [0].
        coverage_list (list[float]): List of immunization coverages to explore. Defaults to [0.1].
        beta_list (list[float]): List of transmission rates to explore in the SIR simulation. Defaults to [0.08]
        gamma_list (list[float]): List of recovery rates to explore in the SIR simmulation. Defaluts to [0.2]
        n_network_reps (int, optional): Number of network instances to test per network configuration (in terms of rewire steps). Defaults to 1.
        output_path (str, optional): Where to save the resulting .parquet file.  Defaults to 'results/full_sweep.parquet'.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = pq.ParquetWriter(output_path, schema)
    
    for rewire_steps in tqdm(rewire_steps_list, desc='modularity'):
        for net_rep in tqdm(range(n_network_reps), desc='network rep', leave=False):
            # --- Network generation-------------------------------------------
            G, partition, Q = generate_community_network(rewire_steps=rewire_steps)
            
            for coverage in tqdm(coverage_list, desc='immunization coverage', leave=False):            
                # --- Immunization --------------------------------------------
                immunized = {}
                for algo_name, algo_fn in immunization_funcs.items():
                    immunized[algo_name] = algo_fn(G, coverage=coverage)

                # --- SIR simulation ------------------------------------------
                for gamma in tqdm(gamma_list, desc='transmission rate', leave=False):
                    for beta in tqdm(beta_list, desc='recovery rate', leave=False):
                        sir_cfg = SIRConfig(beta=beta, gamma=gamma, n_reps=n_sir_reps)
                
                        # SIR batch (parallelised across algorithms)
                        rows = []
                        # Parallelise across algorithms; each call is independent 
                        # n_reps SIR batch
                        #algo_results: list[list[dict]] = jl.Parallel(n_jobs=-1)(
                        #    jl.delayed(run_sir_batch)(G, immunized[algo], sir_cfg)
                        #    for algo in immunization_funcs
                        #)
                        algo_results = [
                            run_sir_batch(G, immunized[algo], sir_cfg)
                            for algo in immunization_funcs
                            ]
                        
                        for algo_name, batch in zip(immunization_funcs, algo_results):
                            for r in batch:
                                rows.append({
                                    'network_rep': net_rep,
                                    'rewire_steps': rewire_steps,
                                    'modularity': Q,
                                    'algorithm': algo_name,
                                    'coverage': coverage,
                                    'beta': sir_cfg.beta,
                                    'gamma': sir_cfg.gamma,
                                    **r,
                                })
                        
                        # checkpoint: flush after every network replicate
                        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            
    writer.close()



# SIMULATION WITH CHECKPOINTS
# Directory layout inside checkpoint_dir:
#
#   checkpoint_dir/
#   ├── manifest.json                           (tracks completed param combos)
#   ├── networks/
#   │   ├── G_rw{rw}_rep{rep}.joblib            (saves current NetworkX graph)
#   │   └── imm_rw{rw}_rep{rep}_cov{cov}.joblib (saves immunized-node dicts)
#   └── data/
#       └── rw{rw}_rep{rep}_cov{cov}_b{b}_g{g}.parquet  (saved intermediate result chunks)
#

# --- Setup directories & build paths ---------------------------------------

# networks subdirectory (where current network info will be saved)
def _net_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, 'networks')

# data subrdirectory (where checkpoint data will be saved)
def _data_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, 'data')

# setting them up
def _setup_checkpoint_dirs(checkpoint_dir: str) -> None:
    """Create all required sub-directories."""
    os.makedirs(_net_dir(checkpoint_dir),  exist_ok=True)
    os.makedirs(_data_dir(checkpoint_dir), exist_ok=True)

# builds the path for specific network described by rewire_steps and index of
# current repetition 
def _network_cache_path(checkpoint_dir: str,
                        rewire_steps: int, net_rep: int) -> str:
    return os.path.join(
        _net_dir(checkpoint_dir),
        f'G_rw{int(rewire_steps)}_rep{int(net_rep)}.joblib',
    )


def _imm_cache_path(checkpoint_dir: str,
                    rewire_steps: int, net_rep: int,
                    coverage: float, imm_rep: int) -> str:  # ← imm_rep added
    return os.path.join(
        _net_dir(checkpoint_dir),
        f'imm_rw{int(rewire_steps)}_rep{int(net_rep)}'
        f'_cov{float(coverage):.8f}_irep{int(imm_rep)}.joblib',  # ← irep in filename
    )


def _data_chunk_path(checkpoint_dir: str,
                     rewire_steps: int, net_rep: int,
                     coverage: float, imm_rep: int,           # ← imm_rep added
                     beta: float, gamma: float) -> str:
    fname = (
        f'rw{int(rewire_steps)}_rep{int(net_rep)}'
        f'_cov{float(coverage):.8f}_irep{int(imm_rep)}'       # ← irep in filename
        f'_b{float(beta):.8f}_g{float(gamma):.8f}.parquet'
    )
    return os.path.join(_data_dir(checkpoint_dir), fname)



# --- Manifest (keeping track of completion status) -------------------------

# manifest for keeping track of successfully completed combos, store them as JSON array
# 5-element lists: [[rewire_steps, net_rep, coverage, beta, gamma], ...]
def _manifest_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, 'manifest.json')

# load the manifest with completed combos
def _load_manifest(checkpoint_dir: str) -> set[tuple]:
    """Return set of completed (rewire_steps, net_rep, coverage, beta, gamma) tuples."""
    path = _manifest_path(checkpoint_dir)
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        raw = json.load(fh)
    # Reconstruct tuples; JSON preserves float values via Python's shortest-
    # repr encoding, so round-trip equality with the original list values holds.
    return {tuple(item) for item in raw}

# update with completed combo
def _update_manifest(checkpoint_dir: str, completed: set[tuple]) -> None:
    """Atomically persist the completed-combo set.

    Uses a write-then-rename pattern so a crash during the write never
    leaves a corrupt manifest on disk.
    """
    path = _manifest_path(checkpoint_dir)
    tmp  = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump([list(item) for item in completed], fh)
    os.replace(tmp, path)   # atomic on POSIX; best-effort on Windows

def _update_manifest_with_retries(checkpoint_dir: str, completed: set[tuple]) -> None:
    """Atomically persist the completed-combo set.

    Uses write-to-tmp + rename. On Windows, antivirus tools can briefly
    lock a freshly written file; the retry loop with back-off handles that.
    """
    path = _manifest_path(checkpoint_dir)
    tmp  = path + '.tmp'

    with open(tmp, 'w') as fh:
        json.dump([list(item) for item in completed], fh)

    for attempt in range(6):                        # ~3 s total wait max
        try:
            os.replace(tmp, path)
            return                                  # success — done
        except PermissionError:
            if attempt < 5:
                time.sleep(0.1 * 2 ** attempt)     # 0.1, 0.2, 0.4, 0.8, 1.6 s
            else:
                # Last resort: explicit delete + rename.
                # Not atomic, but safe here because the manifest is only
                # written by this process and we already have the new data
                # in `tmp`. A crash in this tiny window would leave `tmp`
                # on disk, which _load_manifest ignores (it reads `path`).
                try:
                    if os.path.exists(path):
                        os.remove(path)
                    os.rename(tmp, path)
                except OSError as e:
                    warnings.warn(
                        f"_update_manifest: could not replace manifest after "
                        f"6 attempts — {e}. The chunk was saved successfully; "
                        f"re-run with resume=True to repair the manifest."
                    )


# --- Network / immunization cache ------------------------------------------

# save the current network
def _save_network(G: nx.Graph, Q: float,
                  checkpoint_dir: str,
                  rewire_steps: int, net_rep: int) -> None:
    path = _network_cache_path(checkpoint_dir, rewire_steps, net_rep)
    jl.dump({'G': G, 'Q': Q}, path, compress=3)

# load the network
def _load_network(checkpoint_dir: str,
                  rewire_steps: int,
                  net_rep: int) -> tuple[nx.Graph | None, float | None]:
    path = _network_cache_path(checkpoint_dir, rewire_steps, net_rep)
    if not os.path.exists(path):
        return None, None
    data = jl.load(path)
    return data['G'], data['Q']

# save immunization dict containing set of immunized nodes for each
# algorithm (algo: set of immunized nodes), together with coverage and nw info

def _save_immunization(immunized: dict, checkpoint_dir: str,
                       rewire_steps: int, net_rep: int,
                       coverage: float, imm_rep: int) -> None:  # ← imm_rep added
    path = _imm_cache_path(checkpoint_dir, rewire_steps, net_rep, coverage, imm_rep)
    jl.dump(immunized, path, compress=3)


def _load_immunization(checkpoint_dir: str,
                       rewire_steps: int, net_rep: int,
                       coverage: float, imm_rep: int) -> dict | None:  # ← imm_rep added
    path = _imm_cache_path(checkpoint_dir, rewire_steps, net_rep, coverage, imm_rep)
    if not os.path.exists(path):
        return None
    return jl.load(path)


# merge all checkpoint chunks into one 
def merge_checkpoints(checkpoint_dir: str, output_path: str) -> None:
    """Concatenate all per-chunk parquet files into a single output table.

    Safe to call at any point (e.g. to inspect partial results). Can also
    be called manually after a completed run if the final merge step was
    interrupted.

    Args:
        checkpoint_dir: Root checkpoint directory.
        output_path:    Destination parquet file.
    """
    files = sorted(glob.glob(os.path.join(_data_dir(checkpoint_dir), '*.parquet')))
    if not files:
        print('merge_checkpoints: no chunk files found — nothing to merge.')
        return
    tables = [pq.read_table(f) for f in files]
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pq.write_table(pa.concat_tables(tables), output_path)
    print(f'merge_checkpoints: merged {len(files)} chunks → {output_path}')


# --- The full simulation  ---------------------------------------
# new version of generic sweep across a ton of parameters
# with checkpoints!
def run_gen_sweep(
        rewire_steps_list: list[int]   = [0],
        coverage_list:     list[float] = [0.1],
        beta_list:         list[float] = [0.8],
        gamma_list:        list[float] = [0.2],
        n_network_reps:    int         = 1,
        n_immunize_reps:   int         = 10,
        n_sir_reps:        int         = 100,
        output_path:       str         = 'results/full_sweep.parquet',
        checkpoint_dir:    str         = 'results/checkpoints',
        resume:            bool        = False,
) -> None:
    """Run network generation, immunization, and SIR simulation across a
    parameter grid, saving one parquet chunk per
    (rewire_steps, net_rep, coverage, beta, gamma) combination.

    Checkpointing
    -------------
    After every chunk is written a JSON manifest is updated atomically.
    If the process is killed and restarted with ``resume=True``, already-
    completed chunks are skipped and cached networks / immunization sets
    are reloaded from disk so results remain reproducible.

    Args:
        rewire_steps_list: Values of rewire_steps to sweep (controls modularity).
        coverage_list:     Immunization coverage fractions to sweep.
        beta_list:         SIR transmission rates to sweep.
        gamma_list:        SIR recovery rates to sweep.
        n_network_reps:    Independent network replicates per rewire_steps value.
        n_immunize_reps:   Independend immunization process repetitions per algorithm & coverage.
        n_sir_reps:        Independent SIR simulation repetitions per beta-gamma-combination.
        output_path:       Final merged parquet file.
        checkpoint_dir:    Root directory for checkpoint sub-directories.
        resume:            If True, skip already-completed combos via manifest.
    """
    # --- Setup directories & checkpoints -------------------------------------

    # setup the checkpoint directories
    _setup_checkpoint_dirs(checkpoint_dir)
    # check output directory & create if necessary
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # if resuming from checkpoints, load manifest of completed combinations
    completed: set[tuple] = _load_manifest(checkpoint_dir) if resume else set()
    if resume:
        # Total now includes n_immunize_reps as a multiplier
        total_combos = (len(rewire_steps_list) * n_network_reps
                        * len(coverage_list) * n_immunize_reps  # ← added
                        * len(beta_list) * len(gamma_list))
        print(f'[resume] {len(completed)} / {total_combos} combination(s) done - skipping.')

    # --- Start simulations ---------------------------------------------------
    # --- Network setup -------------------------------------------------------
    for rewire_steps in tqdm(rewire_steps_list, desc='rewire_steps'):
        for net_rep in tqdm(range(n_network_reps), desc='network_rep', leave=False):

            # Fast-path: skip this network instance if all its (coverage, beta, gamma)
            # combinations are already marked as completed in the manifest
            all_for_net = { # all parameter combinations for this nw instance
                (rewire_steps, net_rep, cov, b, g)
                for cov in coverage_list
                for irep in range(n_immunize_reps)         #
                for b   in beta_list
                for g   in gamma_list
            }
            if all_for_net.issubset(completed): # if all combos are completed, continue
                continue

            # Load the cached network if available 
            # (if resuming partially completed sims) 
            G, Q = _load_network(checkpoint_dir, rewire_steps, net_rep)
            if G is None:  # otherwise generate a new one and cache it
                G, _, Q = generate_community_network(rewire_steps=rewire_steps)
                _save_network(G, Q, checkpoint_dir, rewire_steps, net_rep)

            # --- Immunization ----------------------------------------------
            for coverage in tqdm(coverage_list, desc='coverage', leave=False):
                
                # Fast-path: all (imm_rep, beta, gamma) done for this coverage
                all_for_cov = {
                    (rewire_steps, net_rep, coverage, irep, b, g)  # ← irep added
                    for irep in range(n_immunize_reps)
                    for b    in beta_list
                    for g    in gamma_list
                }
                if all_for_cov.issubset(completed):
                    continue
                

                # --- Immunization repetition loop ----------------------
                # each rep independently re-runs all stochastic immunization
                # algorithms, producing a different set of immunized nodes
                # Deterministic algorithms (Degree, None) will produce the
                # same result each time, but are re-cached per-rep anyway
                # for consistency and to keep the downstream indexing uniform.
                for imm_rep in tqdm(range(n_immunize_reps), desc='imm_rep', leave=False):

                    # Fast-path: all (beta, gamma) done for this imm_rep
                    all_for_irep = {
                        (rewire_steps, net_rep, coverage, imm_rep, b, g)
                        for b in beta_list
                        for g in gamma_list
                    }
                    if all_for_irep.issubset(completed):
                        continue

                    # Load or compute immunized sets for this specific rep
                    immunized = _load_immunization(
                        checkpoint_dir, rewire_steps, net_rep, coverage, imm_rep)  # ← imm_rep
                    if immunized is None:
                        immunized = {
                            name: fn(G, coverage=coverage)
                            for name, fn in immunization_funcs.items()
                        }
                        _save_immunization(
                            immunized, checkpoint_dir,
                            rewire_steps, net_rep, coverage, imm_rep)  
                        
                    # --- SIR simulations -------------------------------------
                    for gamma in tqdm(gamma_list, desc='gamma', leave=False):
                        for beta in tqdm(beta_list, desc='beta', leave=False):

                            key = (rewire_steps, net_rep, coverage, imm_rep, beta, gamma)  # ← imm_rep
                            if key in completed:
                                continue

                            sir_cfg = SIRConfig(beta=beta, gamma=gamma, n_reps=n_sir_reps)

                            algo_results = [
                                run_sir_batch(G, immunized[algo], sir_cfg)
                                for algo in immunization_funcs
                            ]

                            rows = [
                                {
                                    'network_rep':  net_rep,
                                    'rewire_steps': rewire_steps,
                                    'modularity':   Q,
                                    'algorithm':    algo_name,
                                    'coverage':     coverage,
                                    'imm_rep':      imm_rep,
                                    'beta':         beta,
                                    'gamma':        gamma,
                                    **r,
                                }
                                for algo_name, batch in zip(immunization_funcs, algo_results)
                                for r in batch
                            ]

                            chunk_path = _data_chunk_path(
                                checkpoint_dir, rewire_steps, net_rep,
                                coverage, imm_rep, beta, gamma)   # ← imm_rep
                            pq.write_table(
                                pa.Table.from_pylist(rows, schema=schema),
                                chunk_path,
                            )
                            completed.add(key)
                            _update_manifest_with_retries(checkpoint_dir, completed)

    # Merge all chunks into the final output file 
    merge_checkpoints(checkpoint_dir, output_path)


# RESUME CONVENIENCE WRAPPER
def resume_gen_sweep(
        rewire_steps_list: list[int]   = [0],
        coverage_list:     list[float] = [0.1],
        beta_list:         list[float] = [0.8],
        gamma_list:        list[float] = [0.2],
        n_network_reps:    int         = 1,
        n_immunize_reps:   int         = 10,   # ← was missing entirely before
        n_sir_reps:        int         = 100,
        output_path:       str         = 'results/full_sweep.parquet',
        checkpoint_dir:    str         = 'results/checkpoints',
) -> None:
    """Resume an interrupted :func:`run_gen_sweep` from its checkpoint state.

    Call with the **identical** parameter space as the original call.
    The function reads the manifest in ``checkpoint_dir``, skips every
    (rewire_steps, net_rep, coverage, beta, gamma) combination that is
    already recorded as complete, and runs only the remaining work.

    Networks and immunization sets are restored from the cache in
    ``checkpoint_dir/networks/`` so results from resumed runs are
    consistent with those from the original run.

    Args:
        rewire_steps_list: Must match the original call.
        coverage_list:     Must match the original call.
        beta_list:         Must match the original call.
        gamma_list:        Must match the original call.
        n_network_reps:    Must match the original call.
        n_immunzie_reps:   Must match the original call.
        n_sir_reps:        Must match the original call.
        output_path:       Destination for the final merged parquet file.
        checkpoint_dir:    Must point to the same directory as the original call.
    """
    run_gen_sweep(
        rewire_steps_list=rewire_steps_list,
        coverage_list=coverage_list,
        beta_list=beta_list,
        gamma_list=gamma_list,
        n_network_reps=n_network_reps,
        n_immunize_reps=n_immunize_reps,   # ← was missing
        n_sir_reps=n_sir_reps,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )