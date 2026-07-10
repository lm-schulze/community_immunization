import numpy as np   
import networkx as nx
import collections
import random
import warnings

class RandomWalkError(Exception):
    pass

def BHD(G: nx.Graph, coverage: float=0.01, maxiter: int=100000, verbose: bool=False):
    """Finds nodes to immunize in a given network G via Bridge-Hub Detection (BHD) algorithm.

    Args:
        G (nx.Graph): Input contact network (should be undirected, unweighted)
        coverage (float, optional): Fraction of nodes to immunize. Defaults to 0.01.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10ß000.
        verbose (bool, optional): Whether to print verbose output for debugging. Defaults to False.

    Returns:
        set: Set of nodes selected for immunization.
    """
    immunized_nodes = set()
    iterations = 0
    nodes = list(G.nodes())

    neighbor_cache = {}  # lazily populated as nodes are visited
    def get_neighbors(n):
        if n not in neighbor_cache:
            neighbor_cache[n] = set(G.neighbors(n))
        return neighbor_cache[n]

    target_count = int(coverage * G.number_of_nodes())
    while len(immunized_nodes) < target_count and iterations < maxiter:
        iterations += 1
        RW = set() # nodes visited by the random walk
        F = set() # "friendship circle" (union of neighborhoods of visited nodes)

        # --- 2 steps to initialise the friendship circle ---
        # We just walk and accumulate F, no target checking yet
        node_curr = np.random.choice(nodes) # start at random node
        RW.add(node_curr) # add to RW set
        f_curr = get_neighbors(node_curr) # get neighbors of current node

        if verbose:
            print(f"\nIteration {iterations}: Starting node {node_curr}, neighbors {f_curr}")

        rw_options = f_curr - RW # options for self-avoiding continuation of the walk
        if not rw_options:
            if verbose:
                print(f"No options to continue from node {node_curr}. Restarting walk.")
            continue # if no options, start a new walk
        F.update(f_curr)  # add step 1 neighbourhood to F before moving to step 2
        node_curr = random.sample(list(rw_options), 1)[0]# move to next node
        RW.add(node_curr) # add to RW set
        f_curr = get_neighbors(node_curr) # get neighbors

        if verbose:
            print(f"Step 2: Moved to node {node_curr}, neighbors {f_curr}")

        rw_options = f_curr - RW # options for self-avoiding continuation of the walk
        if not rw_options:
            if verbose:
                print(f"No options to continue from node {node_curr}. Restarting walk.")
            continue # if no options, start a new walk
        F.update(f_curr)  # add step 2 neighbourhood to F before moving to step 3
        node_curr = random.sample(list(rw_options), 1)[0] # move to next node
        RW.add(node_curr) # add to RW set

        # --- Main walk: check for immunization targets ---
        # At this point F contains neighbourhoods of steps 1 and 2,
        # and node_curr is at step 3 — ready for the first check
        restart = False
        while len(RW) < 20 and not restart: # set max walk length to 20 as suggested in the paper
            f_curr = get_neighbors(node_curr) # get neighbors of current node
    
            # Compare f_curr against F_{t-1} (F does NOT yet contain f_curr)
            candidate_nodes = f_curr.difference(F) # candidates are neighbours not yet in the circle F
            immunization_targets = {
                c for c in candidate_nodes # go through candidates
                if not get_neighbors(c).difference({node_curr}).intersection(F) # check if candidate links back to F
            }

            if verbose:
                print(f"At node {node_curr}, neighbors {f_curr}, friendship circle F {F}")
                print(f"Candidate nodes for immunization: {candidate_nodes}")
                print(f"Immunization targets (candidates with no links back to F): {immunization_targets}")

            if immunization_targets: # if we found targets
                immunized_nodes.add(node_curr) # immunize current node
                # do a quick additional check to see if the target_count has been reached
                # just so we don't overshoot and have a fair comparison with the other algorithms
                if len(immunized_nodes) >= target_count:
                    if verbose:
                        print(f"Reached target immunization count with node {node_curr}. Ending algorithm.")
                    break
                immunized_nodes.add(random.sample(list(immunization_targets), 1)[0]) # immunize random neighbour that doesn't link back
                restart = True # we found a target, so we restart the walk
                if verbose:
                    print(f"Immunized node {node_curr} and target {immunization_targets}. Restarting walk.")
            else:
                if verbose:
                    print(f"No immunization targets found at node {node_curr}. Continuing walk.")
                # Only now add f_curr to F, then advance the walk
                rw_options = f_curr - RW # check for self-avoiding walk options
                if not rw_options:
                    restart = True # if no options, we restart the walk
                    if verbose:
                        print(f"No options to continue from node {node_curr}. Restarting walk.")
                else:
                    F.update(f_curr)  # update circle with current neighbors before moving to next node
                    node_curr = random.sample(list(rw_options), 1)[0] # move to next node
                    RW.add(node_curr) # add to RW set

    if iterations >= maxiter:
        warnings.warn(f"\nReached maximum iterations ({maxiter}) without immunizing the desired fraction of nodes. Immunized nodes: {immunized_nodes}")
    else:
        if verbose:
            print(f"\nFinished after {iterations} iterations. Immunized nodes: {immunized_nodes}")  
    return immunized_nodes


def ACQ(G: nx.Graph, coverage=0.01, n_acq=2):
    """Finds nodes to immunize in a given network G via Acquaintance immunization algorithm (ACQ). 

    Args:
        G (networkx.Graph): Input contact network (should be undirected, unweighted)
        coverage (float, optional): Fraction of nodes to immunize. Defaults to 0.01.
        n_acq (int, optional): Number of times a node needs to be found as acquaintance for it to be immunized. Defaults to 2.

    Returns:
        set: Set of nodes selected for immunization.
    """
    nodes = list(G.nodes())
    acquaintance_counts = {node: 0 for node in nodes} # count how many times each node is chosen as an acquaintance

    immunized_nodes = set()
    target_count = int(coverage * G.number_of_nodes())

    while len(immunized_nodes) < target_count:
        node = np.random.choice(nodes)
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue # skip isolated nodes
        acquaintance = np.random.choice(neighbors)
        acquaintance_counts[acquaintance] += 1
        if acquaintance_counts[acquaintance] >= n_acq and acquaintance not in immunized_nodes:
            immunized_nodes.add(acquaintance) # immunize the acquaintance if it has been chosen as an acquaintance at least n times

    return immunized_nodes


def BNI_LI(G:nx.Graph, coverage:float=0.01, rw_reps:float=0.05, L:int=20, doLocalProbing=False, verbose:bool=False, maxiter:int=10000):
    """Finds nodes to immunize in a given network G via Bridge-Neighbourhood Immunization with Local Information (BNI-LI) algorithm.
    Starts each trial at a new, randomly selected node.    

    Args:
        G (networkx.Graph): Input contact network (should be undirected, unweighted)
        rw_reps (float, optional): Number of random walk trials as a fraction of number of nodes. Note that 2x this upper bounds the coverage. Defaults to 0.05.
        coverage (float, optional): Fraction of nodes to immunize. Defaults to 0.01.
        L (int, optional): Upper limit for random walk length (reset for each trial). Defaults to 20.
        doLocalProbing (bool, optional): Whether to do additional local probing around bridge-hub candidates. Defaults to False.
        verbose (bool, optional): Whether to print verbose output for debugging. Defaults to False.
        maxiter (int, optional): Maximum number of (re)initialisation attempts as safeguard. Defaults to 1000.

    Returns:
        set: Set of nodes selected for immunization.
    """
    if coverage >= rw_reps:
        warnings.warn(f"Maximum number of expected immunization candidates {rw_reps * G.number_of_nodes()} is below desired coverage {int(coverage * G.number_of_nodes())}! Automatically setting rw_reps to coverage. Any missing slots will be filled randomly.")
        rw_reps = 2*coverage # just to be safe
    R = int(rw_reps * G.number_of_nodes()) # number of random walk repetitions per iteration

    immunized_nodes = set() # nodes that will be immunized
    nodes = list(G.nodes())
    target_candidates = set() # immunization target (bridge-hub) candidates that will later be ranked
    if doLocalProbing:
        hub_candidates = set() # hub candidates that will be ranked and immunized

    neighbor_cache = {}  # lazily populated as nodes are visited
    def get_neighbors(n):
        if n not in neighbor_cache:
            neighbor_cache[n] = set(G.neighbors(n))
        return neighbor_cache[n]
    
    def find_hub(v0): 
        hubRW = collections.defaultdict(int) # for counting node visits
        hubRW[v0] = 1 # add starting node
        v_curr = v0
        if verbose:
            print(f"Start walking for hub finding at node {v_curr}.")
        while max(hubRW.values()) < 2:
            nbs = list(get_neighbors(v_curr)) # get neighbours
            if not nbs:  # should be impossible to happen but juuuuuust in case 
                break
            v_curr = random.choice(nbs) # RW step
            hubRW[v_curr] += 1 # add count to visits dict
        if verbose:
            print("Visited nodes + visit counts:\n", hubRW)
        return max(hubRW, key=hubRW.get)
    
    def run_init(RW):
        walkSuccessful = False
        iterations = 0
        while not walkSuccessful and iterations < maxiter:
            iterations += 1
            RW_init = RW.copy() # (re)start from prior RW
            F = set() # initialise "friendship circle" (union of neighborhoods of visited nodes)
            node_curr = np.random.choice(nodes) # start at random node
            RW_init.add(node_curr) # add to RW set
            f_curr = get_neighbors(node_curr) # get neighbors of current node

            if verbose:
                print(f"\nStarting node {node_curr}, neighbors {f_curr}")

            rw_options = f_curr - RW_init # options for self-avoiding continuation of the walk
            if not rw_options:
                if verbose:
                    print(f"No options to continue from node {node_curr}. Restarting walk.")
                continue # if no options, start a new walk
            F.update(f_curr)  # add step 1 neighbourhood to F before moving to step 2
            node_curr = random.sample(list(rw_options), 1)[0] # move to next node
            RW_init.add(node_curr) # add to RW set
            f_curr = get_neighbors(node_curr) # get neighbors

            if verbose:
                print(f"Step 2: Moved to node {node_curr}, neighbors {f_curr}")

            rw_options = f_curr - RW_init # options for self-avoiding continuation of the walk
            if not rw_options:
                if verbose:
                    print(f"No options to continue from node {node_curr}. Restarting walk.")
                continue # if no options, start a new walk
            F.update(f_curr)  # add step 2 neighbourhood to F before moving to step 3
            walkSuccessful=True

        if iterations >= maxiter:
            warnings.warn(f"\nReached maximum iterations ({maxiter}) without successfully completing the walk (re)initialisation.")
            return False, node_curr, set(), set(), set()

        return walkSuccessful, node_curr, rw_options, F, RW_init  # return current node and built friendship circle

    for trial in range(R):   
        RW = set() # start with an empty set of RW steps at each trial
        # I've interpreted "independent trials" as the self-avoiding rw starting from scratch each trial     
        
        # --- Initialisation phase: 2 RW steps + building friendship circle ---
        initSuccessful, node_curr, rw_options, F, RW = run_init(RW)
        if not initSuccessful: # if we fail at this stage already, it's super bad
            # and we should check the network structure
            raise RandomWalkError("\nFailed to initialize self-avoiding random walk on given network.")
        # At this point F contains neighbourhoods of steps 1 and 2

        # --- Main walk: check for immunization targets ---
        # we're setting a constant max. length L for the Random walk, to avoid any issues.
        # trial is done when either bridge hub + hub candidate have been found or max walk length L is reached.
        while len(RW) < L: # while random walk has made less than L steps
            node_curr = random.choice(list(rw_options)) # move to next node
            RW.add(node_curr) # add to RW set
            f_curr = get_neighbors(node_curr) # get neighbors of current node
            f_diff = f_curr.difference(F) # neighbours of current node that are not yet in F
            # get all nodes from f_diff that link back to F
            backlinks = {c for c in f_diff # go through candidates
                    if get_neighbors(c).difference({node_curr}).intersection(F) # check if c links back to F
                }
            count = len(backlinks) # number of candidates that link back to F
            if count > len(f_diff) / 2: # if more half of the candidates link back
                F.update(backlinks)
                if verbose:
                    print(f"Node {node_curr} at trial {trial} has {count}/{len(f_diff)} backlinks to F. No candidate, continuing walk.")
            else:
                target_candidates.add(node_curr) # add current node to target candidates
                if verbose:
                    print(f"Node {node_curr} at trial {trial} added to target candidate pool with unconnected neighbour ratio({count}/{len(f_diff)}).")
                    if doLocalProbing:
                            print(f"Starting local probing around candidate node {node_curr}.")
                if doLocalProbing: # probe locally to identify hub candidates
                    # find hub via rw from node_curr and select first node visited twice as hub cndidate
                    hub_candidate = find_hub(node_curr)
                    hub_candidates.add(hub_candidate) # add to hub candidate pool
                    if verbose:
                        print(f"Node {hub_candidate} at trial {trial} added to hub candidate pool after local probing.")
                break # if candidates have been found, break out of while loop to move on with the next trial & restart rw
            
            # if no candidate has been found this step, check if self-avoiding walk can continue
            rw_options = f_curr - RW # options for self-avoiding continuation of the walk
            if not rw_options: # if we hit a dead end: warn & teleport
                warnings.warn("No options to continue the walk. Teleporting to different node and rebuilding friendship circle.")
                if verbose:
                    print(f"Random walk cannot continue from node {node_curr}. Teleporting to random node.")
                initSuccessful, node_curr, rw_options, F, RW = run_init(RW) # teleport to random node, redo the initialisation steps of friendship circle, but keep the RW steps
                if verbose:
                    print(f"Teleported to node {node_curr} and reeinitialised friendship circle. Init successful: {initSuccessful}")
                if not initSuccessful: # if we fail the reinitialisation here it's not great, but we still have done some walking we can work with
                    # so let's warn and stop the trials, but not throw an error.
                    warnings.warn(f"\n Avoiding dead ends via teleportation failed. Unable to complete the walk.")
                    break 

        if verbose:
            if len(RW) >= L:
                print(f"Trial {trial} finished because maximum random walk length was reached.")

    
    # rank target candidates by degree and immunize top fraction
    if doLocalProbing:
        target_candidates = target_candidates | hub_candidates # joint pool of bridge-hub and hub candidates
    
    if verbose:
        print(f"Finished all {R} trials! {len(target_candidates)} immunization candidates found.")

    target_candidates = list(target_candidates)
    target_candidates.sort(key=lambda n: G.degree(n), reverse=True) # sort candidates by degree, descending
    target_count = int(coverage * G.number_of_nodes())
    if target_count > len(target_candidates):
        warnings.warn(f" Not enough target candidates ({len(target_candidates)}) to immunize the desired fraction ({coverage}). Filling remaining slots randomly.")
        additional_random = np.random.choice(list(set(nodes) - set(target_candidates)), target_count - len(target_candidates), replace=False)
        immunized_nodes.update(additional_random) # add random nodes to immunized set if we don't have enough candidates
        immunized_nodes.update(target_candidates) # immunize all candidates if we don't have enough to fill the fraction
    
    else:
        immunized_nodes.update(target_candidates[:target_count]) # immunize top fraction

    return immunized_nodes


#  No immunization (baseline) 
def no_immunization(G, coverage=0.1):
    return set()


# random immunization
def random_immunization(G, coverage=0.1):
    """Randomly select the desired fraction of nodes to be immunized.

    Args:
        G (networkx.Graph): Input contact network
        coverage (float, optional): Fraction of nodes to immunize. Defaults to 0.1.

    Returns:
        list: list of nodes to be immunized.
    """
    nodes = list(G.nodes())
    target_count = int(coverage * G.number_of_nodes())
    immunized_nodes = set(np.random.choice(nodes, size=target_count, replace=False))
    return immunized_nodes


# Degree-based (deterministic) 
def degree_immunization(G, coverage):
    """Select nodes with highest degrees for immunization.

    Args:
        G (networkx.Graph): Input contact network
        coverage (float): Fraction of nodes to immunize

    Returns:
        set: set of nodes to be immunized.
    """
    n_vaccinate = int(coverage * len(G.nodes()))
    ranked = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    return set(ranked[:n_vaccinate])


# ── 5c. Community Bridge Finder — CBF (stochastic) ────────
#
#  implementation of the algorithm described in the paper
#  (Salathé & Jones 2010, Methods section + Figure 7):
#
#  1. Pick a random starting node v_0.
#  2. Follow a random path (never revisiting within the same walk).
#  3. At every node v_i (i >= 2):
#       Check back-connections of v_i to ALL previously visited nodes.
#       - If MORE than one back-connection → still inside a dense cluster,
#         keep walking.
#       - If EXACTLY one back-connection (only to v_{i-1}) →
#         v_{i-1} is a POTENTIAL target (candidate community bridge).
#         Additional check: pick 2 random neighbours of v_i
#         (other than v_{i-1}) and look for connections to previously
#         visited nodes.
#           · If such connections exist → dismiss v_{i-1}, continue walk.
#           · If no such connections → IMMUNIZE v_{i-1}.
#  4. Reset walk after each immunization (or if walk length > MAX_WALK).
#  5. Safety net: any node visited >= k times across all walks
#     is automatically immunized.
# ──────────────────────────────────────────────────────────
def cbf_immunization(G, coverage, max_walk=10, k=2):
    """
    Community Bridge Finder algorithm (Salathé & Jones).

    Parameters
    ----------
    G        : NetworkX graph
    coverage : fraction of nodes to immunize  (0 < coverage < 1)
    max_walk : max path length before resetting          (paper uses 10)
    k        : auto-immunize node visited >= k times     (paper uses 2)
    """

    n_vaccinate  = int(coverage * len(G.nodes()))
    nodes        = list(G.nodes())
    immunized    = set()
    visit_count  = collections.Counter()   # tracks visits across all walks

    while len(immunized) < n_vaccinate:

        # ── start a new random walk ────────────────────────
        v0 = random.choice(nodes)
        path = [v0]                         # ordered list of visited nodes
        visited_set = {v0}                  # fast membership check
        visit_count[v0] += 1

        # auto-immunize if visited >= k times
        if visit_count[v0] >= k and v0 not in immunized:
            immunized.add(v0)
            if len(immunized) >= n_vaccinate:
                break

        walk_step = 0

        while walk_step < max_walk and len(immunized) < n_vaccinate:

            current = path[-1]

            # ── choose next unvisited neighbour ───────────
            unvisited_nbs = [nb for nb in G.neighbors(current)
                             if nb not in visited_set]
            if not unvisited_nbs:
                break                       # dead end → reset walk

            next_node = random.choice(unvisited_nbs)
            path.append(next_node)
            visited_set.add(next_node)
            visit_count[next_node] += 1

            # auto-immunize safety net
            if visit_count[next_node] >= k and next_node not in immunized:
                immunized.add(next_node)
                if len(immunized) >= n_vaccinate:
                    break

            i = len(path) - 1              # index of next_node in path

            # ── check only applies from i >= 2 ────────────
            if i < 2:
                walk_step += 1
                continue

            # ── count back-connections of path[i] to ALL
            #    previously visited nodes (path[0..i-1]) ───
            prev_nodes = set(path[:i])     # all nodes before current
            back_connections = [pv for pv in prev_nodes
                                if G.has_edge(next_node, pv)]

            # more than one back-connection → still in dense cluster
            if len(back_connections) > 1:
                walk_step += 1
                continue

            # exactly one back-connection (to path[i-1]) →
            # path[i-1] is a POTENTIAL target
            potential_target = path[i - 1]

            # ── additional verification check ─────────────
            # pick up to 2 random neighbours of next_node
            # (other than potential_target) and check if
            # they connect back to any previously visited node
            other_nbs = [nb for nb in G.neighbors(next_node)
                         if nb != potential_target and nb not in visited_set]
            random.shuffle(other_nbs)
            check_nbs = other_nbs[:2]

            back_found = any(
                any(G.has_edge(cn, pv) for pv in prev_nodes)
                for cn in check_nbs
            )

            if back_found:
                # dismiss — still in the same cluster, continue walk
                walk_step += 1
                continue
            else:
                # ✔ confirmed community bridge → immunize
                if potential_target not in immunized:
                    immunized.add(potential_target)

                # reset — discard all walk information
                break

            walk_step += 1

    return immunized