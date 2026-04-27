import numpy as np   
import matplotlib.pyplot as plt
import networkx as nx
import EoN

def BHD(G: nx.Graph, immunized_fraction: float=0.01, maxiter: int=1000, verbose: bool=False):
    """Finds nodes to immunize in a given network G via Bridge-Hub Detection (BHD) algorithm.

    Args:
        G (nx.Graph): Input contact network (should be undirected, unweighted)
        immunized_fraction (float, optional): Fraction of nodes to immunize. Defaults to 0.01.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
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

    target_count = int(immunized_fraction * G.number_of_nodes())
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
        node_curr = np.random.choice(list(rw_options)) # move to next node
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
        node_curr = np.random.choice(list(rw_options)) # move to next node
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
                immunized_nodes.add(np.random.choice(list(immunization_targets))) # immunize random neighbour that doesn't link back
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
                    node_curr = np.random.choice(list(rw_options)) # move to next node
                    RW.add(node_curr) # add to RW set

    if verbose:
        if iterations >= maxiter:
            print(f"\nReached maximum iterations ({maxiter}) without immunizing the desired fraction of nodes. Immunized nodes: {immunized_nodes}")
        else:
            print(f"\nFinished after {iterations} iterations. Immunized nodes: {immunized_nodes}")  
    return immunized_nodes


def ACQ(G: nx.Graph, immunized_fraction=0.01, n_acq=2):
    """Finds nodes to immunize in a given network G via Acquaintance immunization algorithm (ACQ). 

    Args:
        G (networkx.Graph): Input contact network (should be undirected, unweighted)
        immunized_fraction (float, optional): Fraction of nodes to immunize. Defaults to 0.01.
        n_acq (int, optional): Number of times a node needs to be found as acquaintance for it to be immunized. Defaults to 2.

    Returns:
        set: Set of nodes selected for immunization.
    """
    nodes = list(G.nodes())
    acquaintance_counts = {node: 0 for node in nodes} # count how many times each node is chosen as an acquaintance

    immunized_nodes = set()
    target_count = int(immunized_fraction * G.number_of_nodes())

    while len(immunized_nodes) < target_count:
        node = np.random.choice(nodes)
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue # skip isolated nodes
        acquaintance = np.random.choice(neighbors)
        acquaintance_counts[acquaintance] += 1
        if acquaintance_counts[acquaintance] >= n_acq:
            immunized_nodes.add(acquaintance) # immunize the acquaintance if it has been chosen as an acquaintance at least n times

    return immunized_nodes


def BNI_LI(G, rw_reps= 0.05, immunized_fraction=0.01, verbose=False, maxiter=1000):
    """Finds nodes to immunize in a given network G via Bridge-Neighbourhood Immunization with Local Information (BNI-LI) algorithm.    

    Args:
        G (networkx.Graph): Input contact network (should be undirected, unweighted)
        rw_reps (float, optional): Number of random walk trials as a fraction of number of nodes. Defaults to 0.05.
        immunized_fraction (float, optional): Fraction of nodes to immunize. Defaults to 0.01.
        verbose (bool, optional): Whether to print verbose output for debugging. Defaults to False.
        maxiter (int, optional): Maximum number of iterations as safeguard. Defaults to 1000.

    Returns:
        set: Set of nodes selected for immunization.
    """
    R = int(rw_reps * G.number_of_nodes()) # number of random walk repetitions per iteration
    immunized_nodes = set() # nodes that will be immunized
    nodes = list(G.nodes())
    target_candidates = set() # immunization target candidates that will later be ranked

    neighbor_cache = {}  # lazily populated as nodes are visited
    def get_neighbors(n):
        if n not in neighbor_cache:
            neighbor_cache[n] = set(G.neighbors(n))
        return neighbor_cache[n]
    
    walkSuccessful = False
    iterations = 0
    while not walkSuccessful and iterations < maxiter:
        iterations += 1
        RW = set() # nodes visited by the random walk
        F = set() # "friendship circle" (union of neighborhoods of visited nodes)

        # --- 2 steps to initialise the friendship circle ---
        # We just walk and accumulate F, no target checking yet
        node_curr = np.random.choice(nodes) # start at random node
        RW.add(node_curr) # add to RW set
        f_curr = get_neighbors(node_curr) # get neighbors of current node

        if verbose:
            print(f"\nStarting node {node_curr}, neighbors {f_curr}")

        rw_options = f_curr - RW # options for self-avoiding continuation of the walk
        if not rw_options:
            if verbose:
                print(f"No options to continue from node {node_curr}. Restarting walk.")
            continue # if no options, start a new walk
        F.update(f_curr)  # add step 1 neighbourhood to F before moving to step 2
        node_curr = np.random.choice(list(rw_options)) # move to next node
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
        
        walkSuccessful = True # we successfully completed the initial walk, so we can proceed to the main loop
        if verbose:
            print(f"Start successful. First RW nodes: {RW}, initial friendship circle: {F}. Starting main walk.")
    
        # At this point F contains neighbourhoods of steps 1 and 2
        for trial in range(R):   
            # --- Main walk: check for immunization targets ---
            node_curr = np.random.choice(list(rw_options)) # move to next node
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
                # optionally probe locally? maybe add this later
                if verbose:
                    print(f"Node {node_curr} at trial {trial} added to target candidate pool with unconnected neighbour ratio({count}/{len(f_diff)}).")
            rw_options = f_curr - RW # options for self-avoiding continuation of the walk
            if not rw_options:
                print("Houston, we have a problem: no options to continue the walk. Time to restart.")
                walkSuccessful = False # if no options, we restart the whole process
                break # break out of the trial loop to restart the walk

    if iterations >= maxiter:
        print(f"\nReached maximum iterations ({maxiter}) without successfully completing the walk.")

    # rank target candidates by degree and immunize top fraction
    target_candidates = list(target_candidates)
    target_candidates.sort(key=lambda n: G.degree(n), reverse=True) # sort candidates by degree, descending
    target_count = int(immunized_fraction * G.number_of_nodes())
    if target_count > len(target_candidates):
        print(f"Warning: Not enough target candidates ({len(target_candidates)}) to immunize the desired fraction ({immunized_fraction}). Filling remaining slots randomly.")
        additional_random = np.random.choice(list(set(nodes) - set(target_candidates)), target_count - len(target_candidates), replace=False)
        immunized_nodes.update(additional_random) # add random nodes to immunized set if we don't have enough candidates
        immunized_nodes.update(target_candidates) # immunize all candidates if we don't have enough to fill the fraction
    
    else:
        immunized_nodes.update(target_candidates[:target_count]) # immunize top fraction

    return immunized_nodes


