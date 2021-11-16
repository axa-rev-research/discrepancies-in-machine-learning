import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean


def get_refined_paths_from_node(n, G, n_previous=None):
    """
    From a X_train node (n) in (G), get all the direct path to other X_train nodes (edges of G).
    Output: list of nodes [X_train node, refinement node, ..., X_train node]
    """

    n_neigh = list( G.neighbors(n) )

    if n_previous is not None:
        # Prevent going backward in walking through refined paths
        n_neigh.remove(n_previous)
    #else:
    #    # Select neighbours that have been created during the refinement process (indices of those nodes are negative)
    #    n_neigh = [n for n in n_neigh if n<0]

    if len(n_neigh)==0:
        return -1

    # Continue to walk through the refined paths by recursion (starting by direct neighbours)
    all_paths = []
    for n_next in n_neigh:

        # If the next node is a refined node (i.e. not reached a X_train node), continue walking
        if n_next <0: # attention, now n>0 also has fake twins
            paths = get_refined_paths_from_node(n_next, G, n_previous=n)

            
            # There is probably a bug in that case /!\
            if paths == -1:
                break

            # Gather all refined paths that have been explored
            tmp = []
            for p in paths:
                tmp.append([n]+p)
            paths = tmp
        
        # If the next node is a X_train node, stop walking (stop recursion at that point): we have the whole refined path
        else:
            paths = [[n,n_next]]

        # Gather all refined paths that have been explored
        for p in paths:
            all_paths.append(p)

    return all_paths


def get_all_refined_paths(G):
    """
    Return all the refined path of the graph (hence between X_train nodes)
    """

    
    # Get all X_train nodes from the graph
    X_train_nodes = [n for n in G.nodes if n>=0] #Attention: now, n>=0 also has fake twins
    
    # Get all the refined paths FROM every X_train nodes
    all_paths = {}
    for n in X_train_nodes:
        paths = get_refined_paths_from_node(n, G)

        if paths !=-1 and len(paths)>=1: ### pas clair pour moi on s'int"resse uniquement au cas où il y a plus d'un chemin en partant du n? J'ai changé le > en >=.
            all_paths[n] = paths
            
    # Transform the dict of paths (by start X_train nodes) into a flat list of refined path (unique + crossing an area of discrepancies)
    all_paths_list = []
    for k in all_paths.keys():
        for p in all_paths[k]:

            # Make sure this refined path (or its reverse) is not already in the all_path_list
            if p not in all_paths_list or p[::-1] not in all_paths_list:

                # Make sure the refined path cross an area of discrepancies or at least its X_train nodes have opposite predictions
                # N.B. the check is not ebout that it "crosses" an area of discrepancy here, but if the nodes are in it (p[0] and p[-1])
                # NB2 why it happens if we get refined paths?
                if (G.nodes[p[0]]['pool_predictions'].iloc[0] != G.nodes[p[-1]]['pool_predictions'].iloc[0]) or (G.nodes[p[0]]['discrepancies'] or G.nodes[p[1]]['discrepancies']):
                    all_paths_list.append(p)

    return all_paths_list


def get_path_discrepancies(path, G):
    """
    For a refined path, get a mask (list) with identified non-contiguous discrepancies intervals: 0 for a node without discrepancies, 1 for nodes belonging to the first discrepancies interval, 2 for nodes belonging to a second discrepancies interval. E.g. [0, 0, 1, 1, 1, 0, 2, 2, 0]
    """

    discrepancies = []
    # Counter of non-contiguous discrepancies intervals
    num_discrepancies = 0

    # For each successive node of the refined path
    for i in range(len(path)):
        # If the current node has discrepancies
        if G.nodes(data=True)[path[i]]['discrepancies']==1:
            # If it is either the first node of the path OR the previous node along the path doesn't have discrepancies
            if i==0 or discrepancies[i-1]==0:# T: pq discrepancies[-1] ??? SHOULD BE i-1???? JAI CHANGE
                # Then increment the counter (new on-contiguous discrepancies interval)
                num_discrepancies += 1
            # State that the current node belong to a discrepancies interval
            discrepancies.append(num_discrepancies)

        else:
            # 0 if the current node doesn't belong to a discrepancies interval
            discrepancies.append(0)

    return np.array(discrepancies)


class DiscrepancyInterval:

    def __init__(self, X_train_nodes, discrepancies_interval_mask, path, G):

        self.G = G
        # nodes of the edges (analyzed refined path)
        self.X_train_nodes = X_train_nodes
        self.path = np.array(path)
        self.discrepancies_interval_mask = discrepancies_interval_mask

        self.get_discrepancies_borders()
        self.get_border_features()


    def get_discrepancies_borders(self):
        """
        Get the borders of the discrepancies interval (the nodes on the refined path right outside the interval -i.e. without discrepancies)
        """

        self.border_discrepancies = []
        self.border_bounds_discrepancies = []

        # Walk through the interval
        for i in range(len(self.discrepancies_interval_mask)):

            # Check if the current node (i) is right outside a discrepancies interval - or has discrepancies and is in first or last position in the refined path
            if ((i==0 or i+1==len(self.path)) and self.discrepancies_interval_mask[i]) or (i+1<len(self.path) and not self.discrepancies_interval_mask[i] and self.discrepancies_interval_mask[i+1]) or (i-1>=0 and not self.discrepancies_interval_mask[i] and self.discrepancies_interval_mask[i-1]):
                
                self.border_discrepancies.append(True)

                # Get if bounds of the discrepancies interval are closed or open
                if i-1<0 and self.discrepancies_interval_mask[i] is True:
                    self.border_bounds_discrepancies.append(']')
                elif self.discrepancies_interval_mask[i-1] is False and self.discrepancies_interval_mask[i] is True:
                    self.border_bounds_discrepancies.append('[')
                elif self.discrepancies_interval_mask[i] is True and i+1>len(self.discrepancies_interval_mask)-1:
                    self.border_bounds_discrepancies.append('[')
                elif self.discrepancies_interval_mask[i] is True and self.discrepancies_interval_mask[i+1] is False:
                    self.border_bounds_discrepancies.append(']')
                else:
                    self.border_bounds_discrepancies.append('X')

            else:
                self.border_discrepancies.append(False)


    def get_border_features(self):
        """
        Get the coordinates in the original domain of the borders of the discrepancies interval
        """

        self.border_features = []

        for n in self.path[self.border_discrepancies]:
            tmp = self.G.nodes[n]['features']
            self.border_features.append(tmp)
        self.border_features = pd.DataFrame(self.border_features)


    def get_min_dist_to_point(self, x):
        """
        Return the min distance between a point in the original domain and the nodes within the discrepancies interval
        """
        return np.min([euclidean(self.G.nodes[n]['features'], x) for n in self.path[self.discrepancies_interval_mask]])


def get_discrepancies_intervals(G):
    
    all_paths = get_all_refined_paths(G)
    intervals = []
    # For every refined paths
    for path in all_paths:
        # Get every discrepancies intervals along that path
        discrepancies_intervals_mask = get_path_discrepancies(path, G)
        for d in np.unique(discrepancies_intervals_mask):
            # Discrepancies interval have a mask value > 0 (see get_path_discrepancies method)
            if d>0:
                # Mask to extract nodes of the current discrepancies interval
                discrepancies_mask = discrepancies_intervals_mask==d
                # Create a DiscrepancyInterval object (with X_train nodes of the path, the mask for the current discrepancies interval and the refined path it belongs to)

                di = DiscrepancyInterval((path[0],path[-1]), discrepancies_mask, path, G)
                intervals.append(di)

    return intervals
