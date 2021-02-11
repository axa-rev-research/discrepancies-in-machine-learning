import warnings
warnings.filterwarnings('ignore')

from itertools import product, combinations, chain
import toolz
from heapq import heappush, heappop
import logging

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

# TODO: make sure the k=0 case is managed properly
# TODO: 20 newsgroup is crashing
# TODO: deal with categorical variables

class pool2graph:

    def __init__(self, Xtrain, Ytrain, pool, k_init=10, k_refinement=0, X_feature_discrete=None):

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.pool = pool

        # if no precisions on discrete features, it is assumed all features are  continuous
        # TODO: distinction continuous/discrete not use for now. For the refinement algo the idea would be to create 2 nodes, with the corresponding edges, with each of them having the discrete values of one of the two original nodes (the numeric values being shared as usual, as the middle of the two original nodes). It remains the question of the calculation of the distance (heap queue, k nearest nodes)
        if X_feature_discrete is None:
            self.X_feature_discrete = [False for _ in range(self.Xtrain.shape[1])]
            
        else:
            if len(X_feature_discrete)==self.Xtrain.shape[1] and sum([t in [True, False] for t in np.unique([True, False, False, True])])==self.Xtrain.shape[1]:
                self.X_feature_discrete = X_feature_discrete
            else:
                raise ValueError

        self.k_init = k_init
        self.k_refinement = k_refinement

        self.G = nx.Graph()
        self.n_epoch = 0

        # Indicates that precomputed information (e.g. subgraphs of discrepancies) needs to be updated
        self._precomputed_data_deprecated = True
        self._cache = {}


    def fit(self, max_epochs=0, stopping_criterion=None):
        """Fit a graph that describes predictions' discrepancies across the input domain (Xtrain).

        The graph is initialized at no cost (almost, if predictions where recycled from the training/evaluation of the pool), where each node of the graph is a point of the training set Xtrain.

        All the nodes are linked with their k nearest nodes. Edges whose nodes have predictions' discrepancies or different prediction's labels describe an area of predictions' discrepancies. These edges are called edges with discrepancies.

        The graph can be refined for a more precise description of the areas of predictions' discrepancies.
        The refinement can be made during several iterations (controled by max_epochs) or until convergence (controled by stopping_criterion).
        Each refinement's iteration aim at decreasing the length of edges with discrepancies, to have a more precise definition of the location of discrepancies areas. To do so, edges with discrepancies are split in half, a new node is inserted in the graph (at the position of the split) with its attributes (presence of prediction discrepancies). The former edge is removed from the graph and 2 new edges are added that link the 2 original nodes to the new node. At least one of the 2 new edges is an edge with discrepancies.
        
        After several iterations of graph refinement, edges with discrepancies have shorter lengths, leading to a more precise location of areas of discrepancies in the feature space of Xtrain.

        Parameters
        ----------
        max_epoch : int
            Number of iterations of graph's refinement (default=0)

        stopping_criterion : float in [0,1] or None
            Expressed as the minimal decrease in percentage of the sum of the lengths of edges with discrepancies between t-1 and t to continue the graph's refinement (i.e. proceed with iteration t+1). If None, not used.

        Returns
        -------
        self : pool2graph
            Fitted pool2graph
        """
        
        ## Pre-compute (1) euclidean distance between every point of the training set and (2) get for each point the predictions of each classifier of the pool
        
        #_euclidean_distances_X = euclidean_distances(self.Xtrain)
        _preds = self.pool.predict(self.Xtrain)
        _preds.index = self.Xtrain.index
        _discrepancies = self.pool.predict_discrepancies(self.Xtrain)
        _discrepancies.index = self.Xtrain.index

        ## Create nodes from Xtrain and add them to the graph

        _nodes = [(i,
                {"features":self.Xtrain.loc[i],
                "pool_predictions":_preds.loc[i], "discrepancies":_discrepancies.loc[i],
                "y_true":self.Ytrain.loc[i],
                "ground_truth":True,
                "Xtrain_index":i})
                for i in self.Xtrain.index]

        self.G.add_nodes_from(_nodes)

        ## Create edges and add them to the graph

        _edges = self.get_edges_kneighbors()
        self.G.add_edges_from(_edges)

        # If the graph has actually edges to refine
        if len(_edges)>0:

            # Graph refinement
            if self.n_epoch < max_epochs:

                # Initialize and populate heapqueue (to prioritize edges' refinement)
                self.heapq = []
                for e in self.G.edges(data=True):
                    if self._edge_selection(e):
                        heappush(self.heapq, (-e[2]['distance'], (e[0],e[1])))

                # Index for new nodes (index *stricly negative* to distinguish from actual points from X_train)
                self.new_nodes_index = 0

            previous_sum_distances = None
            for n_epoch in range(max_epochs):

                sum_distances = self.get_sum_distances()

                # If not the first iteration (previous_sum_distances != None) and if the sum_distances decreases at least at the following pace: (previous_sum_distances - sum_distances) / previous_sum_distances)< stopping_criterion, where stopping_criterion is expressed in %
                # TODO: not a good stoppping criterion. When self.k_refinement>0, we add many edges at each iteration => the overall distance increases
                if (stopping_criterion != None) and (previous_sum_distances != None) and (np.abs(previous_sum_distances - sum_distances) / previous_sum_distances) < stopping_criterion:
                    break
                else:
                    previous_sum_distances = sum_distances

                logging.info("### EPOCH #"+str(n_epoch)+" - Sum of distances ="+str(sum_distances)+"\n")

                self.n_epoch = n_epoch
                
                self._refine_graph()

        # Indicates that precomputed information (e.g. subgraphs of discrepancies) needs to be updated because the graph has changed
        self._precomputed_data_deprecated = True

        return self


    def get_edges_kneighbors(self, lnodes=None, k=None):
        """Returns the edges between each node and its k nearest nodes in the graph. If lnodes is None, all the edges between every node of the graph and their k nearest nodes are returned. If lnodes is a list of nodes (by their index), the method returns the edges between the lnodes only and their k nearest nodes.

        Parameters
        ----------
        lnodes : None or list
            If None (default), the method returns the edges for all the nodes with their k nearest nodes. If lnodes is a list of nodes (by their index), the edges for the lnodes with their k nearest nodes is returned

        k : None or int
            Number of nearest neighbors to search. If None (default), use self.k_init or can be manually defined (int).

        Returns
        -------
        _edges : np.array
            Array of Networkx edges (NOT added to the graph self.G: they NEED to be added to the graph self.G)
        """

        # Get features from all nodes of the graph
        nodes = self.G.nodes(data=True)
        nodes_features = {n[0]:n[1]['features'] for n in nodes}
        nodes_features = pd.concat(nodes_features, axis=1).T

        if k is None:
            k = self.k_init

        # n_neighbors=self.k_init+1 because "the query set matches the training set, the nearest neighbor of each point is the point itself, at a distance of zero." (sklearn documentation)
        _NN = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
        _NN = _NN.fit(nodes_features)

        if lnodes is None:
            lnodes = nodes_features.index

        _distances, _indices = _NN.kneighbors(nodes_features.loc[lnodes])
        _indices = nodes_features.index[_indices]

        # Generate pairs of nodes to be connected by an edge
        indices_distances = np.stack((_indices, _distances), axis=2)
        iterables = [product([nodes_features.index[i]], indices_distances[i][1:]) for i in range(len(indices_distances))]
        _edges = list(chain(*iterables))
        
        # Reformat, with standard edge format: (node1,node2, {'distance':edge_length})
        _edges = [(_edges[i][0],_edges[i][1][0],{'distance':_edges[i][1][1]}) for i in range(len(_edges))]

        # Remove duplicate edges (if tuple-edge in both directions)
        # _edges = list({tuple(row) for row in _edges})
        # _edges = np.vstack(_edges)
        _edges = self.unique_edges(_edges)

        return _edges


    def _refine_graph(self):
        """
        Subroutine for the graph refinement (fit of the graph).
        """

        self.n_epoch += 1

        ## Step 1: get all the edges that can be refined without breaking the order of edge refinement (don't split an edge smaller than an edge resulting from the split of the biggest edge to split from the batch)

        to_refine = []

        # First iteration - to avoid the test at each loop
        e = heappop(self.heapq)
        to_refine.append(e)
        threshold = e[0]/2

        # Check which edges can be included in this batch of refinement to preserve the order of processing (longest edges first)
        while self.heapq:
            e = heappop(self.heapq)
            if e[0]>threshold:
                heappush(self.heapq, (e[0], (e[1][0],e[1][1])))
                break
            else:
                to_refine.append(e)

        to_refine = np.array(to_refine, dtype=object)
        
        ## Step 2: pre-compute the positions of the new nodes in the feature space, their blackbox' label and their distance to their nodes (edge being split)
        
        u, v, w = [],[],[]
        for i in range(len(to_refine)):
            u.append(self.G.nodes[to_refine[i,1][0]]['features'])
            v.append(self.G.nodes[to_refine[i,1][1]]['features'])
            w.append((u[-1]+v[-1])/2)

        u = np.array(u)
        v = np.array(v)
        w = np.array(w)
        w_preds = self.pool.predict(w)

        w_discrepancies = self.pool.predict_discrepancies(w)

        d_uw = np.linalg.norm(u-w, axis=1)
        d_vw = np.linalg.norm(v-w, axis=1)

        w = pd.DataFrame(np.array(w), columns=self.Xtrain.columns)
        
        ## Step 3: add the new nodes and edges to the graph and the heapq
        
        self.G.remove_edges_from(to_refine[:,1])

        new_nodes = []
        for i in range(len(w)):
            self.new_nodes_index += -1

            new_node = (self.new_nodes_index,
            {'pool_predictions':w_preds.iloc[i],
            'features':w.iloc[i],
            'discrepancies':w_discrepancies.iloc[i],
            'y_true':None,
            'ground_truth':False,
            'Xtrain_index':self.new_nodes_index})

            new_nodes.append(new_node)

        new_nodes = np.array(new_nodes)

        self.G.add_nodes_from(new_nodes)

        # Format, with standard edge format: (node1,node2, {'distance':edge_length})
        new_edges1 = [(to_refine[i,1][0], new_nodes[i,0], {'distance':d_uw[i]}) for i in range(len(new_nodes))]
        new_edges2 = [(to_refine[i,1][1], new_nodes[i,0], {'distance':d_vw[i]}) for i in range(len(new_nodes))]

        # If the option to add edges with the k nearest nodes of the new nodes
        k_edges = []
        if self.k_refinement>0:
            # Get index (in the graph) of the new nodes
            new_nodes_index = [n[0] for n in new_nodes]
            # Compute the self.k_refinement nearest nodes of the new nodes
            k_edges = self.get_edges_kneighbors(lnodes=new_nodes_index, k=self.k_refinement)

        new_edges = new_edges1+new_edges2+k_edges

        # Remove duplicate edges (if tuple-edge in both directions)
        new_edges = self.unique_edges(new_edges)

        self.G.add_edges_from(new_edges)

        # Add new edges to the heapqueue if they meet refinement's policy criterions
        for i in range(len(new_edges)):
            u = new_edges[i][0]
            v = new_edges[i][1]
            e = (u,v)

            if self._edge_selection(e):
                heappush(self.heapq, (-new_edges[i][2]['distance'], e))

        # Indicates that precomputed information (e.g. subgraphs of discrepancies) needs to be updated because the graph has changed
        self._precomputed_data_deprecated = True


    def unique_edges(self, edges):
        """Return a list of edges where each edge is unique, no matter the direction (u,v)=(v,u): the pair of nodes u,v will have a unique edge no matter the order.

        Parameters
        ----------
        edges : list
            List of edges to "uniqueify", edge with the standard format ((node1, node2), {'distance':distance_float})

        Returns
        -------
        _edges : list
            List of unique edge, with the standard format (node1, node2, {'distance':distance_float})
        """

        def edge_uniqueness(e):
            return tuple(sorted([e[0],e[1]]))

        _edges = []
        for e in toolz.unique(edges, key=edge_uniqueness):
            _edges.append(e)

        return _edges


    def _edge_selection(self, e, policy='discrepancy+differentPredictions'):
        """Return True if a graph's edge in input should be refined (i.e. edge being split to better locate area of discrepancies).
        Different selection policies can be used. Only the 'discrepancy+differentPredictions' policy is implemented.
        
        Parameters
        ----------
        e : Networkx's edge, tuple (node u, node v, edge's attributes)
            The edge to evaluate according to refinement's policy
        policy : str, optional
            Name of the refinement's policy, options:
            - 'discrepancy+differentPredictions': check (1) if vertices of the edge have different predicted labels OR (2) if one of the vertex has prediction discrepancies.

        Returns
        -------
        selected : boolean
            Return True if the edge meets policy's criterion and should be refined. False if not.
        """

        selected = False

        if policy == 'discrepancy+differentPredictions':
            # If vertices of the edge have different predicted labels (first prediction ['pool_predictions'].iloc[0] is only checked: because discrepancy in prediction is also catched) OR if one of the vertex has prediction discrepancies
            selected = (self.G.nodes[e[0]]['pool_predictions'].iloc[0] != self.G.nodes[e[1]]['pool_predictions'].iloc[0]) or (self.G.nodes[e[0]]['discrepancies'] or self.G.nodes[e[1]]['discrepancies'])

        return selected

    #############################################
    ## Extract information from the graph
    #############################################

    def get_sum_distances(self):
        """Return the sum of edges' distances for edges that respect the selection criterion (nodes with discrepancies, nodes with opposite class predictions).

        Returns
        -------
        sum_distances : float
            Sum of edges' distances for edges that meet edges' refinement policy.
        """

        sum_distances = 0
        for e in self.G.edges(data=True):
            if self._edge_selection(e):
                sum_distances += e[2]['distance']

        return sum_distances


    def get_nodes(self, discrepancies=None):
        """Return all the nodes of the graph (default), or the subset of nodes with predictions' discrepancy, or the subset of nodes without predictions' discrepancy.

        Parameters
        ----------
        discrepancies : None or boolean, optional
            Characteristics of the returned nodes, options:
            - None: default, return all the graph's nodes.
            - True: return all the graph's nodes with predictions' discrepancy.
            - False: return all the graph's nodes without predictions' discrepancy.

        Returns
        -------
        selected : list
            List of selected nodes.
        """

        if discrepancies is None:
            # Get all nodes
            nodes = [node for node in self.G.nodes(data=True)]
        elif discrepancies in [True, False] :
            # Get nodes with OR without discrepancies
            nodes = [node for node in self.G.nodes(data=True) if node[1]['discrepancies']==discrepancies]
        else:
            raise ValueError
            
        return nodes


    def get_nodes_attributes(self, lnodes):
        """Get consolidated attribute information for a list of nodes under the form of pandas.DataFrame/Series.

        Parameters
        ----------
        lnodes : list
            List of nodes for which attributes have to be returned

        Returns
        -------
        features : pd.DataFrame
            Coordinates of the nodes' point in the original domain (Xtrain). Index: nodes. Columns: features' values
        
        pool_predictions : pd.DataFrame
            Pool's predictions for the nodes' points. Index: nodes. Columns: one prediction per pool's predictor

        y_true : pd.Series
            True label (ytrain) for nodes' points

        ground_truth : pd.Series
            True/False according to appartenance of the nodes' points to Xtrain

        """

        Xtrain_index = [n[1]['Xtrain_index'] for n in lnodes]

        features = pd.DataFrame([n[1]['features'] for n in lnodes], index=Xtrain_index)
        pool_predictions = pd.DataFrame([n[1]['pool_predictions'] for n in lnodes], index=Xtrain_index)
        y_true = pd.DataFrame([n[1]['y_true'] for n in lnodes], index=Xtrain_index)
        ground_truth = pd.Series([n[1]['ground_truth'] for n in lnodes], index=Xtrain_index)

        return features, pool_predictions, y_true, ground_truth


    def get_subgraphs_discrepancies(self, ordered=True):
        """Returns a list of connected subgraphs (components) whose nodes have predictions' discrepancies.

        Parameters
        ----------
        ordered : bool, optional
            True (default) to return a list sorted by decreasing number of nodes

        Returns
        -------
        components : list
            List of subgraphs

        """

        # Get nodes with discrepancies
        nodes = (
            node
            for node, data
            in self.G.nodes(data=True)
            if data.get("discrepancies") == True
        )

        # Extract the subgraph (with edges) corresponding to the selected nodes
        G_discrepancies = self.G.subgraph(nodes)

        # Split the subgraph into components (non-connected graphs)
        components = [self.G.subgraph(c).copy() for c in nx.connected_components(G_discrepancies)]

        if ordered and len(components)>1:
            n_nodes_by_component = [len(component) for component in components]
            order = np.argsort(n_nodes_by_component)[::-1]
            components = [components[i] for i in order]

        return components


    def get_discrepancies_components(self):

        lnodes, lcomponents = [], []

        G_discrepancies_components = self.get_subgraphs_discrepancies(ordered=True)

        for i in range(len(G_discrepancies_components)):

            lnodes_i = [node[1]['Xtrain_index'] for node in G_discrepancies_components[i].nodes(data=True)]
            lnodes += lnodes_i
            lcomponents += [i+1]*len(lnodes_i)

        lcomponents = pd.Series(lcomponents, index=lnodes, name='cluster')

        return lcomponents


    def get_discrepancies_dataset(self, binarize=True):
        """

        binarize: if True, every points with discrepancies will have label '1' (vs. label '0' for points without discrepancies). If False, the label will be an integer in [2, +inf[ corresponding to the cluster number the point with discrepancies belongs to.
        """

        ## Get data from the graph

        # Get attributes for nodes WITH discrepancies
        lnodes = self.get_nodes(discrepancies=True)
        DISCR_features, DISCR_pool_predictions, DISCR_y_true, DISCR_ground_truth = self.get_nodes_attributes(lnodes)
        DISCR_lclusters = self.get_discrepancies_components()

        # Get attributes for nodes WITHOUT discrepancies
        lnodes = self.get_nodes(discrepancies=False)
        features, pool_predictions, y_true, ground_truth = self.get_nodes_attributes(lnodes)

        ## Gather dataset
        X_discr = pd.concat((features, DISCR_features), axis=0)

        y_discr = pd.Series([0]*len(features))
        y_discr = pd.concat((y_discr, DISCR_lclusters), axis=0)
        y_discr.index = X_discr.index

        if binarize:
            y_discr = (y_discr>0).astype('int')

        return X_discr, y_discr


    ################################################
    ## Display/Plot Information from the graph/pool
    ################################################

    def get_TSNE_projection(self, ldf):

        if self._precomputed_data_deprecated:

            X_to_transform = pd.concat(ldf, axis=0)
            X_embedded = TSNE(n_components=2, init='pca').fit_transform(X_to_transform)

            ldf_embedded = []
            cursor = 0
            for i in range(len(ldf)):
                tmp = X_embedded[cursor:cursor+len(ldf[i])]
                tmp = pd.DataFrame(tmp, index=ldf[i].index)
                ldf_embedded.append(tmp)

            # Cache results
            self._cache['ldf_embedded'] = ldf_embedded

        else:
            ldf_embedded = self._cache['ldf_embedded']

        return ldf_embedded


    def plot_db(self):
        """
        /!/ right now in 2d - add t-sne or something for for D ?
        """

        ax = plt.subplot()

        cmap = plt.cm.get_cmap('plasma')
        norm_colors = matplotlib.colors.Normalize(vmin=0, vmax=len(self.pool.models))

        try:
            self.X_mesh
        except:

            x_min = self.Xtrain.iloc[:,0].min()
            x_max = self.Xtrain.iloc[:,0].max()
            y_min = self.Xtrain.iloc[:,1].min()
            y_max = self.Xtrain.iloc[:,1].max()
            x_step = y_step = 0.005

            x = np.arange(x_min, x_max, x_step)
            y = np.arange(y_min, y_max, y_step)
            xx, yy = np.meshgrid(x, y)

            self.X_mesh = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
            self.xx = xx
            self.yy = yy

            self.X_mesh_preds = {} 
            for clf_name in self.pool.models:
                clf = self.pool.models[clf_name]
                self.X_mesh_preds[clf_name] = clf.predict(self.X_mesh)
                self.X_mesh_preds[clf_name] = self.X_mesh_preds[clf_name].reshape((self.xx.shape[0], self.yy.shape[1]))

        for i in range(len(self.pool.models)):
            clf = list(self.pool.models.keys())[i]
            plt.contour(self.xx, self.yy, self.X_mesh_preds[clf], colors=[cmap(norm_colors(i))])

        left, right = plt.xlim()
        plt.xlim((left*1.1, right*1.1))
        bottom, top = plt.ylim()
        plt.ylim((bottom*1.1, top*1.1))

        return ax


    def plot_subgraphs_discrepancies(self):

        G_discrepancies_components = self.get_subgraphs_discrepancies_in_predictions()

        pos = {}
        for n in self.G.nodes(data=True):
            node = n[0]
            features = (n[1]['features'].to_list())
            pos[node] = features
        nx.draw(self.G, pos=pos, node_size=30)

        colors = iter(cm.rainbow(np.linspace(0,1,len(G_discrepancies_components))))

        for i in range(len(G_discrepancies_components)):
            pos = {}
            for n in G_discrepancies_components[i].nodes(data=True):
                node = n[0]
                features = (n[1]['features'].to_list())
                pos[node] = features

            c = next(colors)
            nx.draw(G_discrepancies_components[i], pos=pos, node_color=c, node_size=30)


    def plot_domain(self, plot_db=False, n_cluster=None):
        """
        Plot the domain (feature space) in 2D (using T-SNE to reduce the dimensionality if dim(X)>2). Points are both the training set X_train and new points drawn during the refinement of the graph. Cluster of discrepancies are highlighted.
        """

        ## Get data from the graph

        # Get attributes for nodes WITH discrepancies
        lnodes = self.get_nodes(discrepancies=True)
        DISCR_features, DISCR_pool_predictions, DISCR_y_true, DISCR_ground_truth = self.get_nodes_attributes(lnodes)
        DISCR_lclusters = self.get_discrepancies_clusters()

        # Get attributes for nodes WITHOUT discrepancies
        lnodes = self.get_nodes(discrepancies=False)
        features, pool_predictions, y_true, ground_truth = self.get_nodes_attributes(lnodes)

        ## Plot Graph

        ax = plt.subplot()

        if self.Xtrain.shape[1]>2:
            # TODO: how to plot efficiently the decision boundary in dim(X)>2?
            DISCR_features_embedded, features_embedded = self.get_TSNE_projection([DISCR_features, features])

        else:
            if plot_db:
                self.plot_db()
            DISCR_features_embedded = DISCR_features
            features_embedded = features

        features_embedded[ground_truth==True].plot(kind='scatter', x=0, y=1, marker='o', c='grey', ax=ax)
        features_embedded[ground_truth==False].plot(kind='scatter', x=0, y=1, marker='x', c='grey', ax=ax)

        features_lcluster = pd.concat((DISCR_features_embedded, DISCR_lclusters), axis=1)

        if n_cluster is None:
            
            features_lcluster[DISCR_ground_truth==True].plot(kind='scatter', x=0, y=1, c='cluster', marker='d', colormap='tab20', ax=ax)

        else:
            
            features_lcluster[DISCR_ground_truth==True][features_lcluster.cluster==n_cluster].plot(kind='scatter', x=0, y=1, c='cluster', marker='d', colormap='tab20', ax=ax)

