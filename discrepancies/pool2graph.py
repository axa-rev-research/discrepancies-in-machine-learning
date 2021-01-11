from itertools import product, combinations, chain
from heapq import heappush, heappop
import logging

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from tqdm import tqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class pool2graph:

    def __init__(self, Xtrain, Ytrain, pool, k=10):

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.pool = pool

        self.k = k

        self.G = nx.Graph()
        self.n_epoch = 0

        # Indicates that precomputed information (e.g. subgraphs of discrepancies) needs to be updated
        self._precomputed_data_deprecated = True
        self._cache = {}


    def fit(self, max_epochs=0):
        
        ## Pre-compute (1) euclidean distance between every point of the training set and (2) get for each point the predictions of each classifier of the pool
        
        #_euclidean_distances_X = euclidean_distances(self.Xtrain)
        _preds = self.pool.predict(self.Xtrain)
        _discrepancies = self.pool.predict_discrepancies(self.Xtrain)

        ## Create nodes from Xtrain and add them to the graph

        _nodes = [(i, {"coords":self.Xtrain.iloc[i], "preds":_preds.iloc[i], "discrepancies":_discrepancies.iloc[i], "y":self.Ytrain.iloc[i], "ground_truth":True, "Xtrain_index":self.Xtrain.index[i]}) for i in range(self.Xtrain.shape[0])]

        self.G.add_nodes_from(_nodes)

        ## Create edges and add them to the graph

        # n_neighbors=self.k+1 because "the query set matches the training set, the nearest neighbor of each point is the point itself, at a distance of zero." (sklearn documentation)
        _NN = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto')
        _NN = _NN.fit(self.Xtrain)
        _distances, _indices = _NN.kneighbors(self.Xtrain)

        # Generate pairs of nodes to be connected by an edge
        indices_distances = np.stack((_indices, _distances), axis=2)
        iterables = [product([i], indices_distances[i][1:]) for i in range(len(indices_distances))]
        _edges = list(chain(*iterables))

        # Reformat: [(node1,node2), edge_length]
        _edges = [[(_edges[i][0],_edges[i][1][0]),_edges[i][1][1]] for i in range(len(_edges))]

        # Remove duplicate edges (if tuple-edge in both directions)
        _edges = np.vstack({tuple(row) for row in _edges})

        # Add edges to the graph
        tmp = [(n1,n2, {'distance':d}) for (n1,n2),d in _edges]
        self.G.add_edges_from(tmp)

        # Graph refinement
        if self.n_epoch < max_epochs:
            self.init_heapqueue()
            # Index for new nodes (index *stricly negative* to distinguish from actual points from X_train)
            self.new_nodes_index = 0

        for n_epoch in tqdm(range(max_epochs)):
            self.n_epoch = n_epoch
            self.refine_graph()

        # Indicates that precomputed information (e.g. subgraphs of discrepancies) needs to be updated
        self._precomputed_data_deprecated = True


    def refine_graph(self):

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
            u.append(self.G.nodes[to_refine[i,1][0]]['coords'])
            v.append(self.G.nodes[to_refine[i,1][1]]['coords'])
            w.append((u[-1]+v[-1])/2)

        u = np.array(u)
        v = np.array(v)
        w = np.array(w)
        try:
            w_preds = self.pool.predict(w)
        except:
            print(u)
            print(v)
            print(w)
        w_discrepancies = self.pool.predict_discrepancies(w)

        d_uw = np.linalg.norm(u-w, axis=1)
        d_vw = np.linalg.norm(v-w, axis=1)

        w = pd.DataFrame(np.array(w))
        
        ## Step 3: add the new nodes and edges to the graph and the heapq
        
        self.G.remove_edges_from(to_refine[:,1])

        new_nodes = []
        for i in range(len(w)):
            self.new_nodes_index += -1
            new_node = (self.G.number_of_nodes()+i, {'preds':w_preds.iloc[i], 'coords':w.iloc[i], 'discrepancies':w_discrepancies.iloc[i], 'y':None, 'ground_truth':False, "Xtrain_index":self.new_nodes_index})

            new_nodes.append(new_node)
        new_nodes = np.array(new_nodes)

        self.G.add_nodes_from(new_nodes)

        # new_nodes = np.array([(self.G.number_of_nodes()+i, {'preds':w_preds.iloc[i], 'coords':w.iloc[i], 'discrepancies':w_discrepancies.iloc[i], 'y':None, 'ground_truth':False, "Xtrain_index":-1}) for i in range(len(w))])
        # self.G.add_nodes_from(new_nodes)

        new_edges1 = [(to_refine[i,1][0], new_nodes[i,0], {'distance':d_uw[i]}) for i in range(len(new_nodes))]
        new_edges2 = [(to_refine[i,1][1], new_nodes[i,0], {'distance':d_vw[i]}) for i in range(len(new_nodes))]

        new_edges = np.array(new_edges1+new_edges2)

        self.G.add_edges_from(new_edges)

        # add new edges to heapq
        for i in range(len(new_edges)):
            u = new_edges[i][0]
            v = new_edges[i][1]

            if (self.G.nodes[u]['y'] != self.G.nodes[v]['y']) or (self.G.nodes[u]['discrepancies']==1 or self.G.nodes[v]['discrepancies']==1):
                heappush(self.heapq, (-new_edges[i][2]['distance'], (new_edges[i][0],new_edges[i][1])))

        # Indicates that precomputed information (e.g. subgraphs of discrepancies) needs to be updated
        self._precomputed_data_deprecated = True


    def init_heapqueue(self):
        self.heapq = []

        for e in self.G.edges(data=True):
            # If vertices of the edge have different ground truth labels OR if one of the vertex has prediction discrepancies OR if vertices have predicted labels (first prediction ['preds'].iloc[0] is only checked: because discrepancy in prediction are also catched)
            if (self.G.nodes[e[0]]['y'] != self.G.nodes[e[1]]['y']) or (self.G.nodes[e[0]]['discrepancies']==1 or self.G.nodes[e[1]]['discrepancies']==1 or (self.G.nodes[e[0]]['preds'].iloc[0] != self.G.nodes[e[1]]['y'].iloc[0])):

                heappush(self.heapq, (-e[2]['distance'], (e[0],e[1])))


    #############################################
    ## Extraction of information from the graph
    #############################################
    
    def get_nodes(self, discrepancies=None):
        if discrepancies:
            # Get nodes with discrepancies
            nodes = [node for node in self.G.nodes(data=True) if node[1]['discrepancies']==1]

        elif not discrepancies:
            # Get nodes without discrepancies
            nodes = [node for node in self.G.nodes(data=True) if node[1]['discrepancies']==0]

        else:
            # Get all nodes
            nodes = [node for node in self.G.nodes(data=True)]

        return nodes


    def get_nodes_attributes(self, lnodes):
        """
        Get consolidated attribute information for a list of nodes under the form of pandas.DataFrame/Series.
        """

        # TODO: replace 'Xtrain_index' by a universal index, also eligible for new nodes not in the train set
        Xtrain_index = [n[1]['Xtrain_index'] for n in lnodes]

        coords = pd.DataFrame([n[1]['coords'] for n in lnodes], index=Xtrain_index)
        preds = pd.DataFrame([n[1]['preds'] for n in lnodes], index=Xtrain_index)
        y_true = pd.DataFrame([n[1]['y'] for n in lnodes], index=Xtrain_index)
        ground_truth = pd.Series([n[1]['ground_truth'] for n in lnodes], index=Xtrain_index)

        return coords, preds, y_true, ground_truth


    def get_subgraphs_discrepancies(self, ordered=True):
        # Get nodes with discrepancies
        nodes = (
            node
            for node, data
            in self.G.nodes(data=True)
            if data.get("discrepancies") == 1
        )

        # Extract the subgraph (with edges) corresponding to the selected nodes
        G_discrepancies = self.G.subgraph(nodes)

        # Split the subgraph into components (non-connected graphs)
        G_discrepancies_components = [self.G.subgraph(c).copy() for c in nx.connected_components(G_discrepancies)]

        if ordered and len(G_discrepancies_components)>1:
            n_nodes_by_component = [len(component) for component in G_discrepancies_components]
            order = np.argsort(n_nodes_by_component)[::-1]
            G_discrepancies_components = np.array(G_discrepancies_components)[order]

        return G_discrepancies_components


    def get_discrepancies_clusters(self):

        lnodes, lcluster = [], []

        G_discrepancies_components = self.get_subgraphs_discrepancies(ordered=True)

        for i in range(len(G_discrepancies_components)):

            lnodes_i = [node[1]['Xtrain_index'] for node in G_discrepancies_components[i].nodes(data=True)]
            lnodes += lnodes_i
            lcluster += [i+1]*len(lnodes_i)

        lcluster = pd.Series(lcluster, index=lnodes, name='cluster')

        return lcluster


    def get_discrepancies_dataset(self, binarize=True):
        """

        binarize: if True, every points with discrepancies will have label '1' (vs. label '0' for points without discrepancies). If False, the label will be an integer in [2, +inf[ corresponding to the cluster number the point with discrepancies belongs to.
        """

        ## Get data from the graph

        # Get attributes for nodes WITH discrepancies
        lnodes = self.get_nodes(discrepancies=True)
        DISCR_coords, DISCR_preds, DISCR_y_true, DISCR_ground_truth = self.get_nodes_attributes(lnodes)
        DISCR_lclusters = self.get_discrepancies_clusters()

        # Get attributes for nodes WITHOUT discrepancies
        lnodes = self.get_nodes(discrepancies=False)
        coords, preds, y_true, ground_truth = self.get_nodes_attributes(lnodes)

        ## Gather dataset
        X_discr = pd.concat((coords, DISCR_coords), axis=0)

        if binarize:
            y_discr = pd.Series([0]*len(coords)+[1]*len(DISCR_coords))
        else:
            y_discr = pd.Series([0]*len(coords), index=coords.index)
            y_discr = pd.concat((y_discr,DISCR_lclusters), axis=0)

        return X_discr, y_discr.loc[X_discr.index]


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


    def plot_subgraphs_discrepancies(self):

        G_discrepancies_components = self.get_subgraphs_discrepancies_in_predictions()

        pos = {}
        for n in self.G.nodes(data=True):
            node = n[0]
            coords = (n[1]['coords'].to_list())
            pos[node] = coords
        nx.draw(self.G, pos=pos, node_size=30)

        colors = iter(cm.rainbow(np.linspace(0,1,len(G_discrepancies_components))))

        for i in range(len(G_discrepancies_components)):
            pos = {}
            for n in G_discrepancies_components[i].nodes(data=True):
                node = n[0]
                coords = (n[1]['coords'].to_list())
                pos[node] = coords

            c = next(colors)
            nx.draw(G_discrepancies_components[i], pos=pos, node_color=c, node_size=30)


    def plot_domain(self, plot_db=False, n_cluster=None):
        """
        Plot the domain (feature space) in 2D (using T-SNE to reduce the dimensionality if dim(X)>2). Points are both the training set X_train and new points drawn during the refinement of the graph. Cluster of discrepancies are highlighted.
        """

        ## Get data from the graph

        # Get attributes for nodes WITH discrepancies
        lnodes = self.get_nodes(discrepancies=True)
        DISCR_coords, DISCR_preds, DISCR_y_true, DISCR_ground_truth = self.get_nodes_attributes(lnodes)
        DISCR_lclusters = self.get_discrepancies_clusters()

        # Get attributes for nodes WITHOUT discrepancies
        lnodes = self.get_nodes(discrepancies=False)
        coords, preds, y_true, ground_truth = self.get_nodes_attributes(lnodes)

        ## Plot Graph

        ax = plt.subplot()

        if self.Xtrain.shape[1]>2:
            # TODO: how to plot efficiently the decision boundary in dim(X)>2?
            DISCR_coords_embedded, coords_embedded = self.get_TSNE_projection([DISCR_coords,coords])

        else:
            if plot_db:
                self.plot_db()
            DISCR_coords_embedded = DISCR_coords
            coords_embedded = coords

        coords_embedded[ground_truth==True].plot(kind='scatter', x=0, y=1, marker='o', c='grey', ax=ax)
        coords_embedded[ground_truth==False].plot(kind='scatter', x=0, y=1, marker='x', c='grey', ax=ax)

        coords_lcluster = pd.concat((DISCR_coords_embedded, DISCR_lclusters), axis=1)

        if n_cluster is None:
            
            coords_lcluster[DISCR_ground_truth==True].plot(kind='scatter', x=0, y=1, c='cluster', marker='d', colormap='tab20', ax=ax)

        else:
            
            coords_lcluster[DISCR_ground_truth==True][coords_lcluster.cluster==n_cluster].plot(kind='scatter', x=0, y=1, c='cluster', marker='d', colormap='tab20', ax=ax)

