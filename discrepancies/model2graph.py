from itertools import product, combinations
from heapq import heappush, heappop
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


"""
TODO and ideas
- multiple crossing of the decision boundary is handled ?
- T: is it relevant to connect all the points ? we may generate edges OOD. instead: hyper-parameter on distance or k-nn
- Assessment of the m2g fidelity to the original model (test set? monte carlo?)
- Compare m2g with the habitual decision boundary representation (based on a systematic mesh on 2D, like sklearn representations of 2D-models)
- Metrics on the m2g: "complexity", "isolated db/pockets", etc.
- Compare m2g of distinct models: compare segment from 1 point of the training set to another (where the DB intersect it) => highlight the differences at different levels of granularity
- Compare the highlighted discrepancies with difference of predictions
- Take into account ordinal and categorical variables
- (Extension to images and text ?)
"""

class model2graph:
    """

    Documentation of the properties of the graph G
    - Nodes
        - touchpoint: this node is the best approximation of the location of the decision boundary on the edge - on its side of the decision boundary (for nodes of its class)

    - Edges
    """

    def __init__(self, Xtrain, Ytrain, clf):
        """
        Xtrain: pd.DataFrame, training set's input
        Ytrain: pd.DataFrame, training set's target
        clf: trained classifier respecting sklearn API
        """

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.clf = clf

        self.G = nx.Graph()

        self._euclidean_distances_X = euclidean_distances(self.Xtrain)
        self._preds = clf.predict(self.Xtrain)

        # Create nodes from Xtrain and add them to the graph
        for x in range(len(self.Xtrain)):
            self.add_node(x, self._preds[x], self.Xtrain.iloc[x], None, True)

        # Recall original list of nodes, from Xtrain
        self.Xtrain_nodes = list(self.G)

        # Create edges and add them to the graph
        tmp = [(n1,n2, {'distance':self._euclidean_distances_X[n1,n2]}) for n1,n2 in product(self.G.nodes, self.G.nodes) if (n1 != n2) and (self.G.nodes[n1]['pred'] != self.G.nodes[n2]['pred'])]
        self.G.add_edges_from(tmp)
        
        self.h = []
        self.pairs = []
        for u,v,d in self.G.edges(data=True):
            if self.G.nodes[u]['pred'] != self.G.nodes[v]['pred']:
                # Invert distance to be able to pop the edge with the largest distance
                heappush(self.h, (-d['distance'], (u,v)))
                self.pairs.append([u,v])

        self.n_epochs = 0

        self.log_graph_stats()


    def fit(self, max_epoch=10, early_stopping=True, tol=0.05):

        mean_nodes_dist, std_nodes_dist = [], []

        _mean_nodes_dist, _std_nodes_dist = self.get_graph_stats()
        mean_nodes_dist.append(_mean_nodes_dist)
        std_nodes_dist.append(_std_nodes_dist)

        for n_iter in range(max_epoch):
    
            self.refine_graph()

            if n_iter>0 and early_stopping:
                _mean_nodes_dist, _std_nodes_dist = self.get_graph_stats()
                mean_nodes_dist.append(_mean_nodes_dist)
                std_nodes_dist.append(_std_nodes_dist)

                # Compute rate of converge for the mean edge distance and std
                n_iter_no_change=1

                perc_diff_mean = (mean_nodes_dist[-1-n_iter_no_change]-mean_nodes_dist[-1])/mean_nodes_dist[-1-n_iter_no_change]

                perc_diff_std = (std_nodes_dist[-1-n_iter_no_change]-std_nodes_dist[-1])/std_nodes_dist[-1-n_iter_no_change]

                logging.info(" Mean node distance : "+str(mean_nodes_dist[-1]))
                logging.info(" Std node distance : "+str(std_nodes_dist[-1]))
                logging.info(" Mean node distance - Change in % : "+str(np.round(perc_diff_mean,4)*100))
                logging.info(" Std node distance - Change in % : "+str(np.round(perc_diff_std,4)*100))

                # Stop iteration as soon as the mean distances between nodes is 0 or the rate of convergence of both the mean and std distances between nodes is below tolerance
                if (mean_nodes_dist[-1]==0 or perc_diff_mean<tol) and (std_nodes_dist[-1]==0 or perc_diff_std<tol):
                    break


    def get_graph_stats(self):

        tmp = [d['distance'] for u,v,d in self.G.edges(data=True) if self.G.nodes[u]['pred'] != self.G.nodes[v]['pred']]

        mean_nodes_dist = np.round(np.mean(tmp),2)
        std_nodes_dist = np.round(np.std(tmp),2)

        return mean_nodes_dist, std_nodes_dist


    def log_graph_stats(self):

        mean_nodes_dist, std_nodes_dist = self.get_graph_stats()

        logging.info(" Number of edges for G: "+str(self.G.number_of_edges()))
        logging.info(" Mean distance between nodes: "+str(mean_nodes_dist)+' +/- '+str(std_nodes_dist))


    def add_node(self, id_node, pred, features, Xtrain_origin, touchpoint):
        self.G.add_node(id_node, pred=pred, features=features, Xtrain_origin=Xtrain_origin, touchpoint=touchpoint)


    def refine_graph(self):

        self.n_epochs += 1
        logging.info("\n ### EPOCH #"+str(self.n_epochs)+" ### \n")

        ## Step 1: get all the edges that can be refined without breaking the order of edge refinement (don't split an edge smaller than an edge resulting from the split of the biggest edge to split from the batch)

        to_refine = []

        # First iteration - to avoid the test at each loop
        e = heappop(self.h)
        to_refine.append(e)
        threshold = e[0]/2

        # Check which edges can be included in this batch of refinement to preserve the order of processing (longest edges first)
        while self.h:
            e = heappop(self.h)
            if e[0]>threshold:
                heappush(self.h, (e[0], (e[1][0],e[1][1])))
                break
            else:
                to_refine.append(e)

        to_refine = np.array(to_refine, dtype=object)
        
        ## Step 2: pre-compute the positions of the new nodes in the feature space, their blackbox' label and their distance to their nodes (edge being split)
        
        u, v, w, Xtrain_origin = [],[],[],[]
        for i in range(len(to_refine)):
            u.append(self.G.nodes[to_refine[i,1][0]]['features'])
            v.append(self.G.nodes[to_refine[i,1][1]]['features'])
            w.append((u[-1]+v[-1])/2)

            if self.G.nodes[to_refine[i,1][0]]['Xtrain_origin'] is None and self.G.nodes[to_refine[i,1][1]]['Xtrain_origin'] is None:
                Xtrain_origin.append([to_refine[i,1][0], to_refine[i,1][1]])

            elif self.G.nodes[to_refine[i,1][0]]['Xtrain_origin'] is None:
                Xtrain_origin.append(self.G.nodes[to_refine[i,1][1]]['Xtrain_origin'])

            else:
                Xtrain_origin.append(self.G.nodes[to_refine[i,1][0]]['Xtrain_origin'])

        u = np.array(u)
        v = np.array(v)
        w = np.array(w)

        w_preds = self.clf.predict(w)

        logging.info(" New label distribution: "+str(np.bincount(w_preds)))

        d_uw = np.linalg.norm(u-w, axis=1)
        d_vw = np.linalg.norm(v-w, axis=1)

        w = pd.DataFrame(np.array(w))
        
        ## Step 3: add the new nodes and edges to the graph and the heapq
        
        self.G.remove_edges_from(to_refine[:,1])

        new_nodes = np.array([(self.G.number_of_nodes()+i, {'touchpoint':True, 'pred':w_preds[i], 'features':w.iloc[i], 'Xtrain_origin':Xtrain_origin[i]}) for i in range(len(w))])
        self.G.add_nodes_from(new_nodes)

        new_edges1 = [(to_refine[i,1][0], new_nodes[i,0], {'distance':d_uw[i]}) for i in range(len(new_nodes))]
        new_edges2 = [(to_refine[i,1][1], new_nodes[i,0], {'distance':d_vw[i]}) for i in range(len(new_nodes))]

        new_edges = np.array(new_edges1+new_edges2)

        self.G.add_edges_from(new_edges)

        # add new edges to heapq
        for i in range(len(new_edges)):
            u = new_edges[i][0]
            v = new_edges[i][1]
            if self.G.nodes[u]['pred'] != self.G.nodes[v]['pred']:
                heappush(self.h, (-new_edges[i][2]['distance'], (new_edges[i][0],new_edges[i][1])))
            else:
                # This node (closest point of the new node with the same class on the edge) is not a touchpoint anymore
                self.G.nodes[u]['touchpoint'] = False 

        self.log_graph_stats()

        return 


    def plot_db(self):
        """
        /!/ right now in 2d - add t-sne or something for for D ?
        """

        x_min = self.Xtrain.iloc[:,0].min()
        x_max = self.Xtrain.iloc[:,0].max()
        y_min = self.Xtrain.iloc[:,1].min()
        y_max = self.Xtrain.iloc[:,1].max()
        x_step = y_step = 0.005

        x = np.arange(x_min, x_max, x_step)
        y = np.arange(y_min, y_max, y_step)
        xx, yy = np.meshgrid(x, y)

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for c in np.unique(self.Ytrain):
            a = self.Xtrain[self.Ytrain.iloc[:,0]==c]
            plt.scatter(a.iloc[:,0], a.iloc[:,1])

        X_tmp = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
        preds = self.clf.predict(X_tmp)
        preds = preds.reshape((xx.shape[0], yy.shape[1]))
        plt.contour(xx,yy,preds)

        left, right = plt.xlim()
        plt.xlim((left*1.1, right*1.1))
        bottom, top = plt.ylim()
        plt.ylim((bottom*1.1, top*1.1))


    def plot_m2g(self):
        """
        /!/ right now in 2d - add t-sne or something for for D ?
        """

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        tmp = [[self.G.nodes[u]['features'], self.G.nodes[v]['features']] for u,v,d in self.G.edges(data=True) if self.G.nodes[u]['pred'] != self.G.nodes[v]['pred']]
        tmp = np.array(tmp).reshape(-1,2)

        preds = self.clf.predict(tmp)
        for c in np.unique(preds):
            plt.scatter(tmp[:,0][preds==c], tmp[:,1][preds==c], marker='x')


class pool2graph:

    def __init__(self, Xtrain, Ytrain, pool):
        """
        Xtrain: pd.DataFrame, training set's input
        Ytrain: pd.DataFrame, training set's target
        pool: Pool object, pool of trained models
        """

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.pool = pool


    def fit(self, max_epoch=10, early_stopping=True, tol=0.05):
        
        self.Gs = {}

        # 1. Generate a model2graph for every model
        for key, model in self.pool.models.items():

            logging.info("\n ### Processing model: "+str(key)+" ### \n")

            m2g = model2graph(self.Xtrain, self.Ytrain, model)
            m2g.fit(early_stopping=early_stopping, max_epoch=max_epoch, tol=tol)

            self.Gs[key] = m2g


        # 2. Generate a graph that summarizes discrepancies
        self.G = nx.Graph()

        # 2.1 Get all nodes generated from the training set

        # Use any m2g to get data shared across all the m2g
        _m2g = self.Gs[list(self.Gs.keys())[0]]

        Xtrain_pairs = [(n1,n2) for n1,n2 in combinations(_m2g.Xtrain_nodes, 2) if (n1 != n2)]

        # 2.2 For every pair of nodes generated from the training set

        for n1,n2 in Xtrain_pairs[:]:

            Xtrain_pair_touchpoints_features = np.empty(shape=(0,_m2g.Xtrain.shape[1]))
            Xtrain_pair_touchpoints_models = []

            # 2.2.1 For every models in the pool
            for k,m2g in self.Gs.items():

                # 2.2.2 Get all the touchpoints between the pair of nodes
                _Xtrain_pair_touchpoints = [x for x,y in m2g.G.nodes(data=True) if y['Xtrain_origin']==[n1,n2] and y['touchpoint']==True]

                if len(_Xtrain_pair_touchpoints)==0:
                    continue

                tmp = np.array([m2g.G.nodes[n]['features'] for n in _Xtrain_pair_touchpoints])
                
                Xtrain_pair_touchpoints_features = np.concatenate((Xtrain_pair_touchpoints_features, tmp), axis=0)

                Xtrain_pair_touchpoints_models = Xtrain_pair_touchpoints_models+ [k]*len(tmp)

            # Add all the touchpoints between the pair on an edge in the graph
            self.G.add_node(n1, features=m2g.G.nodes[n1]['features'])
            self.G.add_node(n2, features=m2g.G.nodes[n2]['features'])

            Xtrain_pair_touchpoints_models = np.array(Xtrain_pair_touchpoints_models)
            tmp = [(n1,n2, {'Xtrain_pair_touchpoints_models':Xtrain_pair_touchpoints_models, 'Xtrain_pair_touchpoints_features':Xtrain_pair_touchpoints_features, 'distance':m2g._euclidean_distances_X[n1,n2]})]

            self.G.add_edges_from(tmp)


    def get_local_discrepancies(self, i, k=10):
        """
        Return the discrepancies for an instance.
        Starting from the instance to explain, get k closest points of the training set and retrieve the discrepancies along the segments (edges)

        For now, i is the index in Xtrain. Then, the method should accept an instance and either (1) attached it to the closest point of the training set or (2) learn specific edges for it
        """

        pairs = np.array(list(self.G.edges(i)))
        pairs_dist = [self.G.edges[(i,j)]['distance'] for i,j in self.G.edges(i)]
        pairs_k = pairs[np.argsort(pairs_dist)][:k]

        ax = plt.subplot()

        for pair in pairs_k:

            Xtrain_pair_touchpoints_features = self.G.edges[pair]['Xtrain_pair_touchpoints_features']

            u = self.G.nodes[pair[0]]['features']
            v = self.G.nodes[pair[1]]['features']

            u_closest = Xtrain_pair_touchpoints_features[np.argsort(euclidean_distances(u.to_frame().T, Xtrain_pair_touchpoints_features))][0][0]

            v_closest = Xtrain_pair_touchpoints_features[np.argsort(euclidean_distances(v.to_frame().T, Xtrain_pair_touchpoints_features))][0][0]

            nodes = pd.concat((u,v), axis=1).T
            touchpoints = pd.DataFrame(np.array([u_closest,v_closest]))

            nodes.plot(kind='scatter', x=0, y=1, ax=ax)
            touchpoints.plot(kind='scatter', x=0, y=1, c='r', ax=ax)