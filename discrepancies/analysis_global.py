from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances


#from sklearn.tree import DecisionTreeClassifier
from sktree.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, recall_score, precision_score



class GlobalDiscrepancyAnalyzer:
    
    
    def __init__(self, p2g, pool, intervals, X, categorical_names=None):
        self.intervals = intervals
        self.p2g = p2g
        self.categorical_names = categorical_names
        self.continuous_names = [c for c in X.columns if c not in categorical_names]
        self.X = X.copy()
        self.pool = pool
        
        self._build_amplitude_dataset()
        self._preprocess_input_data()
        self._build_discrepancy_nodes_dataset()
        self._build_nodes_dataset()
        
        

    def _build_amplitude_dataset(self):
        """
        Output dataset: one row for each interval 
        Continuous features columns: how much was moved in the feature
        Categorical features columns: Value of the category (intervals do not move along categorical features)
        """
        out = []
        continuous_changes = pd.DataFrame([di.border_features.T.iloc[:,0] - di.border_features.T.iloc[:,1] for di in self.intervals]) # could be faster is we change how intervals are stored
        continuous_changes = continuous_changes[self.continuous_names].abs()
       
        if len(self.categorical_names) > 0:
            categorical_coord = pd.DataFrame([di.border_features[self.categorical_names].iloc[0, :] for di in self.intervals])
            self.X[self.categorical_names] = (self.X[self.categorical_names] > 0).astype('int')

            self.amplitude_dataset = continuous_changes.join(categorical_coord.reset_index(drop=True))
        else:
            self.amplitude_dataset = continuous_changes
        print("Intervals amplitude dataset (self.amplitude_dataset): shape", self.amplitude_dataset.shape)
    
    
    def _preprocess_input_data(self):
        """
        Preprocess:
            - Categorical features in 0-1
        """
        if len(self.categorical_names)  > 0:
            self.X[self.categorical_names] = (self.X[self.categorical_names] > 0).astype('int')
            
        self.X["discrepancies"] = self.pool.predict_discrepancies(self.X).values
        print("Input data preprocessed (self.X): shape", self.X.shape)
    

    def _build_discrepancy_nodes_dataset(self):
        """
        Output dataset: one row for each node that was in a discrepancy area.
        """
        #self.disc_nodes_dataset = [n[1]['features'] for n in self.p2g.G.nodes(data=True) if n[1]['discrepancies'] == 1]
        #self.disc_nodes_dataset = pd.DataFrame(self.disc_nodes_dataset)
        
        self.disc_nodes_dataset = self.p2g.get_nodes(discrepancies=True)
        self.disc_nodes_dataset = pd.DataFrame([n[1]['features'] for n in self.disc_nodes_dataset], index=[n[0] for n in self.disc_nodes_dataset])
        if len(self.categorical_names) > 0:
            self.disc_nodes_dataset[self.categorical_names] = (self.disc_nodes_dataset[self.categorical_names] > 0).astype('int')
            
        print("Discrepancy nodes dataset (self.nodes_dataset): shape", self.disc_nodes_dataset.shape)
            
    def _build_nodes_dataset(self):
        """
        Output dataset: one row for each node that was in a discrepancy area.
        """
        
        nodes_data = self.p2g.G.nodes(data=True)
        self.nodes_dataset = [n[1]['features'] for n in nodes_data]
        self.nodes_dataset = pd.DataFrame(self.nodes_dataset, index=[n[0] for n in nodes_data])
        if len(self.categorical_names) > 0:
            self.nodes_dataset[self.categorical_names] = (self.nodes_dataset[self.categorical_names] > 0).astype('int')
            
        discs = np.array([n[1]['discrepancies'] for n in nodes_data])
        self.nodes_dataset["discrepancies"] = discs
        print("Nodes dataset (self.nodes_dataset): shape", self.nodes_dataset.shape)
        
     
    
    #######################
    # MACROS FOR ANALYSIS #
    #######################
    
    def get_closest_intervals_from_x(self, x, intervals, k=1, parallel=False):
    
        n_jobs = -1 if parallel==True else 1
    
        distances = pairwise_distances(x.values.reshape(1, -1), self.disc_nodes_dataset, n_jobs=n_jobs)
        idx_closest = self.disc_nodes_dataset.iloc[np.argsort(-1 * distances)[0,:], :].index 
        closest_nodes_iter = iter(idx_closest)
        interval_list = []
        while len(interval_list) < k:
            intclos = next(closest_nodes_iter)

            for ci in intervals:
                if intclos in ci.path:
                    interval_list.append(ci.border_features)
                if len(interval_list) == k:
                    break
        return interval_list
    
    def local_intervals(self, instance, intervals, scaler, k=1, savefig=False, savefolder='results', savefname='localinterval', parallel=False):
        sns.set_style("white")

        Fblue = '#07519e'
        Fblue2 = '#2ba8e0'
        Fgreen = '#47DBCD'
        Fpink = '#F3A0F2'
        Fpurple = '#83236a'
        Fviolet = '#661D98'
        Famber = '#F5B14C'
        
        closest_k_intervals = self.get_closest_intervals_from_x(instance, intervals, k, parallel)
        aggregatedf = pd.concat(closest_k_intervals)
        interval_relevant = pd.DataFrame([aggregatedf.max(), aggregatedf.min()])

        #interval_scaled = pd.DataFrame(scaler.inverse_transform(interval_relevant))
        #interval_scaled.columns = X_train.columns
        #amplitude = pd.DataFrame(interval_scaled.iloc[0, :] - interval_scaled.iloc[1, :])
        #amplitude_normalized = interval_relevant.iloc[0, :] - interval_relevant.iloc[1, :]
        
        cont_names = self.continuous_names
        cat_names = self.categorical_names
        
        # centered
        interval_relevant2 = interval_relevant - instance
        
        
        Xcols = self.X.columns[:-1]
        instance_orig_space = pd.DataFrame(scaler.inverse_transform(instance.values.reshape(1, -1)), columns=Xcols)

        
        XMIN = (self.X[cont_names] - instance[cont_names]).min().min()
        XMAX = (self.X[cont_names] - instance[cont_names]).max().max()

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))


        cont_names = [f for f in Xcols if f in cont_names]

        for i in range(len(cont_names)):
            #DISCREPANCY INTERVALS
            DISCR_LEFT = float(interval_relevant2[cont_names[i]].min())
            DISCR_RIGHT = float(interval_relevant2[cont_names[i]].max())
            DISCR_WIDTH = DISCR_RIGHT - DISCR_LEFT

            ### Left
            if i == 0:
                ax1.barh(y=i,left=XMIN-10.0, height=1.0, 
                         width=DISCR_LEFT-(XMIN-10.0), 
                         color="#118AB2", edgecolor='k', alpha=0.1, label="Unanimous prediction: 0")

                ### Right
                ax1.barh(y=i,left=DISCR_RIGHT, height=1.0, 
                         width=DISCR_RIGHT+30.0, 
                         color="#EF476F", edgecolor='k', alpha=0.1, label="Unanimous prediction: 1")
                ### Middle
                ax1.barh(y=i,left=DISCR_LEFT, height=1.0, 
                         width=DISCR_WIDTH, 
                         color="#06D6A0", edgecolor='k', alpha=0.7, label="Discrepancy Interval")

            else:
                ### Left
                ax1.barh(y=i,left=XMIN-10.0, height=1.0, 
                         width=DISCR_LEFT-(XMIN-10.0), 
                         color="#118AB2", edgecolor='k', alpha=0.1)

                ### Right
                ax1.barh(y=i,left=DISCR_RIGHT, height=1.0, 
                         width=DISCR_RIGHT+30.0, 
                         color="#EF476F", edgecolor='k', alpha=0.1)
                ### Middle
                ax1.barh(y=i,left=DISCR_LEFT, height=1.0, 
                         width=DISCR_WIDTH, 
                         color="#06D6A0", edgecolor='k', alpha=0.7)



            #coordonnée instance
            ax1.text(x=XMIN-0.9, y=i-0.1, 
                     s=round(float(instance_orig_space.iloc[:, i]), 4), color='black',
                    rotation=-0, fontsize=12)

            #min interval
            ax1.text(x=float(interval_relevant2[cont_names[i]].min()) - 0.6, y=i+0.25, 
                     s=round(pd.DataFrame(scaler.inverse_transform(interval_relevant), columns=Xcols)[cont_names[i]].min(), 2),
                     color="#118AB2", rotation=-0, fontsize=10)
            #max interval
            ax1.text(x=float(interval_relevant2[cont_names[i]].max()) + 0.1, y=i+0.25, 
                     s=round(pd.DataFrame(scaler.inverse_transform(interval_relevant), columns=Xcols)[cont_names[i]].max(), 2),
                     color="#EF476F", rotation=-0, fontsize=10)

        # context: training ranges
        tmp = self.X[cont_names] - instance[cont_names]
        tmp = tmp.melt()
        matchcols = {v:i for i, v in enumerate(cont_names)}
        tmp.variable = tmp.variable.apply(lambda x: matchcols[x]).astype("str")
        sns.violinplot(data=tmp, y="variable", x="value", color="#F7E9CF", alpha=1.0, edgecolor=None, inner=None,
                       ax=ax1)

        ax1.set_yticks(ticks=range(len(cont_names)))   
        contlabels = cont_names
        ax1.set_yticklabels(labels=contlabels, fontsize=13, weight="bold")
        ax1.set_xticks([])
        #ax1.set_xlim((interval_relevant2[cont_names].values.min()-2.0, interval_relevant2[cont_names].values.max()+2.0))
        ax1.vlines(x=0, ymin=-1.5, ymax = len(cont_names)+1.5, color='black', linewidth=6, alpha=1, label="Instance $x_0$ coordinates")


        ax1.set_xlim( XMIN - 1.0,
                     XMAX + 1.0)
        ax1.set_ylim((-0.5, i+0.5))


        ax1.legend(fontsize=12)
        ax1.set_xlabel(None)
        ax1.set_ylabel(None)

        plt.tight_layout()
        if savefig == True:
            plt.savefig('../../' + savefolder + '/' + savefname + '.pdf')
        plt.show()

        ###TABLE
        same_cat = list(np.array(cat_names)[np.where((interval_relevant[cat_names]>=0).astype('int').mean() * instance[cat_names])])
        variable_names= ["credit_history", "checking_status", "saving_status", "employment", 
                         "purpose", "personal_status", "own_telephone" ,"foreign_worker",
                        "other_parties","property_magnitude", "housing", "other_payments", "job"]

        out = []
        for v in same_cat:
            for v2 in variable_names:
                if v2 in v:
                    _, mod = v.split(v2+'_')
                    out.append(['$\mathbf{%s}$'%v2.replace('_', '\_'), mod])

        out = np.array(out)
        colors_array = np.array(["#06D6A0", "#06D6A0"]*out.shape[0]).reshape(out.shape[0], 2)
        table = ax2.table(cellText=out,
                               #colWidths = [1.0, 1.0],
                               rowLabels=None,
                               loc='center',
                          cellColours=colors_array)
        for cell in table._cells:
            table._cells[cell].set_alpha(.1)
        table.set_fontsize(14)
        table.scale(1.0, 1.5)
        ax2.axis("off")

        plt.tight_layout()

        #plt.savefig('../../results/interval_german_withcat5.pdf')
        plt.show()
            
            

    
    
    def get_global_discrepancy_importances(self, min_expo=0):
        """
        Returns two feature rankings of how represented the feature is in the discrepancy intervals:
        one for continuous and one for categorical attributes.
        - for continuous features, the average amplitude of change is measured (e.g. Age-> 5yrs of age in average)
        - for categorical features, the representation of the feature is returned (e.g. 80% of discrepancy intervals were found for Males), normalized by the presence in the Xtrain
        N.B. rankings cannot be compared
        """
        out_dict = {}
        
        out_dict['continuous_features'] = self.amplitude_dataset[self.continuous_names].mean(axis=0).sort_values(ascending=False)
        
        if len(self.categorical_names) > 0:
            ### Maybe a class parameter? Or do we want to change it everytime ?
            features_with_min_expo = [c for c in self.categorical_names if self.X.sum(axis=0)[c] >= min_expo]
            
            out_dict['categorical_features'] = ((self.amplitude_dataset[features_with_min_expo].mean(axis=0) - self.X[features_with_min_expo].mean(axis=0)) /(self.X[features_with_min_expo].mean(axis=0))).sort_values(ascending=False)
        
        self.feature_importances = out_dict
        
        
        return out_dict
    
    def plot_feature_importances(self, palettes=["rocket", "rocket"], savefig=False, fname=None):
        plt.figure(figsize=(len(self.feature_importances['continuous_features']),4))
        plt.xticks(rotation=45)
        sns.barplot(data=pd.DataFrame(self.feature_importances['continuous_features']).T, palette=palettes[0])
        plt.title("Ranking for continuous features: average range of intervals")
        plt.tight_layout()
        plt.show()
        
        if 'categorical_features' in self.feature_importances: 
            plt.figure(figsize=(len(self.feature_importances['categorical_features']),4))
            plt.xticks(rotation=45)
            sns.barplot(data=pd.DataFrame(self.feature_importances['categorical_features']).T, palette=palettes[1])
            plt.title("Ranking for categorical features: normalized exposition")
            plt.tight_layout()
            plt.show()
        
            
    def get_discrepancy_segments(self, X_exposition=None, y_exposition=None, min_expo=0, min_purity=0.0, min_purity_expo=0.0):
        #inspired from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        
        prediction_X_expo = self.pool.predict(X_exposition)

        if len(self.categorical_names) > 0:
            X_exposition[self.categorical_names] = (X_exposition[self.categorical_names] > 0).astype('int')
        
        cols = [c for c in self.X.columns if c != "discrepancies"]
        node_train = np.array(self.nodes_dataset[cols])
        node_train = np.array(self.nodes_dataset[cols].append(X_exposition[cols]))
        node_y = self.pool.predict_discrepancies(node_train).values
        
        dt = DecisionTreeClassifier(min_samples_expo=min_expo)
        dt.fit(node_train, node_y, X_expo=X_exposition)
        #self.cart_discrepancies = dt
        
        #a enlever?
        y_disc_exposition = self.pool.predict_discrepancies(X_exposition)
        print('Discrepancy segments tree accuracy on nodes:', f1_score(node_y, dt.predict(node_train)))
        print('... on exposition data', f1_score(y_disc_exposition, dt.predict(X_exposition)))
        
        feature = dt.tree_.feature
        threshold = dt.tree_.threshold
        
        node_leaves = dt.apply(node_train)
        leaves_expo = dt.apply(X_exposition)
        treenode_indicator = dt.decision_path(node_train)
        
        
        # get the decision path of one instance of each leaf
        leaves_list = list(set(node_leaves))
        representants = {}
        
        self.leaf_found = []
        # iteration over leaves
        for leaf_index in leaves_list:
            
            #get exposition in segment                  
            X_segment_expo = X_exposition.iloc[np.where(leaves_expo == leaf_index)[0], :]
            y_segment_expo = y_exposition.iloc[np.where(leaves_expo == leaf_index)[0]]
            
            segment_exposition = X_segment_expo.shape[0]/ X_exposition.shape[0]
            segment_purity = self.pool.predict_discrepancies(node_train[np.where(node_leaves == leaf_index)[0], :]).mean()
            
            try:
                segment_purity_expo = self.pool.predict_discrepancies(X_segment_expo).mean()
            except ValueError: #if no expo in segment (may happen if min_expo =0)
                segment_purity_expo = 0.0
                
            if ((segment_purity < min_purity) or
                (segment_purity_expo < min_purity_expo)):
                continue
            
            # get accuracy of clf in segment on expo
            #preds_segment_expo = self.pool.predict(X_segment_expo) #PROBLEME ICI CAR X_expo categorielles ont ete mises à 0;1
            preds_segment_expo = prediction_X_expo.iloc[np.where(leaves_expo == leaf_index)[0], :]
            segment_accuracy_expo = {c: f1_score(y_segment_expo, preds_segment_expo[c]) for c in preds_segment_expo.columns}
            
            # get accuracy of clf in segment on expo over preds with discrepancies
            #preds_segment_expo = prediction_X_expo.iloc[np.where((leaves_expo == leaf_index) * )[0], :]
            #segment_accuracy_expo = {c: f1_score(y_segment_expo, preds_segment_expo[c]) for c in preds_segment_expo.columns}

            # get segment size
            n_nodes_segment = len(np.where(node_leaves == leaf_index)[0]) / node_leaves.shape[0]
            n_disc_nodes_segment = len(np.where((node_leaves == leaf_index) & (node_y == True))[0]) / (node_y == True).sum()
            
            
            
            # get segment path
            #get one representative
            leaf_representant_id = np.where(node_leaves == leaf_index)[0][0] #first one
            leaf_representant_features = node_train[leaf_representant_id, :]
            # get list of tree nodes activated by the representant
            treenode_index = treenode_indicator.indices[treenode_indicator.indptr[leaf_representant_id]: treenode_indicator.indptr[leaf_representant_id + 1]]
            
            self.leaf_found.append(leaf_index)
            print("====== SEGMENT {segment} ======".format(segment=leaf_index))
            print("=== Segment description:")
            for treenode_id in treenode_index:
                  # continue to the next node if it is a leaf node
                if node_leaves[leaf_representant_id] == treenode_id:
                      continue
                        
                # check if value of the split feature for sample 0 is below threshold
                if (node_train[leaf_representant_id, feature[treenode_id]] <= threshold[treenode_id]):
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                print("decision node {node} : {feature} "
                    "{inequality} {threshold})".format(
                              node=treenode_id,
                              feature=self.X.columns[feature[treenode_id]],
                              inequality=threshold_sign,
                              threshold=threshold[treenode_id]))
                  
            print("=== Segment characteristics")
            print("Segment exposition: {expo}".format(expo=segment_exposition))
            print("Segment node population (proxy for size?): {n_nodes}".format(n_nodes=n_nodes_segment))
            print("Percent of the discrepancy nodes contained here: {n_nodes}".format(n_nodes=n_disc_nodes_segment))
            print("Segment purity: {purity}".format(purity=segment_purity))
            print("Segment purity (X_expo): {purity_expo}".format(purity_expo=segment_purity_expo))
            print("Accuracy of classifiers (F1 on X_expo) on segment: {acc_segment}".format(acc_segment=segment_accuracy_expo))
        print("Number of discrepancy segments found: %i"%len(self.leaf_found))
        
        #save tree (useful for other stuff...not clean). it was saved twice!
        self.segments_tree = dt
            
              
    """def plot_tsne(self, n_samples=1000, ):
        data = gda.disc_nodes_dataset.iloc[:n_samples, :]
        data_leaves = self.cart_discrepancies.apply(data)
        
        palette = ["red", "blue", "green", "purple", "orange", "yellow", "lime", "cyan"] 
        
        #TODO
        is_segment = [
        colors = [palette[i] for i
        
        tsne = TSNE(perplexity=30, n_jobs=-1).fit_transform(data)
        tsne = pd.DataFrame(tsne, columns=["Dim 0", "Dim 1"])
        return 2"""


        
        
def get_tree_path(dt, representant):
    
    path = dt.decision_path(representant.reshape(1, -1))
    
    node_index = path.indices[path.indptr[sample_id]:path.indptr[sample_id + 1]]

    return 1
    

def get_segment_characteristics(segment_index, dt, X_expo):    
    return 1
    



    
    
    
    
    