import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.
    
    Returns:
        The root node of the decision tree.
    """
    zero = DecisionNode(None, None, None, 0)
    one = DecisionNode(None, None, None, 1)
    two  = DecisionNode(None, None, None, 2)
    R3 = DecisionNode(zero,one,lambda feature: feature[3] <= -0.8276, None)
    L1 = DecisionNode(R3,zero,lambda feature: feature[0] >= .8096, None)
    R2 = DecisionNode(zero,two,lambda feature: feature[0] <= -2.9350, None)
    dt_root = None
    dt_root = DecisionNode(None, None, lambda feature: feature[2] <= -.7045, None)     
    dt_root.left = R2
    dt_root.right = L1
    return dt_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    c_matrix = [[0]*n_classes for _ in range(n_classes)]
    
    for i in range(len(classifier_output)):
        feat = int(classifier_output[i])
        if classifier_output[i] == true_labels[i]:
            n = feat
            c_matrix[n][n] = c_matrix[n][n]+1
        else:
            r = true_labels[i]
            c = feat
            c_matrix[r][c] = c_matrix[r][c] + 1

    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """
    precision = [0] * n_classes
    pe_matrix = confusion_matrix(true_labels,classifier_output,n_classes)
    col_totals = [ sum(x) for x in zip(*pe_matrix)]
    for i in range(n_classes):        
        if col_totals[i]!=0:
            precision[i] = pe_matrix[i][i]/col_totals[i]
    return precision

def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """
    recall = [0] * n_classes
    pe_matrix = confusion_matrix(true_labels,classifier_output,n_classes)
    row_totals = [ sum(x) for x in pe_matrix ]
    for i in range(n_classes):
        if row_totals[i]!=0:
            recall[i] = pe_matrix[i][i]/row_totals[i]
    return recall


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
    total = 0
    pe_matrix = confusion_matrix(true_labels,classifier_output,n_classes)
    row_totals = [ sum(x) for x in pe_matrix ]
    num = 0
    for i in range(n_classes):
        num = pe_matrix[i][i] + num
        total = total + row_totals[i]
    if total!=0:
        return num/total
    else:
        return 0


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """
    sum = 0
    total = 0
    gini = 0
    counts ={}
    for i in range(len(class_vector)):
        val = class_vector[i] 
        sum = sum + 1
        if val in counts:
            counts[val] = counts[val] + 1
        else:
            counts[val] = 1

    keys = list(counts.keys())
    for i in range(len(keys)):
        if sum!=0:
            num = counts[keys[i]]/sum
            total = total + (num * num)
    
    gini = 1 - total
    return gini


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """
    length = len(current_classes)
    sum = [0] * length
    sum_zero = [0] * length
    sum_one = [0] * length
    ig = [0] * length
    total = 0
    counts ={}

    for i in range(length):
        for j in range(len(current_classes[i])):
            val = current_classes[i][j] 
            sum[i] = sum[i] + 1
            if val in counts:
                counts[val][i] = counts[val][i] + 1
            else:
                counts[val] = [0]*length
                counts[val][i] = 1
    prev_counts = {}
    for n in range(len(previous_classes)):
        val = previous_classes[n]
        total = total + 1
        if val in prev_counts:
            prev_counts[val] = prev_counts[val] + 1
        else:
            prev_counts[val] = 1


    gini_total = 0
    for y in range(length):
        gini_total = gini_total + ((sum[y]/total) *gini_impurity(current_classes[y]))

    return gini_impurity(previous_classes) - gini_total


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        if features.shape[0] == 0:
            return DecisionNode(None,None,None,None)

        uniq = Counter(classes)
        if len(uniq) == 1:
            return DecisionNode(None, None, None, classes[0])
        if depth == self.depth_limit:
            max_num = 0
            freq1 = uniq[1]
            freq0 = uniq[0]
            if freq1 >= freq0:
                max_num = 1
            max_num = int(max(uniq, key=uniq.get))
            return DecisionNode(None, None, None, max_num)
        
        means = []
        disc_feats = []
        gains = []
        max_gain = -10000000
        alpha_best_index = 0
        for i in range(len(features[0])):

            mean = np.mean(features[:,i])
            curr = np.where(features[:,i]>=mean,1,0)
            means.append(mean)
            if disc_feats == []:
                disc_feats = curr
            else:
                 disc_feats = np.column_stack((disc_feats, curr))
            alpha = []
            disc_count = Counter(curr)
            ks = disc_count.keys()
            for k in ks:
                temp = []
                for x in range(len(curr)):
                    if curr[x] == k:
                        temp.append(classes[x])               
                alpha.append(temp)
            gain = gini_gain(classes, alpha)
            gains.append(gain)
            if gain > max_gain:
                max_gain = gain
                alpha_best_index = i

        sum_gains = sum(gains)

        if sum_gains== 0:
            return DecisionNode(None, None, None, 0)
        gains = np.divide(gains, sum_gains)

        alpha_best = disc_feats[:,alpha_best_index]

        zeros = []
        ones = []
        classes_zero =[]
        classes_one =[]
        for i in range(len(alpha_best)):
            if alpha_best[i]==0:
                zeros.append(i)
                classes_zero.append(classes[i])
            elif alpha_best[i]==1:
                ones.append(i)                
                classes_one.append(classes[i])


        depth = depth + 1
        
        split_left = np.delete(features,zeros,0)
        split_right = np.delete(features,ones,0)
        left = self.__build_tree__(split_left, classes_one, depth)
        right = self.__build_tree__(split_right, classes_zero, depth)

        return DecisionNode(left, right, lambda feature: feature[alpha_best_index] > means[alpha_best_index])



    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for i in range(len(features)):
            class_labels.append(self.root.decide(features[i]))
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    folds = [0]*k
    arrsize = 0
    leftOver = 0

    arrSize = int(dataset[0].shape[0]/k)
    length =  len(dataset[1])
    for i in range(k):
        classes = dataset[1]
        curr_classes = []
        features = dataset[0]
        curr_features = []

        for j in range(arrSize):
            rem_index = np.random.randint(0, len(classes) - 1)
            curr_features.append(features[rem_index])
            features = np.delete(features, rem_index, 0)
            elem = int(classes[rem_index])
            classes  = np.delete(classes, rem_index, 0)
            curr_classes.append(elem)

        testing = (curr_features, curr_classes)
        training = (features, classes)
        folds[i] = (training, testing)

    return folds  


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=200, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.trees_inds = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        data = np.column_stack((features,classes))
        num_features = len(features[0])
        feature_samples = int(self.attr_subsample_rate * num_features)

        
        length = len(classes)
        samples = int(self.example_subsample_rate * length)

        for i in range(0,self.num_trees):
            np.random.shuffle(data)
            feats = data[:,0:-1]
            cls = data[:,-1]
            training = [0] * samples
            testing = [0] * samples
            for s in range(samples):
                r = np.random.randint(0, length - 1)
                training[s] = list(feats[r,:])
                testing[s] = cls[r]


            keep = []
            cols = list(range(num_features))
            for c in range(feature_samples):
                if cols == []:
                    break
                r = np.random.randint(0, num_features)
                while r not in cols:
                    r = np.random.randint(0, num_features)
                cols.remove(r)
                keep.append(r)                
            keep = np.sort(np.array(keep))

            self.trees_inds.append(keep)
            training = np.array(training)
            training = training[:,keep]
            tree = DecisionTree(self.depth_limit)
            tree.fit(training,testing)
            self.trees.append(tree)

    def classify(self, features):
            """Classify a list of features based on the trained random forest.
            Args:
                features (m x n): m examples with n features.
            Returns:
                votes (list(int)): m votes for each element
            """
            votes = []
            feats = np.array(features)
            labels = []
            trees = self.trees

            for i in range(self.num_trees):
                curr_tree_label = trees[i].classify(list(feats[:,self.trees_inds[i]]))
                labels.append(curr_tree_label)

            labels = np.array(labels)
            for i in range(len(features)):
                feat = labels[:,i]               
                unique_dict = Counter(feat)
                votes.append(int(max(unique_dict, key=unique_dict.get)))

            return votes


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        vectorized = (np.multiply(data, data) + data)
        return vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        vectorized = data[:100,:]
        sum = np.sum(vectorized, axis=1)
        max = np.amax(sum)
        index = np.where(sum == max)
        return max,index[0]

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = np.hstack(data)
        flattened = data
        positive = flattened[flattened>0]
        unique_dict = Counter(positive)
        return unique_dict.items()

    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        
        vectorized = data
        if dimension=='c':
            vectorized = np.hstack([vectorized,vector.reshape(vector.size, 1)])
        elif dimension=='r':
            vectorized = np.vstack([vectorized,vector])
        else:
            return vectorized
        
        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = data
        vectorized = np.where(vectorized >= threshold, vectorized, vectorized*vectorized)
        return vectorized


def return_your_name():
    # return your name
    return 'Shruti Mhasawade'
