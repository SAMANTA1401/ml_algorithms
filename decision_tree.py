from collections import Counter
import numpy as np


class DecisionTree:
    def __init__(self, max_depth, min_node_size):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.final_tree = {}



    def calculate_gini(self, child_nodes):
        n = 0
        # Calculate number of all instances of the parent node
        for node in child_nodes:
            n = n + len(node)
        gini = 0
        # Calculate gini index for each child node
        for node in child_nodes:
            m = len(node)

            # Avoid division by zero if a child node is empty
            if m == 0:
                continue

            # Create a list with each instance's class value
            y = []
            for row in node:
                y.append(row[-1])

            # Count the frequency for each class value
            freq = Counter(y).values()
            node_gini = 1
            for i in freq:
                node_gini = node_gini - (i / m) ** 2
            gini = gini + (m / n) * node_gini
        return gini


    def apply_split(self, feature_index, threshold, data):
        instances = data.tolist()
        left_child = []
        right_child = []
        for row in instances:
            if row[feature_index] < threshold:
                left_child.append(row)
            else:
                right_child.append(row)
        left_child = np.array(left_child)
        right_child = np.array(right_child)
        return left_child, right_child



    def find_best_split(self, data):
        num_of_features = len(data[0]) - 1
        gini_score = 1000
        f_index = 0
        f_value = 0
        # Iterate through each feature and find minimum gini score
        for column in range(num_of_features):
            for row in data:
                value = row[column]
                l, r = self.apply_split(column, value, data)
                children = [l, r]
                score = self.calculate_gini(children)
                # print("Candidate split feature X{} < {} with Gini score {}".format(column,value,score))
                if score < gini_score:
                    gini_score = score
                    f_index = column
                    f_value = value
                    child_nodes = children
        # print("Chosen feature is {} and its value is {} with gini index {}".format(f_index,f_value,gini_score))
        node = {"feature": f_index, "value": f_value, "children": child_nodes}
        return node


    def calc_class(self, node):
        # Create a list with each instance's class value
        y = []
        for row in node:
            y.append(row[-1])
        # Find most common class value
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]



    def recursive_split(self, node, depth):
        l, r = node["children"]
        del node["children"]
        if l.size == 0:
            c_value = self.calc_class(r)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        elif r.size == 0:
            c_value = self.calc_class(l)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        # Check if tree has reached max depth
        if depth >= self.max_depth:
            # Terminate left child node
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
            # Terminate right child node
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
            return
        # process left child
        if len(l) <= self.min_node_size:
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
        else:
            node["left"] = self.find_best_split(l)
            self.recursive_split(node["left"], depth + 1)
        # process right child
        if len(r) <= self.min_node_size:
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
        else:
            node["right"] = self.find_best_split(r)
            self.recursive_split(node["right"], depth + 1)

    

    def train(self, X):
        # Create initial node
        tree = self.find_best_split(X)
        # Generate the rest of the tree via recursion
        self.recursive_split(tree, 1)
        self.final_tree = tree
        return tree


    def print_dt(self, tree, depth=0):
        if "feature" in tree:
            print(
                "\nSPLIT NODE: feature #{} < {} depth:{}\n".format(
                    tree["feature"], tree["value"], depth
                )
            )
            self.print_dt(tree["left"], depth + 1)
            self.print_dt(tree["right"], depth + 1)
        else:
            print(
                "TERMINAL NODE: class value:{} depth:{}".format(
                    tree["class_value"], tree["depth"]
                )
            )


    def predict_single(self, tree, instance):
        if not tree:
            print("ERROR: Please train the decision tree first")
            return -1
        if "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                return self.predict_single(tree["left"], instance)
            else:
                return self.predict_single(tree["right"], instance)
        else:
            return tree["class_value"]

    def predict(self, X):
        y_predict = []
        for row in X:
            y_predict.append(self.predict_single(self.final_tree, row))
        return np.array(y_predict)