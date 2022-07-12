import sys              # extracts the arguments
import pandas as pd     # reads the csv file and puts it in a df
import numpy as np      # allows to change the data type of the df
import math             # needed for math.log2
import random           # needed for random.randint

class Decision_Tree:
    class Decision_Tree_Node:
        # attribute - can be int (leaf), or can be a string attribute (tree node)
        def __init__(self, attribute, left = None, right = None):
            self.attribute = attribute
            self.left = left    # branch 0
            self.right = right  # branch 1

    def __init__(self, file, heruistic):
        self.heruistic = heruistic
        self.root = self.build_decision_tree(self.read_csv(file))

    # This reads the csv file and returns the data
    ## NOTE: Maybe it should have return columns?
    def read_csv(self, file):
        return pd.read_csv(file, dtype=bool)

    def print_tree_helper(self, node, string):
        # Scans through the left branch of the current node
        if (node.left and isinstance(node.left.attribute, str)):
            print(f'{string}{node.attribute} = 0 : ')
            self.print_tree_helper(node.left, string + '| ')
        else:
            print(f'{string}{node.attribute} = 0 : {node.left.attribute}')

        # Scans through the right branch of the current node
        if (node.right and isinstance(node.right.attribute, str)):
            print(f'{string}{node.attribute} = 1 : ')
            self.print_tree_helper(node.right, string + '| ')
        else:
            print(f'{string}{node.attribute} = 1 : {node.right.attribute}')       

    
    # Traverses through the tree and prints it out
    def print_tree(self):
        self.print_tree_helper(self.root, '')

    # reads the file and calculate the accuracy of the tree
    def test_tree(self, file):
        df = self.read_csv(file)
        num_correct = 0
        # iterate through each row (entry)
        for index in range(len(df)):
            # iterate through the decision tree
            curr_node = self.root
            while(curr_node.left != None and curr_node.right != None):
                # go down the path specified by current's node attribute and this index's value for the attribute
                curr_node = curr_node.left if df[curr_node.attribute][index] == 0 else curr_node.right
            if curr_node.attribute == df.iloc[index, -1]:
                num_correct += 1
        return num_correct/len(df)



    ########### DEBUGGING THE TREE ###########
    # does a bfs through the tree and prints it out
    def debug_tree(self):
        q = [self.root]
        while q:
            #print('-----------------------------------')
            temp = []
            for i in range(len(q)):
                curr = q.pop(0)
                if (curr.left != None and isinstance(curr.left.attribute, int) and curr.left.attribute != 0):
                    print(f'left child is {curr.left.attribute}')
                if (curr.right != None and isinstance(curr.right.attribute, int) and curr.right.attribute != 1):
                   print(f'right child is {curr.right.attribute}')
                
                t = []
                if (curr.left != None):
                    q.append(curr.left) 
                    t.append(curr.left.attribute)
                if (curr.right != None):
                    q.append(curr.right)
                    t.append(curr.right.attribute)
                temp.append(t)
            print(temp)


    #### BUILDING DECISION TREE ####
    # Scans through the data to see if it can be a leaf.
    def check_potential_leaf(self, data, y):
        for val in data[data.columns[-1]]:
            if val != y:
                return False
        return True
    
    ## Two hueristics: Information gain, Variance impurity
    def information_gain(self, current_impurity, new_impurity):
        return current_impurity - new_impurity


    # Hueristics: Information gain
    def entropy(self, data):
        val_zero, val_one = self.num_of_attr_val(data, data.columns[-1])
        total = val_zero + val_one
        return 0 if val_zero == 0 or val_one == 0 else ((-1 * val_one * math.log2(val_one / total)) / total) - ((val_zero * math.log2(val_zero / total)) / total)
    
    # Checks if we have duplicate features
    def check_duplicates(self, data):
        numpy_temp = data.iloc[:,:-1].values
        return (numpy_temp[0] == numpy_temp).all()

    # Hueristics: Variance impurity
    def variance_gain(self, data):
        val_zero, val_one = self.num_of_attr_val(data, data.columns[-1])
        total = val_zero + val_one
        ## total == 0 --> it splits perfectly? why cna total be 0
        return 0 if val_zero == 0 or val_one == 0 else (val_zero / total) * (val_one / total)

    # Returns a deep copy of the dataframe with only the column_name with attribute val
    def extract_attr_val(self, data, col_name, val, drop_col):
        deep_copy = data.copy()
        # remove the entries (rows) with attribute (column_name) != value (val) --> df only has entries with this attribute
        deep_copy = deep_copy[deep_copy[col_name] == val]
        # drop the that attribute column because we cannot reuse attributes
        if drop_col:
            del deep_copy[col_name]
        return deep_copy

    # Returns the number of occurrences with col_name_val = 0 and col_name_val = 1 (P(x) of Gain(S, A) formula)
    def num_of_attr_val(self, data, col_name):
        num_attr_zero, num_attr_one = 0, 0
        for val in data[col_name]:
            num_attr_zero = num_attr_zero + 1 if not val else num_attr_zero
            num_attr_one = num_attr_one + 1 if val else num_attr_one
        return [num_attr_zero, num_attr_one]

    ## Extracts attribute based on heuristics and information gain
    def extract_best_attribute(self, data):
        current_impurity = self.entropy(data) if self.heruistic == 'entropy' else self.variance_gain(data)
        best_attr, highest_gain = None, -2
        for col_name in data.columns[:len(data.columns)-1]:
            #print(highest_gain)
            num_of_attr_zero, num_of_attr_one = self.num_of_attr_val(data, col_name)
            total_entries = num_of_attr_zero + num_of_attr_one

            # extract the dataset with attr col_name with values 0
            dataset_val_zero = self.extract_attr_val(data, col_name, 0, 0)
            temp_impurity_data_zero = self.entropy(dataset_val_zero) if self.heruistic == 'entropy' else self.variance_gain(dataset_val_zero)
            new_impurity = temp_impurity_data_zero * (num_of_attr_zero/total_entries)

            # extract the dataset with attr col_name with values 1
            dataset_val_one = self.extract_attr_val(data, col_name, 1, 0)
            temp_impurity_data_one = self.entropy(dataset_val_one) if self.heruistic == 'entropy' else self.variance_gain(dataset_val_one)
            new_impurity = (temp_impurity_data_one * (num_of_attr_one/total_entries)) + new_impurity 

            impurity = self.information_gain(current_impurity, new_impurity)
            # check if this column has the highest information gain for the self.herusitic
            if impurity > highest_gain:
                best_attr = col_name
                highest_gain = impurity

        return best_attr

    # This builds the decision tree using the data read from the csv
    def build_decision_tree(self, data):
        if self.check_potential_leaf(data, 0):
            return self.Decision_Tree_Node(0)
        elif self.check_potential_leaf(data, 1):
            return self.Decision_Tree_Node(1)
        elif self.check_duplicates(data):
            num_class_zero, num_class_one = self.num_of_attr_val(data, data.columns[-1])
            leaf_val = 0 if max(num_class_one, num_class_zero) == num_class_zero else 1
            return self.Decision_Tree_Node(random.randint(0,1)) if num_class_one == num_class_zero else self.Decision_Tree_Node(leaf_val)
        else:
            # find the best attribute
            best_attribute = self.extract_best_attribute(data)
            # extract subsets in data where data has best_attribute 0 or 1
            data_zero = self.extract_attr_val(data, best_attribute, 0, 1)
            data_one = self.extract_attr_val(data, best_attribute, 1, 1)
            return self.Decision_Tree_Node(best_attribute, self.build_decision_tree(data_zero), self.build_decision_tree(data_one))

def main():
    ## Checks # of arguments
    if (len(sys.argv) != 6):
        print("Error: Invalid number of arguments")
        return -1
    
    ## Checks valid file types
    if (not (sys.argv[1][-3:] == "csv" and sys.argv[1][-3:] == sys.argv[2][-3:] == sys.argv[3][-3:])):
        print("Error: Invalid file types")
        return -1

    ## Checks for valid yes/no (for printing the tree)
    if (sys.argv[4].lower() !=  "yes" and sys.argv[4].lower() != "no"):
        print("Error: Invalid 4th argument (only yes or no)")
        return -1

    ## Checks for valid h1/h2 (for choosing the metric of the tree)
    if (sys.argv[5].lower() != "h1" and sys.argv[5].lower() != "h2"):
        print("Error: Invalid 5th argument (only h1 or h2)")
        return -1
    
    tree_type = 'entropy' if sys.argv[5].lower() == 'h1' else 'variance'
    decision_tree = Decision_Tree(sys.argv[1], tree_type)

    heuristic_selected = 'H1' if sys.argv[5].lower() == 'h1' else 'H2'
    print(f'{heuristic_selected} NP train {decision_tree.test_tree(sys.argv[1])}')
    print(f'{heuristic_selected} NP valid {decision_tree.test_tree(sys.argv[2])}')
    print(f'{heuristic_selected} NP test {decision_tree.test_tree(sys.argv[3])}')

    if sys.argv[4].lower() == "yes":
        decision_tree.print_tree()


if __name__ == "__main__":
    main()