#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
def fd_histogram(image, mask=None):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist,hist)

    return hist.flatten()


# In[2]:


import cv2
def fd_hu_moments(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    
    return feature


# In[3]:


import cv2
import mahotas
def fd_haralick(image):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    
    return haralic


# In[4]:


import cv2
def fd_sift(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(gray,None)
    
    return kp, des


# In[5]:


import cv2
def fd_surf(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SURF.create()
    kp, des = sift.detectAndCompute(gray,None)
    
    return kp, des


# In[6]:


import cv2
import numpy as np
def fd_sift_histogram(image, sift_cl_centers, fixed_size):
    
    image = cv2.resize(image,fixed_size)
    kp, des = fd_sift(image)
    closest_cl = []

    for i in range(len(des)):
        dist = np.sum(np.power(sift_cl_centers - des[i], 2), axis = 1)
        nearest_cluster = np.argmin(dist)
        closest_cl.append(nearest_cluster)
    
    counts = np.array([], int)
    for j in range(len(sift_cl_centers)):
        counts = np.append(counts, closest_cl.count(j))
        
    return counts


# In[7]:


import cv2
import numpy as np
def fd_surf_histogram(image, surf_cl_centers, fixed_size):
    
    image = cv2.resize(image,fixed_size)
    kp, des = fd_surf(image)
    closest_cl = []

    for i in range(len(des)):
        dist = np.sum(np.power(surf_cl_centers - des[i], 2), axis = 1)
        nearest_cluster = np.argmin(dist)
        closest_cl.append(nearest_cluster)
    
    counts = np.array([], int)
    for j in range(len(surf_cl_centers)):
        counts = np.append(counts, closest_cl.count(j))
        
    return counts


# In[3]:


import random
def train_test_split(df, test_percent):
    
    test_size = len(df)*(test_percent)/100
    
    if isinstance(test_size, float):
        test_size = round(test_size)
    
    indices = df.index.tolist()

    test_indices = random.sample(indices, test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


# In[9]:


def type_of_feature(df):
    
    threshold = round(0.01*len(df))
    feature_type_list = []
    
    for column in df.columns:
        unique_classes = df[column].unique()
        example_class = unique_classes[0]
            
        if (isinstance(example_class, str)) or len(unique_classes) < threshold :
            feature_type_list.append("Catagorical")
        else:
            feature_type_list.append("Continuous")
            
    return feature_type_list


# In[10]:


import random
def get_potential_splits(df, random_subspace):
    
    potential_splits = {}
    feature_type_list = type_of_feature(df)
    _, n_columns = df.shape
    
    column_indices = list(range(n_columns - 1))

    if random_subspace & random_subspace <= len(column_indices):
        column_indices =  random.sample(column_indices, random_subspace)
    
    for i in column_indices:
        potential_splits[i] = []
        unique_values = df.iloc[:,i].unique()
        potential_splits[i] = unique_values
    
    return potential_splits


# In[21]:


import numpy as np
def check_purity(data):
    
    if type(data) != np.ndarray:
        data = np.array(data)
    
    label = data[:,-1]
    unique_classes = np.unique(label)

    if len(unique_classes) == 1:
        return True
    else:
        return False


# In[18]:


import numpy as np
def classify_data(data):
    
    if type(data) != np.ndarray:
        data = np.array(data)
    
    label = data[:,-1]
    unique_classes,count_unique_classes = np.unique(label,return_counts=True)

    index = count_unique_classes.argmax()  
    classification = unique_classes[index]

    return classification


# In[16]:


import numpy as np
def split_data(data,split_column,split_vlaue, df):
    
    if type(data) != np.ndarray:
        data = np.array(data)
    
    feature_type_list = type_of_feature(df)
    
    if feature_type_list[split_column] == 'Continuous':
        
        data_below = data[data[:,split_column] <= split_vlaue]
        data_above = data[data[:,split_column] > split_vlaue]
        
    else:
        
        data_below = data[data[:,split_column] == split_vlaue]
        data_above = data[data[:,split_column] != split_vlaue]
        
    return data_below,data_above


# In[13]:


import numpy as np
def calculate_entropy(data):
    
    if type(data) != np.ndarray:
        data = np.array(data)
    
    label = data[:,-1]
    _,counts = np.unique(label,return_counts=True)

    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


# In[3]:


import numpy as np
def calculate_overall_entropy(data_below,data_above):
    
    n_data_points = len(data_below)+len(data_above)
    n_data_below = len(data_below)/n_data_points
    n_data_above = len(data_above)/n_data_points

    overall_entropy = (n_data_below*calculate_entropy(data_below)) + (n_data_above*calculate_entropy(data_above))

    return overall_entropy


# In[5]:


import numpy as np
def best_splits(data, random_subspace, train_df, df):
    
    if type(data) != np.ndarray:
        data = np.array(data)
    
    overall_entropy = float('inf')
    potential_splits = get_potential_splits(train_df, random_subspace)
  
    for i in potential_splits:
        for j in potential_splits[i]:
            data_below,data_above = split_data(data,i,j,df)
            current_overall_entropy = calculate_overall_entropy(data_below,data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = i
                best_split_value = j

    return best_split_column, best_split_value


# In[17]:


import random
def bootstrapping(train_df, n_bootstrap):
    
    bootstrap_indices = np.random.randint(0, len(train_df), n_bootstrap)
    df_bootstraped = train_df.iloc[bootstrap_indices]
    
    return df_bootstraped


# In[2]:


import numpy as np
def decision_tree_algorithm(data, random_subspace, max_depth, train_df, df, counter=0, min_samples = 10):
    
    if type(data) != np.ndarray:
        data = np.array(data)
    
    if check_purity(data) or (len(data) <= min_samples) or (counter == max_depth):
        return classify_data(data)
    else:
        counter += 1
        
        split_column,split_value = best_splits(data, random_subspace, train_df, df)
        data_below,data_above = split_data(data,split_column,split_value, df)
        
        feature_type_list = type_of_feature(df)
    
        if feature_type_list[split_column] == 'Continuous':
            question = f'{df.columns[split_column]} <= {split_value}'
        else:
            question = f'{df.columns[split_column]} = {split_value}'
            
        sub_tree = {question : []}
        
        yes_answer = decision_tree_algorithm(data_below, random_subspace, max_depth, train_df, df, counter)
        no_answer = decision_tree_algorithm(data_above, random_subspace, max_depth, train_df, df, counter)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree


# In[19]:


def classify_sample(example, tree): 
    
    question = list(tree.keys())[0]
    feature, compare, value = question.split()
    
    if compare == '<=':
        if example[float(feature)] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if example[float(feature)] == float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
        
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_sample(example, residual_tree)


# In[8]:


import multiprocessing
from multiprocessing import Pool
from functools import partial
from p_tqdm import p_map

def random_forest_generation(train_df, n_trees, n_bootstrap, n_features, dt_max_depth, df):
    
    bootstrap_data = []
    for i in range(n_trees):
        df_bootstraped = bootstrapping(train_df, n_bootstrap)
        bootstrap_data.append(df_bootstraped.values)
           
    n_processor = multiprocessing.cpu_count()
    p = Pool(n_processor)
    forest = p_map(partial(decision_tree_algorithm,random_subspace=n_features,max_depth=dt_max_depth,train_df=train_df,df=df), bootstrap_data)
    p.close()
    p.join()
    
    return forest


# In[26]:


def decision_tree_predictions(df, tree):
    
    if len(df) != 0:
    
        prediction = []
        for i in range(len(df)):
            predicted_class = classify_sample(df.iloc[i], tree)
            prediction.append(predicted_class)
        
    return prediction


# In[27]:


import pandas as pd
def random_forest_prediction(df,forest):
    
    if len(df) != 0:
        
        df_predictions = {}
        for i in range(len(forest)):
            name = f'tree-{i}'
            prediction = decision_tree_predictions(df, forest[i])
            df_predictions[name] = prediction

        df_prediction = pd.DataFrame(df_predictions)
        random_forest_prediction = df_prediction.mode(axis = 1)[0]
    
    return random_forest_prediction


# In[28]:


def calculate_accuracy(predictions, labels):
    
    if len(predictions) != 0:
    
        predictions.index = labels.index
        predictions_correct = predictions == labels
        accuracy = predictions_correct.mean()

    return accuracy

