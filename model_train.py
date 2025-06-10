#!/usr/bin/env python
# coding: utf-8

# In[3]:


def compute_accuracy(Y_true, Y_pred):  
    correctly_predicted = 0  
    for true_label, predicted in zip(Y_true, Y_pred):  
        if true_label == predicted:  
            correctly_predicted += 1    
    accuracy_score = correctly_predicted / len(Y_true)  
    return accuracy_score  


# In[4]:


import pickle
import numpy as np
from RandomForestManual import RandomForest
data_dict = pickle.load(open('./data.pickle', 'rb'))
filtered_data = [d for d in data_dict['data'] if isinstance(d, (list, np.ndarray)) and len(d) == 42]
filtered_labels = [label for d, label in zip(data_dict['data'], data_dict['labels']) if isinstance(d, (list, np.ndarray)) and len(d) == 42]

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)
indices = np.random.permutation(len(data))
data = data[indices]
labels = labels[indices]
test_size = 0.2  
split_index = int(len(data) * (1 - test_size))
x_train, x_test = data[:split_index], data[split_index:] 
y_train, y_test = labels[:split_index], labels[split_index:] 
model = RandomForest()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = compute_accuracy(y_test,y_predict)
print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()


# In[ ]:




