import os
import numpy as np
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def get_data(data_path, classes, shuffle=True):
    class_coord_arrays = []
    class_label_arrays = []

    for i, c in enumerate(classes):
        class_coords = []
        class_path = os.path.join(data_path, c)
        videos = os.listdir(class_path)
        for v in videos:
            video_path = os.path.join(class_path, v)
            coords = os.listdir(video_path)
            for coord in coords:
                coord_path = os.path.join(video_path, coord)
                np_coords = np.load(coord_path)
                num_people, num_points, num_features = np_coords.shape
                for j in range(num_people):
                    class_coords.append(np_coords[j].flatten())
        class_coord_arrays.append(np.stack(class_coords))
        class_label_arrays.append(np.full((len(class_coords),), i, dtype=np.float32))

    class_coord_arrays = np.concatenate(class_coord_arrays)
    class_label_arrays = np.concatenate(class_label_arrays)

    if shuffle:
        p = np.random.permutation(class_label_arrays.shape[0])
        class_coord_arrays = class_coord_arrays[p]
        class_label_arrays = class_label_arrays[p]
    
    return class_coord_arrays, class_label_arrays


data_path = "/home/gpa/GPA_DATA/Coords/"
classes = ["bad", "good"]
N_folds = 10
random_state = 28

x, y = get_data(data_path, classes)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/N_folds, random_state=random_state)
kfold = KFold(n_splits=N_folds, random_state=random_state)

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=5000, alpha=0.001, solver='sgd', verbose=0,  random_state=random_state,tol=0.000000001)

clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))

#results = cross_val_score(clf, x, y, cv=kfold)


