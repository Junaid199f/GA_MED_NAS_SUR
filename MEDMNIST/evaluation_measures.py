import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def evaluate_measures(y_true, y_score, task):
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()
    if task == 'multi-label, binary-class':
        threshold = 0.5
        print(y_true.shape)
        print(y_score.shape)
        print(task)
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = f1_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
        print(ret)

    else:
        ret = f1_score(y_true, np.argmax(y_score, axis=-1),average='weighted')
        # ret = f1_score(y_true, np.argmax(y_score, axis=-1))
        # print(y_true.shape)
        # print(y_score.shape)
        # print(task)
        # y_score = y_score.argmax(axis=1)
        # # Assuming we have predicted and actual labels saved in 'y_pred' and 'y_true' respectively
        # report = classification_report(y_true, y_score)
        # precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_score, average='weighted')
        #
        # # Print the classification report
        # print(f1_score)

    return ret

