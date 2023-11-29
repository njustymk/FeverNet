import pdb
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def calculate_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    return auc

def Evaluate_img(args, scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = format(auc(fpr, tpr)*100, '.2f')
    return auc_value

def Evaluate_face(args, errs_face_all, labels_face):
    top1_true = 0
    top5_true = 0
    top10_true = 0
    top20_true = 0
    for errs_face, label_face in zip(errs_face_all, labels_face):

        top1 = np.argsort(errs_face)[-1:]
        if np.isin(label_face, top1).sum()==len(label_face): top1_true += 1

        top5 = np.argsort(errs_face)[-5:]
        if np.isin(label_face, top5).sum()==len(label_face): top5_true += 1
        
        top10 = np.argsort(errs_face)[-10:]
        if np.isin(label_face, top10).sum()==len(label_face): top10_true += 1

        top20 = np.argsort(errs_face)[-20:]
        if np.isin(label_face, top20).sum()==len(label_face): top20_true += 1

    top1_tpr  = top1_true*1.0/len(labels_face)
    top5_tpr  = top5_true*1.0/len(labels_face)
    top10_tpr = top10_true*1.0/len(labels_face)
    top20_tpr = top20_true*1.0/len(labels_face)

    top1_tpr  = format(top1_tpr*100, '.2f')
    top5_tpr  = format(top5_tpr*100, '.2f')
    top10_tpr = format(top10_tpr*100, '.2f')
    top20_tpr = format(top20_tpr*100, '.2f')

    return top1_tpr, top5_tpr, top10_tpr, top20_tpr

# 测试有无核心图约束时的收敛速度
def eval_aucs():
    with_file    = 'work/FAM+DCB+EDCL/FAM+DCB+EDCL.log'
    without_file = 'work/FAM+DCB+EDCL-CTMap/FAM+DCB+EDCL-CTMap.log'

    aucs_with = []
    with open(with_file, 'r') as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        if index <110:
            continue
        line = line.strip().split(' ')
        aucs_with.append(float(line[-1]))


    aucs_without = []
    with open(without_file, 'r') as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        if index <110:
            continue
        line = line.strip().split(' ')

        aucs_without.append(float(line[-1]))

    plt.clf()
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.plot(range(len(aucs_with)), aucs_with, label='With', color = 'blue')
    plt.plot(range(len(aucs_without)), aucs_without, label='Without', color = 'red')
    plt.tick_params(labelsize=16) 
    plt.xlabel('Epoch', size=16)
    plt.ylabel('AUC', size=16)  
    plt.title('AUCs during training', size=16)
    plt.legend(loc="lower right")
    plt.legend(prop={'size': 14})
    plt.savefig('cache/Aucs-CTMap.jpg', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    eval_aucs()