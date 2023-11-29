import os
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.manifold import TSNE
from PIL import Image

import args

def show_scores(args, results, gts, cats, epoch):
    print(show_scores)

    results = np.array(results)
    gts = np.array(gts)
    cats = np.array(cats)

    results_0 = results[cats==0]
    results_1 = results[cats==1]
    results_2 = results[cats==2]
    gts_0 = gts[cats==0]
    gts_1 = gts[cats==1]
    gts_2 = gts[cats==2]

    sample_index = np.linspace(start=0, stop=len(results_0)-1, num=args.num_show).astype(int)
    results_0 = results_0[sample_index]
    gts_0 = gts_0[sample_index]

    sample_index = np.linspace(start=0, stop=len(results_1)-1, num=args.num_show).astype(int)
    results_1 = results_1[sample_index]
    gts_1 = gts_1[sample_index]

    sample_index = np.linspace(start=0, stop=len(results_2)-1, num=args.num_show).astype(int)
    results_2 = results_2[sample_index]
    gts_2 = gts_2[sample_index]

    results = list(results_0) +list(results_1) +list(results_2)
    gts = list(gts_0) +list(gts_1) +list(gts_2)

    results = (results-min(results))/(max(results)-min(results))
    x_fever = []
    x_normal = []
    res_fever = []
    res_normal = []
    for index, [res, gt] in enumerate(zip(results, gts)):
        if gt==0:
            x_normal.append(index)
            res_normal.append(res)
        else:
            x_fever.append(index)
            res_fever.append(res)

    plt.figure(figsize=(15, 4))
    plt.clf()
    plt.ylim(-0.1, 1.1)
    plt.tick_params(labelsize=16) 
    plt.scatter(x_normal, res_normal, c='b')
    plt.scatter(x_fever,  res_fever, s=100, c='r', marker='*')
    plt.savefig(os.path.join(args.results_dir, 'score_{}_vector.jpg'.format(args.Mode)), bbox_inches='tight')

def show_tsne(args, heats, cats, epoch):
    print(show_tsne)
    heats = np.array(heats)
    cats = np.array(cats)

    heats_0 = heats[cats==0]
    heats_1 = heats[cats==1]
    heats_2 = heats[cats==2]
    cats_0  = cats[cats==0]
    cats_1  = cats[cats==1]
    cats_2  = cats[cats==2]

    sample_index = np.linspace(start=0, stop=len(heats_0)-1, num=args.num_show).astype(int)
    heats_0 = heats_0[sample_index]
    cats_0 = cats_0[sample_index]

    sample_index = np.linspace(start=0, stop=len(heats_1)-1, num=args.num_show).astype(int)
    heats_1 = heats_1[sample_index]
    cats_1 = cats_1[sample_index]

    sample_index = np.linspace(start=0, stop=len(heats_2)-1, num=args.num_show).astype(int)
    heats_2 = heats_2[sample_index]
    cats_2 = cats_2[sample_index]

    heats = list(heats_0) +list(heats_1) +list(heats_2)
    cats = list(cats_0) +list(cats_1) +list(cats_2)

    X_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(heats)

    xlim = 20
    ylim = 20

    plt.clf()
    s = 0
    e = len(heats_0)
    plt.figure(figsize=(5, 5))
    plt.scatter(X_tsne[s:e, 0], X_tsne[s:e, 1], s=10, c=cats[s:e])
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.tick_params(labelsize=16) 
    plt.savefig(os.path.join(args.results_dir, '{}_epoch{}_tsne1.png'.format(args.Mode, epoch)), dpi=120, bbox_inches='tight')
    plt.close()

    plt.clf()
    s = len(heats_0)
    e = len(heats_0)+len(heats_1)
    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.scatter(X_tsne[s:e, 0], X_tsne[s:e, 1], s=10, c=cats[s:e])
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.tick_params(labelsize=16) 
    plt.savefig(os.path.join(args.results_dir, '{}_epoch{}_tsne2.png'.format(args.Mode, epoch)), dpi=120, bbox_inches='tight')
    plt.close()

    plt.clf()
    s = len(heats_0)+len(heats_1)
    e = len(heats_0)+len(heats_1)+len(heats_2)
    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.scatter(X_tsne[s:e, 0], X_tsne[s:e, 1], s=10, c=cats[s:e])
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.tick_params(labelsize=16) 
    plt.savefig(os.path.join(args.results_dir, '{}_epoch{}_tsne3.png'.format(args.Mode, epoch)), dpi=120, bbox_inches='tight')
    plt.close()

    plt.clf()
    s = 0
    e = len(heats)
    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.scatter(X_tsne[s:e, 0], X_tsne[s:e, 1], s=10, c=cats)
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.tick_params(labelsize=16) 
    plt.savefig(os.path.join(args.results_dir, '{}_epoch{}_tsneall.png'.format(args.Mode, epoch)), dpi=120, bbox_inches='tight')
    plt.close()

# 保存测试结果
def write_results(args, results, results_file):
    f = open(results_file, 'w')
    for result in results:
        sample_id = result[0]
        bbox = result[1]
        pre = result[2]
        f.write(sample_id)
        for b in bbox:
            f.write(' '+str(b))
        for p in pre:
            f.write(' '+str(p))
        f.write('\n')
    f.close()

# def show_tsne(args, vectors, cats, epoch):
#     tsne_dir = os.path.join(args.workID_dir, 'tsne')
#     if not os.path.exists(tsne_dir): os.mkdir(tsne_dir)

#     plt.clf()
#     plt.figure(figsize=(5, 5))

#     # X_tsne = TSNE().fit_transform(vectors)
#     # X_tsne = TSNE(n_components=2,random_state=33).fit_transform(vectors)
#     X_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(vectors)
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10, c=cats)
#     # plt.legend()

#     plt.savefig(os.path.join(tsne_dir, 'tsne-pca_{}.png'.format(epoch)), dpi=120)
#     plt.close()
    
def array_distance(arr1, arr2):
    '''
    计算两个数组里，每任意两个点之间的L2距离
    arr1 和 arr2 都必须是numpy数组
    且维度分别为 m x 2, n x 2
    输出数组的维度为 m x n
    '''
    m, _ = arr1.shape
    n, _ = arr2.shape
    arr1_power = np.power(arr1, 2)
    arr1_power_sum = arr1_power[:, 0] + arr1_power[:, 1]
    arr1_power_sum = np.tile(arr1_power_sum, (n, 1))
    arr1_power_sum = arr1_power_sum.T
    arr2_power = np.power(arr2, 2)
    arr2_power_sum = arr2_power[:, 0] + arr2_power[:, 1]
    arr2_power_sum = np.tile(arr2_power_sum, (m, 1))
    dis = arr1_power_sum + arr2_power_sum - (2 * np.dot(arr1, arr2.T))
    dis = np.sqrt(dis)
    return dis

def show_aucs(args, aucs):

    with open(os.path.join(args.workID_dir, 'AUCs.txt'), 'w') as f:
        for a in aucs:
            f.write(str(a))
            f.write(' ')

    plt.clf()
    plt.plot(range(len(aucs)), aucs)
    plt.savefig(os.path.join(args.workID_dir, 'AUCs.png'), dpi=120)
    plt.close()

def scatter():
    plt.clf()
    xs = range(0, 100, 10)
    ys = xs
    cats = range(10)
    plt.scatter(xs, ys, c=cats, s= 1000)
    plt.savefig('scatter.png', dpi=120)

def iron():
    iron_r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 10, 13, 18, 23, 25, 29, 33, 36, 40, 43, 47, 51, 54, 57, 60, 64, 67, 69, 74, 77, 79, 82, 86, 89, 92, 94, 96, 101, 103, 109, 113, 115, 118, 120, 123, 126, 128, 131, 132, 134, 138, 141, 144, 145, 149, 149, 150, 155, 156, 157, 161, 163, 164, 166, 169, 172, 172, 174, 177, 180, 179, 182, 183, 186, 189, 192, 193, 192, 197, 198, 198, 201, 200, 201, 206, 207, 208, 208, 210, 212, 213, 215, 215, 218, 219, 219, 220, 222, 222, 223, 226, 226, 227, 229, 229, 230, 231, 233, 234, 236, 238, 237, 237, 239, 239, 240, 241, 243, 244, 243, 244, 246, 245, 246, 248, 249, 248, 249, 249, 250, 250, 248, 249, 249, 248, 250, 250, 249, 249, 250, 249, 249, 250, 250, 249, 249, 250, 250, 249, 250, 249, 249, 250, 248, 249, 249, 248, 248, 249, 249, 250, 250, 249, 250, 250, 249, 249, 249, 250, 249, 249, 248, 250, 249, 249, 248, 249, 249, 250, 249, 249, 250, 250, 249, 249, 250, 249, 249, 249, 248, 249, 249, 250, 248, 248, 247, 247, 248, 247, 247, 247, 245, 246, 244, 243, 244, 242, 240, 242, 241, 241, 239, 239, 240, 238, 238, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236]
    iron_g = [15, 14, 12, 11, 10, 9, 8, 7, 6, 5, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 5, 6, 6, 7, 8, 8, 10, 12, 13, 13, 14, 16, 17, 18, 19, 22, 23, 24, 24, 25, 26, 28, 31, 31, 33, 33, 35, 37, 38, 39, 40, 42, 45, 46, 48, 50, 51, 54, 54, 56, 59, 61, 62, 64, 66, 67, 70, 72, 73, 75, 77, 79, 81, 83, 84, 86, 89, 91, 92, 95, 97, 99, 100, 102, 105, 106, 108, 111, 113, 114, 116, 118, 120, 123, 125, 127, 127, 131, 131, 134, 135, 136, 138, 141, 142, 145, 147, 148, 150, 152, 154, 156, 158, 160, 163, 166, 170, 171, 173, 174, 176, 178, 180, 182, 184, 186, 187, 189, 191, 193, 195, 197, 199, 199, 201, 204, 205, 206, 208, 209, 212, 212, 214, 217, 218, 220, 221, 222, 224, 225, 226, 227, 229, 230, 231, 232, 233, 233, 236, 236, 238, 239, 239, 240, 241, 242, 244, 244, 246, 246, 247, 248, 250, 250, 250, 251, 251, 251, 251, 250, 251, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250]
    iron_b = [20, 25, 31, 37, 43, 50, 56, 60, 66, 72, 75, 82, 85, 88, 95, 97, 102, 106, 109, 112, 119, 120, 123, 126, 130, 131, 134, 137, 138, 140, 143, 144, 145, 148, 149, 150, 151, 152, 155, 155, 156, 157, 159, 161, 160, 160, 161, 160, 162, 162, 161, 162, 160, 161, 162, 159, 160, 160, 159, 158, 155, 154, 153, 154, 154, 153, 150, 148, 149, 146, 147, 145, 144, 143, 138, 137, 134, 135, 133, 132, 129, 127, 124, 123, 119, 116, 117, 115, 112, 111, 108, 105, 102, 101, 99, 96, 93, 92, 89, 86, 85, 82, 80, 77, 76, 73, 70, 65, 66, 64, 61, 60, 57, 52, 49, 48, 47, 45, 42, 39, 36, 35, 32, 31, 31, 28, 25, 23, 22, 19, 20, 17, 17, 14, 13, 11, 9, 7, 6, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 8, 9, 8, 13, 14, 17, 21, 20, 23, 27, 29, 33, 36, 39, 42, 45, 49, 54, 57, 62, 65, 70, 76, 81, 86, 89, 96, 102, 107, 112, 117, 125, 131, 136, 141, 149, 157, 160, 169, 176, 185, 193, 202, 208, 217, 224, 235, 242, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248]
    iron_r = np.array(iron_r)
    iron_g = np.array(iron_g)
    iron_b = np.array(iron_b)
    return [iron_r, iron_g, iron_b]

def gray2iron(data):
    [iron_r, iron_g, iron_b] = iron()
    img_gray = np.array(data, np.uint8)
    img_rgb = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    img_rgb[:, :, 2] = iron_r[img_gray]
    img_rgb[:, :, 1] = iron_g[img_gray]
    img_rgb[:, :, 0] = iron_b[img_gray]
    return img_rgb
    
if __name__ == '__main__':
    pass
    # args = args.get_args()
    # main()
    # args = args.get_args('-')
    # aucs = [1, 2, 3, 4]
    # show_aucs(args, aucs)
    