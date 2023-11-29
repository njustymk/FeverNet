import os, cv2, pdb, tqdm, torch, random, pickle, torchvision
import numpy as np
import matplotlib.pyplot as plt

import utils

def read_bbox(bbox_file):
    bboxes = []
    f = open(bbox_file, 'r')
    datas = f.readlines()
    for data in datas:
        parts = data.strip().split(' ')
        [xmin_vis, ymin_vis, xmax_vis, ymax_vis, xmin_ir, ymin_ir, xmax_ir, ymax_ir, score, temp_max] = parts
        bboxes.append([float(xmin_vis), float(ymin_vis), float(xmax_vis), float(ymax_vis), \
                        float(xmin_ir), float(ymin_ir), float(xmax_ir), float(ymax_ir), float(score), float(temp_max)])
    return bboxes

def disp_to_depth(args, disp):
    
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    disp = disp.squeeze()
    disp = cv2.resize(disp, (args.img_width, args.img_height))
    min_depth = 0.1
    max_depth = 100
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    depth = depth.astype(np.float32)

    return depth

def prepare_face(args, flod):
    print('prepare_face', flod)
    temp_faces      = []
    depth_faces     = []
    sample_ids_face = []
    bboxes_face     = []
    temp_maxs       = []

    bbox_dir = os.path.join(args.data_root, flod, 'bbox_visir')

    # Sample IDs
    sample_ids = [x[:18] for x in os.listdir(bbox_dir)]
    sample_ids = np.sort(sample_ids)
    
    for sample_id in tqdm.tqdm(sample_ids):

        temp  = np.load(os.path.join(args.data_root, flod, 'temp',  sample_id+'_temp.npy'))
        disp  = np.load(os.path.join(args.data_root, flod, 'disp',  sample_id+'_disp.npy'))

        depth = disp_to_depth(args, disp)
        bbox_file = os.path.join(bbox_dir, sample_id+'_bbox_visir.txt')
        bboxes = read_bbox(bbox_file)

        for bbox in bboxes:
            xmin_vis = int(bbox[0])
            ymin_vis = int(bbox[1])
            xmax_vis = int(bbox[2])
            ymax_vis = int(bbox[3])
            xmin_ir  = int(bbox[4])
            ymin_ir  = int(bbox[5])
            xmax_ir  = int(bbox[6])
            ymax_ir  = int(bbox[7])
            score    = float(bbox[8])
            temp_max = float(bbox[9])

            # Face temperature
            temp_face = temp[ymin_ir:ymax_ir, xmin_ir:xmax_ir].astype(np.float32)
            temp_faces.append(temp_face)

            # Face depth
            ymin_depth = ymin_vis+(ymax_vis-ymin_vis)
            ymax_depth = ymax_vis+(ymax_vis-ymin_vis)
            depth_face = depth[ymin_depth:ymax_depth, xmin_vis:xmax_vis]
            depth_faces.append(depth_face)

            # Face max temperature
            temp_maxs.append(temp_max)

            sample_ids_face.append(sample_id)
            bboxes_face.append(np.array([xmin_vis, ymin_vis, xmax_vis, ymax_vis, xmin_ir, ymin_ir, xmax_ir, ymax_ir]))

            # img_show = (disp-disp.min())/(disp.max()-disp.min())
            # # cv2.rectangle(img_show, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 2)  #BGR
            # cv2.imwrite(os.path.join(args.show_dir, sample_id+'.jpg'), img_show*255)
            # pdb.set_trace()

    temp_maxs = np.array(temp_maxs)

    # count
    plt.hist(temp_maxs, bins=np.arange(20, 40, 0.1), facecolor='red')
    plt.savefig(os.path.join(args.show_dir, 'conut.jpg'))

    # num_part = 100
    # len_part = int(len(temp_maxs)/num_part)
    # for i in range(num_part):
    #     temps_part =temp_maxs[len_part*i: len_part*(i+1)]
    #     print('Before Flit1 temps_part:{} mean:{}'.format(i, temps_part.mean()))

    print('Num face ', len(temp_maxs))
    print('Mean ', temp_maxs.mean())
    print('Top10', np.sort(temp_maxs)[-10:])
    print('End10', np.sort(temp_maxs)[:10])

    data_list = [temp_faces, depth_faces, sample_ids_face, bboxes_face]
    return data_list

def trans_face(args, face):
    face = cv2.resize(face, (args.face_size, args.face_size), cv2.INTER_LINEAR)
    face = face.astype(np.float32)
    face[face<args.temp_min] =args.temp_min
    face[face>args.temp_max] =args.temp_max
    face = (face - args.temp_min)/(args.temp_max - args.temp_min)
    return face

def trans_face_normal(args, face):
    face = face.copy()
    face = cv2.resize(face, (args.face_size, args.face_size), cv2.INTER_LINEAR)
    face = face.astype(np.float32)
    face[face<args.temp_min] =args.temp_min
    face[face>args.temp_max] =args.temp_max
    face = (face - args.temp_min)/(args.temp_max - args.temp_min)

    return face

def trans_face_local(args, face, i):
    face = face.copy()
    face = cv2.resize(face, (args.face_size, args.face_size), cv2.INTER_LINEAR)
    face = face.astype(np.float32)

    m, n = face.shape
    index = int(face.argmax())
    xct_f = int(index / n)
    yct_f = index % n

    r = 1000
    w0 = 1
    x = np.linspace(-1, 1, r)
    y = np.linspace(-1, 1, r)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-((pow(x, 2) + pow(y, 2)) / pow(w0, 2)))

    xct_g = int(r/2)
    yct_g = int(r/2)
    xmin_g = xct_g-xct_f
    ymin_g = yct_g-yct_f
    xmax_g = xmin_g+32
    ymax_g = ymin_g+32

    face_g = gaussian[xmin_g:xmax_g, ymin_g:ymax_g]
    fever_degree = random.random()+1
    face_fever = face+face_g*fever_degree
    # face_fever = face+face_g*np.random.randint(0.5, 2)

    # temp_min = face.min()
    # temp_max = face.max()+2
    # # face = (face-face.min())/(face.max()-face.min())*255
    # face = (face - temp_min)/(temp_max - temp_min)*255
    # face = cv2.resize(face, (256, 256), cv2.INTER_LINEAR)
    # face = utils.gray2iron(face)
    # cv2.imwrite(os.path.join(args.show_dir, '{}_face.jpg'.format(i)),   face  )

    # face_g = (face_g-face_g.min())/(face_g.max()-face_g.min())*255
    # face_g = cv2.resize(face_g, (256, 256), cv2.INTER_LINEAR)
    # face_g = utils.gray2iron(face_g)
    # cv2.imwrite(os.path.join(args.show_dir, '{}_face_g.jpg'.format(i)), face_g)

    # # face_fever = (face_fever-face_fever.min())/(face_fever.max()-face_fever.min())*255
    # face_fever = (face_fever - temp_min)/(temp_max - temp_min)*255
    # face_fever = cv2.resize(face_fever, (256, 256), cv2.INTER_LINEAR)
    # face_fever = utils.gray2iron(face_fever)
    # cv2.imwrite(os.path.join(args.show_dir, '{}_face_fever.jpg'.format(i)),   face_fever  )
        
    face_fever[face_fever<args.temp_min] =args.temp_min
    face_fever[face_fever>args.temp_max] =args.temp_max
    face_fever = (face_fever - args.temp_min)/(args.temp_max - args.temp_min)

    return face_fever


def get_pool_train(args, faces_new, labels, depths):
    vector_pools  = []
    face_pools    = []
    depth_pools   = []
    depth_vectors = []
    label_masks   = []
    label_vectors = []
    for m in tqdm.tqdm(range(0, len(faces_new)-args.pool_size*args.pool_size, 5)):
        vector_pool  = np.zeros([args.pool_size*args.pool_size, args.face_size*args.face_size])
        face_pool    = np.zeros([args.face_size*args.pool_size, args.face_size*args.pool_size])
        depth_pool   = np.zeros([args.face_size*args.pool_size, args.face_size*args.pool_size])
        label_mask   = np.zeros([args.face_size*args.pool_size, args.face_size*args.pool_size])
        label_vector = np.zeros(args.pool_size*args.pool_size)
        depth_vector = np.zeros(args.pool_size*args.pool_size)
        
        for h in range(args.pool_size):
            for w in range(args.pool_size):
                n = h+w*args.pool_size
                wmin = w     *args.face_size
                wmax = (w+1) *args.face_size
                hmin = h     *args.face_size
                hmax = (h+1) *args.face_size
                face  = faces_new[m+n]
                label = labels[m+n]

                vector_pool[n] = -np.sort(-face.flatten())
                face_pool[wmin :wmax,  hmin :hmax] = face

                depth = (np.sort(depths[m+n])[5:10]).mean()
                # depth = depths[m+n]

                depth_pool[wmin :wmax,  hmin :hmax] = np.ones([args.face_size, args.face_size])*depth
                depth_vector[n] = depth

                if label==1:
                    label_mask[wmin :wmax,  hmin :hmax] = np.ones([args.face_size, args.face_size])
                    label_vector[n] = 1
                
        label_mask = cv2.resize(label_mask, (args.Tmap_size, args.Tmap_size), cv2.INTER_LINEAR)
        depth_pool = cv2.resize(depth_pool, (args.Tmap_size, args.Tmap_size), cv2.INTER_LINEAR)

        depth_pool[depth_pool>args.depth_max]=args.depth_max
        depth_pool[depth_pool<args.depth_min]=args.depth_min
        depth_pool = (depth_pool-args.depth_min+0.0001)/(args.depth_max-args.depth_min)

        # print(depth_vector.min(), depth_vector.max())
        # pdb.set_trace()
        
        depth_vector[depth_vector>args.depth_max]=args.depth_max
        depth_vector[depth_vector<args.depth_min]=args.depth_min
        depth_vector = (depth_vector-args.depth_min+0.0001)/(args.depth_max-args.depth_min)

        vector_pools .append(vector_pool)
        face_pools   .append(face_pool)
        depth_pools  .append(depth_pool)
        depth_vectors.append(depth_vector)
        label_masks  .append(label_mask)
        label_vectors.append(label_vector)

    return [vector_pools, face_pools, depth_pools, depth_vectors, label_masks, label_vectors]

def get_pool_test(args, faces_new, labels, depths):
    vector_pools  = []
    face_pools    = []
    depth_pools   = []
    depth_vectors = []
    label_masks   = []
    label_vectors = []
    for m in tqdm.tqdm(range(0, len(faces_new)-args.pool_size*args.pool_size, 1)):
        vector_pool  = np.zeros([args.pool_size*args.pool_size, args.face_size*args.face_size])
        face_pool    = np.zeros([args.face_size*args.pool_size, args.face_size*args.pool_size])
        depth_pool   = np.zeros([args.face_size*args.pool_size, args.face_size*args.pool_size])
        label_mask   = np.zeros([args.face_size*args.pool_size, args.face_size*args.pool_size])
        label_vector = np.zeros(args.pool_size*args.pool_size)
        depth_vector = np.zeros(args.pool_size*args.pool_size)
        
        for h in range(args.pool_size):
            for w in range(args.pool_size):
                n = h+w*args.pool_size
                wmin = w     *args.face_size
                wmax = (w+1) *args.face_size
                hmin = h     *args.face_size
                hmax = (h+1) *args.face_size
                face  = faces_new[m+n]
                label = labels[m+n]

                vector_pool[n] = -np.sort(-face.flatten())
                face_pool[wmin :wmax,  hmin :hmax] = face

                depth = (np.sort(depths[m+n])[5:10]).mean()
                # depth = depths[m+n]

                depth_pool[wmin :wmax,  hmin :hmax] = np.ones([args.face_size, args.face_size])*depth
                depth_vector[n] = depth

                if label==1:
                    label_mask[wmin :wmax,  hmin :hmax] = np.ones([args.face_size, args.face_size])
                    label_vector[n] = 1
                
        label_mask = cv2.resize(label_mask, (args.Tmap_size, args.Tmap_size), cv2.INTER_LINEAR)
        depth_pool = cv2.resize(depth_pool, (args.Tmap_size, args.Tmap_size), cv2.INTER_LINEAR)

        depth_pool[depth_pool>args.depth_max]=args.depth_max
        depth_pool[depth_pool<args.depth_min]=args.depth_min
        depth_pool = (depth_pool-args.depth_min+0.0001)/(args.depth_max-args.depth_min)

        depth_vector[depth_vector>args.depth_max]=args.depth_max
        depth_vector[depth_vector<args.depth_min]=args.depth_min
        depth_vector = (depth_vector-args.depth_min+0.0001)/(args.depth_max-args.depth_min)

        vector_pools .append(vector_pool)
        face_pools   .append(face_pool)
        depth_pools  .append(depth_pool)
        depth_vectors.append(depth_vector)
        label_masks  .append(label_mask)
        label_vectors.append(label_vector)

    return [vector_pools, face_pools, depth_pools, depth_vectors, label_masks, label_vectors]

def get_depth(args, face):
    face_h = face.shape[0]
    depth = 1.0/face_h
    return depth

def get_samplelist(args, data_list):
    print('Get samplelist')

    faces      = data_list[0]
    depths_pre = data_list[1]
    depths_h   = []
    face_maxs   = []
    for i in tqdm.tqdm(range(len(faces))):
        face        = faces[i]
        face_maxs.append(face.max())
    temp_mean = sum(face_maxs)/len(face_maxs)
    print('temp_mean', temp_mean)

    faces_new = []
    labels = []
    for i in tqdm.tqdm(range(len(faces))):
        face        = faces[i]
        depth_h     = get_depth(args, face)
        if random.random()<args.fever_rate:
            face_new = trans_face_local(args, face, i)
            labels.append(1)
        else:
            face_new = trans_face_normal(args, face)
            labels.append(0)

        depths_h .append(depth_h)
        faces_new.append(face_new)
    
    print('Num face', len(faces_new))
    samplelist_train = get_pool_train(args, faces_new [:args.num_train], \
                                      labels    [:args.num_train], \
                                      depths_pre[:args.num_train])
    samplelist_test  = get_pool_test(args, faces_new [ args.num_train :], \
                                  labels    [ args.num_train :], \
                                  depths_pre[ args.num_train :])

    return samplelist_train, samplelist_test

class Dataset_TIF(torch.utils.data.Dataset):
    def __init__(self, args, datalist):
        super(Dataset_TIF, self).__init__()
        self.args = args
        self.vector_pools  = datalist[0]
        self.face_pools    = datalist[1]
        self.depth_pools   = datalist[2]
        self.depth_vectors = datalist[3]
        self.label_masks   = datalist[4]
        self.label_vectors = datalist[5]
        self.senceIDs      = datalist[6]

    def __len__(self):
        return len(self.face_pools)

    def __getitem__(self, index):
        vector_pool  = torch.tensor(self.vector_pools[index],  dtype = torch.float32)
        face_pool    = torch.tensor([self.face_pools[index]],  dtype = torch.float32)
        depth_pool   = torch.tensor([self.depth_pools[index]], dtype = torch.float32)
        depth_vector = torch.tensor(self.depth_vectors[index], dtype = torch.float32)
        label_mask   = torch.tensor([self.label_masks[index]], dtype = torch.float32)
        label_vector = torch.tensor(self.label_vectors[index], dtype = torch.float32)
        senceID      = self.senceIDs[index]
        return vector_pool, face_pool, depth_pool, depth_vector, label_mask, label_vector, senceID

def get_data_from_datasetfile(args, flod):
    datasetfile = os.path.join(args.data_root, 'datasetfile_{}.pkl'.format(flod))
    if os.path.isfile(datasetfile):  
        with open(datasetfile, "rb") as f: 
            data_list = pickle.load(f)
    else: 
        data_list = prepare_face(args, flod)
        with open(datasetfile, "wb") as f: 
            pickle.dump(data_list, f)
    return data_list

def get_dataloader_TIF(args):

    samplelist_train_all = [[], [], [], [], [], [], []]
    samplelist_test_all  = [[], [], [], [], [], [], []]
    senceIDs = []
    for k in range(len(args.flods)):
        flod = args.flods[k]
        print('Load data from: {}'.format(flod))
        cache_sample = os.path.join(args.cache_dir, 'cache_flod{}_{}.pkl'.format(flod, args.fever_rate))

        if os.path.isfile(cache_sample):
            with open(cache_sample, "rb") as f: 
                [samplelist_train_flod, samplelist_test_flod] = pickle.load(f)
        else:
            data_list = get_data_from_datasetfile(args, flod)
            samplelist_train_flod, samplelist_test_flod = get_samplelist(args, data_list)

            with open(cache_sample, "wb") as f: 
                pickle.dump([samplelist_train_flod, samplelist_test_flod], f)

        # data_list = get_data_from_datasetfile(args, flod)
        # samplelist_train_flod, samplelist_test_flod = get_samplelist(args, data_list)

        senceIDs = [k]*len(samplelist_train_flod[0])
        samplelist_train_flod.append(senceIDs) 

        senceIDs = [k]*len(samplelist_test_flod[0])
        samplelist_test_flod.append(senceIDs) 

        # Cross-environment testing (The training set and test set are collected from different environments.)
        if len(args.crosse_test)>3:
            if flod==args.crosse_test:
                for i in range(len(samplelist_test_flod)):
                    samplelist_test_all[i] += samplelist_test_flod[i]
            else:
                for i in range(len(samplelist_train_flod)):
                    samplelist_train_all[i]  += samplelist_train_flod[i]
        else:
            for i in range(len(samplelist_train_flod)):
                samplelist_train_all[i] += samplelist_train_flod[i]
                samplelist_test_all[i]  += samplelist_test_flod[i]

    # DataLoader 
    dataset_train = Dataset_TIF(args, datalist=samplelist_train_all)
    dataset_test  = Dataset_TIF(args, datalist=samplelist_test_all)

    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.bs,      shuffle=True,  num_workers=args.nw)
    dataloader_test  = torch.utils.data.DataLoader(dataset=dataset_test,  batch_size=args.bs_test, shuffle=False, num_workers=args.nw)

    return dataloader_train, dataloader_test

def main():

    # label_mask, label_vector
    label_mask = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])
    label_mask = label_mask.view(-1)
    print(label_mask)

if __name__ == '__main__':
    main()