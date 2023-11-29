import tqdm, cv2, os, pdb, torch, pickle, time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
from einops import rearrange

import utils, args, dataset, model

def show(args, epoch, index, vector_pool, face_pool, depth_pool, depth_vector, label_mask, label_vector, heat, pre):

    vector_pool  = vector_pool[0].cpu().numpy().squeeze()
    face_pool    = face_pool[0].cpu().numpy().squeeze()
    depth_pool   = depth_pool[0].cpu().numpy().squeeze()
    depth_vector = depth_vector[0].cpu().numpy().squeeze()
    label_mask   = label_mask[0].cpu().numpy().squeeze()
    label_vector = label_vector[0].cpu().numpy().squeeze()

    face_pool = (face_pool-face_pool.min())*255./(face_pool.max()-face_pool.min()+0.0001)
    face_pool = utils.gray2iron(face_pool)
    cv2.imwrite(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_face_pool.jpg'.format(args.workID, epoch, index)), face_pool)
    
    depth_pool = (depth_pool-depth_pool.min())/(depth_pool.max()-depth_pool.min())
    depth_pool = cv2.resize(depth_pool, (args.face_size *args.pool_size, args.face_size *args.pool_size))
    depth_pool = depth_pool*255
    cv2.imwrite(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_depth_pool.jpg'.format(args.workID, epoch, index)), depth_pool)

    plt.clf()
    plt.plot(range(len(depth_vector)), depth_vector)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_depth_vector.jpg'.format(args.workID, epoch, index)))

    label_mask = cv2.resize(label_mask, (args.face_size *args.pool_size, args.face_size *args.pool_size))
    cv2.imwrite(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_label_mask.jpg'.format(args.workID, epoch, index)), label_mask*255)

    # plt.clf()
    # plt.plot(range(len(label_vector)), label_vector)
    # plt.savefig(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_label_vector.jpg'.format(args.workID, epoch, index)))

    heat = heat[0].detach().cpu().numpy().squeeze()
    # label_mask = label_mask[0].cpu().numpy().squeeze()
    pre = pre[0].detach().cpu().numpy().squeeze()
    pre[pre>1]=1
    # label_vector = label_vector[0].cpu().numpy().squeeze()
    # label_vector = label_vector+0.1

    heat = (heat-heat.min())*255./(heat.max()-heat.min()+0.0001)
    heat = cv2.resize(heat, (args.face_size *args.pool_size, args.face_size *args.pool_size))
    # seg_show = np.concatenate([heat, label_mask], axis=1)
    cv2.imwrite(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_heat.jpg'.format(args.workID, epoch, index)), heat)

# ==========================================================================================================================

    label_faces = []
    for i in range(args.pool_size):
        for j in range(args.pool_size):
            label_face = label_mask[i*args.face_size :(i+1)*args.face_size, j*args.face_size :(j+1)*args.face_size]
            label_faces.append(label_face.max())

    output_faces = []
    for i in range(args.pool_size):
        for j in range(args.pool_size):
            output_face = heat[i*args.face_size :(i+1)*args.face_size, j*args.face_size :(j+1)*args.face_size]
            output_faces.append(output_face)


    ds = []
    for anchor_index, [label1, output_face] in enumerate(zip(label_faces, output_faces)):
        anchor = output_face
        if label1==1:
            continue    
        for index2, [label2, output_face] in enumerate(zip(label_faces, output_faces)):
            if index2<anchor_index:
                continue
            d = anchor.mean()-output_face.mean()
            if label2==1:
                continue
            if d==0:
                continue
            ds.append(d)

    plt.clf()
    ds = np.array(ds)
    # print(ds.min())
    # print(ds.max())
    # ds = (ds-ds.min())/(ds.max()-ds.min())

    plt.hist(ds, bins=np.arange(-120, 120, 3), label='Normal', color='b')
    
    plt.xlim(-120, 120)
    plt.ylim(0, 150)
    # plt.tick_params(labelsize=16) 
    # plt.legend(prop={'size': 14})
    plt.savefig(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_output_hist.jpg'.format(args.workID, epoch, index)), bbox_inches='tight')



# ==========================================================================================================================

    plt.clf()
    # plt.ylim(-0.2, 1.2)
    # plt.plot(range(len(label_vector)), label_vector, 'ro--')
    # # plt.scatter(range(len(label_vector)), label_vector)
    # plt.plot(range(len(pre)), pre, 'bo--')
    # # plt.scatter(range(len(pre)), pre)

    pre = (pre-min(pre))/(max(pre)-min(pre))
    x_fever = []
    x_normal = []
    res_fever = []
    res_normal = []
    for i, [res, gt] in enumerate(zip(pre, label_vector)):
        if gt==0:
            x_normal.append(i)
            res_normal.append(res)
        else:
            x_fever.append(i)
            res_fever.append(res)

    plt.clf()
    plt.ylim(-0.1, 1.1)
    plt.tick_params(labelsize=16) 
    plt.xlabel('Sample', size=16)
    plt.ylabel('Score', size=16)  
    plt.scatter(x_normal, res_normal, c='b', )
    plt.scatter(x_fever,  res_fever, s=200, c='r', marker='*')

    plt.savefig(os.path.join(args.show_dir, '{}_test_epoch{}_index{}_score.jpg'.format(args.workID, epoch, index)), bbox_inches='tight')

def calculate_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    return auc

def sort_heat(heat):

    heat_mean = heat.mean(dim=-1)
    heat_mean_sortidx = heat_mean.sort()[1]

    heat_new = []
    for idx in heat_mean_sortidx:
        heat_new.append(heat[idx])
    heat_new = torch.cat(heat_new)
    
    return heat_new

def Test(args, dataloader_test, Network, epoch):
    Network.eval()

    correct = 0
    total   = 0
    results = []
    gts     = []
    heats   = []
    cats    = []
    times = []
    iterations =  tqdm.tqdm(enumerate(dataloader_test), desc="Test  epoch:{}".format(epoch), total=len(dataloader_test), ncols=150, leave=False)
    for index, [vector_pool, face_pool, depth_pool, depth_vector, label_mask, label_vector, senceID] in iterations:

        vector_pool  = vector_pool  .to(args.device)
        face_pool    = face_pool    .to(args.device)
        depth_pool   = depth_pool   .to(args.device)
        depth_vector = depth_vector .to(args.device)
        label_mask   = label_mask   .to(args.device)
        label_vector = label_vector .to(args.device)

        # time1 = time.time()
        pre, heat, feature = Network(face_pool, depth_pool, depth_vector)
        # time2 = time.time()
        # times.append(time2-time1)
        # if len(times)>110:
        #     print(np.mean(times[10:]))
        #     break

        if args.testonly:
            show(args, epoch, index, vector_pool, face_pool, depth_pool, depth_vector, label_mask, label_vector, heat, pre)

        senceID = senceID.detach().cpu().numpy()
        heat = rearrange(heat, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = args.patch_size, p2 = args.patch_size)
        for b in range(pre.shape[0]):
            p = pre[b]
            h = heat[b]
            f = feature[b]
            l = label_vector[b]
            s = senceID[b]

            results += list(p.detach().cpu().numpy().squeeze()[-args.num_new:])
            gts     += list(l.cpu().numpy().squeeze()[-args.num_new:])

            h = h.squeeze()
            h = h.view(-1)
            heats.append(h.detach().cpu().numpy().squeeze())
            cats.append(s)

        if index>10:
            auc = calculate_auc(gts, results)
            iterations.set_postfix(Auc='{:3.5f}'.format(auc))

    if args.testonly:
        heats_file = os.path.join(args.results_dir, 'heats_{}.pkl'.format(args.Mode))
        with open(heats_file, "wb") as f: 
            pickle.dump([heats, cats], f)

        results_file = os.path.join(args.results_dir, 'results_{}.pkl'.format(args.Mode))
        with open(results_file, "wb") as f: 
            pickle.dump([results, gts], f)

        utils.show_scores(args, results, gts, cats, epoch)
        utils.show_tsne(args, heats, cats, epoch)
        
        # Test Num New
        # sample_index = np.linspace(start=0, stop=len(results)-1, num=int(len(results)/args.num_new)).astype(int)
        # results = np.array(results)[sample_index]
        # gts = np.array(gts)[sample_index]

    return auc, results, gts, heats, cats


if __name__=='__main__':
    # arg & log
    args, logging = args.get_args()
    args.testonly = True
    epoch = 0

    weight_file = './work/{}/weights/best.pth'.format(args.Mode) 

    # Dataloader
    dataloader_train, dataloader_test = dataset.get_dataloader_TIF(args)

    # Network
    Network = model.FeverTransformer(args=args).to(args.device)

    # Load weight
    Network.load_state_dict(torch.load(weight_file))

    # Test
    auc, results, gts, heats, cats = Test(args, dataloader_test, Network, epoch)

    logging.info('Mode:{} | workID:{} | epoch:{:>3d} | auc:{:>9.5f} '.format(args.Mode, args.workID, epoch, auc) )
    print       ('Mode:{} | workID:{} | epoch:{:>3d} | auc:{:>9.5f} '.format(args.Mode, args.workID, epoch, auc) )

# ==============================================================================================================================

    # heats_file = os.path.join('./work/{}/results/heats_{}.pkl'.format(args.Mode, args.Mode))
    # results_file = os.path.join('./work/{}/results/results_{}.pkl'.format(args.Mode, args.Mode))

    # with open(results_file, "rb") as f: 
    #     [results, gts] = pickle.load(f)
    # with open(heats_file, "rb") as f: 
    #     [heats, cats] = pickle.load(f)

    # utils.show_scores(args, results, gts, cats, epoch)
    # utils.show_tsne(args, heats, cats, epoch)

    # auc = calculate_auc(gts, results)

    # logging.info('Mode:{} | workID:{} | epoch:{:>3d} | auc:{:>9.5f} '.format(args.Mode, args.workID, epoch, auc) )
    # print       ('Mode:{} | workID:{} | epoch:{:>3d} | auc:{:>9.5f} '.format(args.Mode, args.workID, epoch, auc) )