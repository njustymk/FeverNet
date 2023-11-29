import os, time, argparse, logging, torch, random

import numpy as np

def get_args():
    workID = time.strftime("%Y%m%d%H%M%S", time.localtime())

    parser = argparse.ArgumentParser()
    
    # Work Dir
    parser.add_argument('-code_ID',          type=str,   default='code13-TIF-CNN+ViT',               help='code ID')
    parser.add_argument('-data_root',        type=str,   default='/test/data/TIF_v3/datas',          help='data dir')
    parser.add_argument('-flods',            type=list,  default=[    
                                                                # '20200825',
                                                                # '20200831',
                                                                '20201117',
                                                                '20201120',
                                                                '20201201',
                                                                        ],                           help='data flods')
    parser.add_argument('-cache_dir',        type=str,   default='../cache',                         help='cache dir')
    parser.add_argument('-work_dir',         type=str,   default='./work',                           help='work dir')
    parser.add_argument('-workID',           type=str,   default=workID,                             help='workID')
    parser.add_argument('-workID_dir',       type=str,   default='./work/{}'.format(workID),         help='workID dir')
    parser.add_argument('-weights_dir',      type=str,   default='./work/{}/weights'.format(workID), help='weights dir')
    parser.add_argument('-results_dir',      type=str,   default='./work/{}/results'.format(workID), help='results dir')
    parser.add_argument('-show_dir',         type=str,   default='./work/{}/show'.format(workID),    help='show dir')
    # Data
    parser.add_argument('-num_train',        type=int,   default=4000,                               help='Number of faces (train)')
    parser.add_argument('-num_test',         type=int,   default=1000,                               help='Number of faces (test)')
    parser.add_argument('-fever_rate',       type=float, default=0.1,                                help='Number of fever in Pool')
    parser.add_argument('-temp_min',         type=int,   default=20,                                 help='Lower facial temperature limit')
    parser.add_argument('-temp_max',         type=int,   default=40,                                 help='Upper facial temperature limit')
    parser.add_argument('-depth_max',        type=float, default=0.4,                                help='Depth Max')
    parser.add_argument('-depth_min',        type=float, default=0.1,                                help='Depth Min')
    parser.add_argument('-face_size',        type=int,   default=32,                                 help='Face size')
    parser.add_argument('-img_width',        type=int,   default=640,                                help='Image width')
    parser.add_argument('-img_height',       type=int,   default=480,                                help='Image height')
    parser.add_argument('-pool_size',        type=int,   default=8,                                  help='Pool size')
    parser.add_argument('-patch_size',       type=int,   default=8,                                  help='Feature patch size')
    parser.add_argument('-Tmap_size',        type=int,   default=64,                                 help='Mask GT size')
    parser.add_argument('-num_classes',      type=int,   default=64,                                 help='Num classes')
    # Transformer
    parser.add_argument('-channels',         type=int,   default=1,                                  help='Channels of input')
    parser.add_argument('-dim',              type=int,   default=64,                                 help='Dimensions of each patch')
    parser.add_argument('-depth',            type=int,   default=1,                                  help='depth of Self-attention')
    parser.add_argument('-heads',            type=int,   default=8,                                  help='heads of Self-attention')
    parser.add_argument('-dim_head',         type=int,   default=64,                                 help='dim head of Self-attention')
    parser.add_argument('-mlp_dim',          type=int,   default=64,                                 help='mlp dim of Self-attention')
    parser.add_argument('-dropout',          type=float, default=0.1,                                help='dropout of Self-attention')
    parser.add_argument('-emb_dropout',      type=float, default=0.1,                                help='emb_dropout of Self-attention')
    # Train
    parser.add_argument('-device',           type=str,   default='cuda',                             help='cuda or cpu')
    parser.add_argument('-deviceID',         type=str,   default='0',                                help='cuda ID')
    parser.add_argument('-lr',               type=float, default=0.0001,                             help='Lrarning Rate')
    parser.add_argument('-bs',               type=int,   default=16,                                 help='Batch size train')
    parser.add_argument('-bs_test',          type=int,   default=32,                                 help='Batch size of test')
    parser.add_argument('-nw',               type=int,   default=8,                                  help='Num workers')
    parser.add_argument('-epochs',           type=int,   default=50,                                 help='Num of epoch')
    parser.add_argument('-alpha',            type=float, default=0.5,                                help='Balance loss1 and loss2')
    parser.add_argument('-seed',             type=int,   default=2023,                               help='set seed for model')
    # Mode
    parser.add_argument('-Mode',             type=str,   default='FAM+DCB+EDCL',                             \
                                                    help='Code Mode (Base | FAM | DCB | EDCL | FAM+DCB | FAM+EDCL | DCB+EDCL | FAM+DCB+EDCL | FAM+DCB+EDCL-CTMap)')
    parser.add_argument('-testonly',         type=bool,  default=False,                              help='test?')
    parser.add_argument('-num_show',         type=int,   default=500,                                help='num show')
    parser.add_argument('-num_new',          type=int,   default=1,                                  help='num new')
    parser.add_argument('-crosse_test',      type=str,   default='0',                                help='0 | 20201117 | 20201120 | 20201201')
    args = parser.parse_args()

    if args.device=='cuda':
        os.environ['CUDA_VISIBLE_DEVICES']=args.deviceID

    if not os.path.exists(args.work_dir):    os.mkdir(args.work_dir)
    if not os.path.exists(args.cache_dir):   os.mkdir(args.cache_dir)
    if not os.path.exists(args.workID_dir):  os.mkdir(args.workID_dir)
    if not os.path.exists(args.weights_dir): os.mkdir(args.weights_dir)
    if not os.path.exists(args.results_dir): os.mkdir(args.results_dir)
    if not os.path.exists(args.show_dir):    os.mkdir(args.show_dir)

    # Log file
    logging.basicConfig(filename=os.path.join(args.workID_dir, '{}.log'.format(args.Mode)), \
        format='%(asctime)s - %(message)s', level=logging.INFO)
    
    # Save the args in the log.
    logging.info('===>Work ID<===: {}'.format(args.workID))
    for k, v in sorted(vars(args).items()):
        logging.info(str(k).ljust(16) +':' +str(v)) 
        print(str(k).ljust(16) +':' +str(v))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    return args, logging