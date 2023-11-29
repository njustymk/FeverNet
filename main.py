import os, torch, pickle
# from torchstat import stat
# from torchsummary import summary

import args, dataset, model, train, test, eval 

def main():
    # Dataloader
    dataloader_train, dataloader_test = dataset.get_dataloader_TIF(args)

    # Network
    Network = model.FeverTransformer(args=args).to(args.device)
    # summary(Network, [(1,256,256), (1, 64, 64), (1, 64)])
    # summary(Network, (1,256,256))

    # Optimizer
    optimizer = torch.optim.Adam(Network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.8)
    
    # Lossfunc
    lossfunc1 = torch.nn.MSELoss(reduction='mean')
    lossfunc2 = torch.nn.MSELoss(reduction='mean')

    # Log
    logging.info('===>Sample train size<===: {}'.format(len(dataloader_train)))
    logging.info('===>Sample test  size<===: {}'.format(len(dataloader_test )))
    logging.info('===>Network<===\n'    +str(Network  ))
    logging.info('===>Optimizer<===\n'  +str(optimizer))
    logging.info('===>Scheduler<===\n'  +str(scheduler))
    logging.info('===>Lossfunc1<===\n'  +str(lossfunc1))
    logging.info('===>Lossfunc2<===\n'  +str(lossfunc2))

    auc_best = 0

    for epoch in range(1, args.epochs+1):

        # Train
        Network, loss = train.Train(args, dataloader_train, Network, lossfunc1, lossfunc2, optimizer, scheduler, epoch)
        
        # Test
        auc, results, gts, heats, cats = test.Test(args, dataloader_test, Network, epoch)

        if auc>auc_best:
            auc_best = auc
            torch.save(Network.state_dict(), os.path.join(args.weights_dir, 'best.pth'))
            
        # Save weights
        torch.save(Network.state_dict(), os.path.join(args.weights_dir, 'epoch_{}.pth'.format(epoch)))

        # Log
        print       ('Mode:{} | workID:{} | epoch:{:>3d} | loss:{:>8.5f} | auc:{:>9.5f} | auc_best:{:>9.5f}'\
            .format(args.Mode, args.workID, epoch, loss, auc, auc_best) )
        logging.info('Mode:{} | workID:{} | epoch:{:>3d} | loss:{:>8.5f} | auc:{:>9.5f} | auc_best:{:>9.5f}'\
            .format(args.Mode, args.workID, epoch, loss, auc, auc_best) )

    # Test best
    print('Test best')
    args.testonly = True

    # Load best weight
    Network.load_state_dict(torch.load(os.path.join(args.weights_dir, 'best.pth')))

    # Test
    auc, results, gts, heats, cats = test.Test(args, dataloader_test, Network, epoch=0)

    logging.info('Mode:{} | workID:{} | epoch:{:>3d} | auc:{:>9.5f} '.format(args.Mode, args.workID, epoch, auc) )
    print       ('Mode:{} | workID:{} | epoch:{:>3d} | auc:{:>9.5f} '.format(args.Mode, args.workID, epoch, auc) )

if __name__ == '__main__':
    
    # arg & log
    args, logging = args.get_args()

    main()

