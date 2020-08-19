from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import torch
import torch.utils.data
from opts import opts
from MOC_utils.model import create_model, load_model, save_model, load_coco_pretrained_model, load_imagenet_pretrained_model
from trainer.logger import Logger
from datasets.init_dataset import get_dataset
from trainer.moc_trainer import MOCTrainer
from inference.stream_inference import stream_inference
from ACT import frameAP
import numpy as np
import random
import tensorboardX


GLOBAL_SEED = 317


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


def main(opt):
    set_seed(opt.seed)

    torch.backends.cudnn.benchmark = True
    print()
    print('dataset: ' + opt.dataset + '   task:  ' + opt.task)
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset(opt, Dataset)

    train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train'))
    epoch_train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train_epoch'))
    val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val'))
    epoch_val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val_epoch'))

    logger = Logger(opt, epoch_train_writer, epoch_val_writer)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = opt.start_epoch

    if opt.pretrain_model == 'coco':
        model = load_coco_pretrained_model(opt, model)
    else:
        model = load_imagenet_pretrained_model(opt, model)

    if opt.load_model != '':
        model, optimizer, _, _ = load_model(model, opt.load_model, optimizer, opt.lr, opt.ucf_pretrain)

    trainer = MOCTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    print('training...')
    print('GPU allocate:', opt.chunk_sizes)
    best_ap = 0
    best_epoch = 0
    stop_step = 0
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print('eopch is ', epoch)
        log_dict_train = trainer.train(epoch, train_loader, train_writer)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'train')
            logger.write('train: {} {:8f} | '.format(k, v))
        logger.write('\n')
        if opt.save_all and not opt.auto_stop:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            model_name = 'model_[{}]_{}.pth'.format(epoch, time_str)
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])
        else:
            model_name = 'model_last.pth'
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])

        # this step evaluate the model
        if opt.val_epoch:
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader, val_writer)
            for k, v in log_dict_val.items():
                logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'val')
                logger.write('val: {} {:8f} | '.format(k, v))
        logger.write('\n')

        if opt.auto_stop:
            tmp_rgb_model = opt.rgb_model
            tmp_flow_model = opt.flow_model
            if opt.rgb_model != '':
                opt.rgb_model = os.path.join(opt.rgb_model, model_name)
            if opt.flow_model != '':
                opt.flow_model = os.path.join(opt.flow_model, model_name)
            stream_inference(opt)
            ap = frameAP(opt, print_info=opt.print_log)
            os.system("rm -rf tmp")
            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                saved1 = os.path.join(opt.save_dir, model_name)
                saved2 = os.path.join(opt.save_dir, 'model_best.pth')
                os.system("cp " + str(saved1) + " " + str(saved2))
            if stop_step < len(opt.lr_step) and epoch >= opt.lr_step[stop_step]:
                model, optimizer, _, _ = load_model(
                    model, os.path.join(opt.save_dir, 'model_best.pth'), optimizer, opt.lr)
                opt.lr = opt.lr * 0.1
                logger.write('Drop LR to ' + str(opt.lr) + '\n')
                print('Drop LR to ' + str(opt.lr))
                print('load epoch is ', best_epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt.lr
                torch.cuda.empty_cache()
                trainer = MOCTrainer(opt, model, optimizer)
                trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
                stop_step = stop_step + 1

            opt.rgb_model = tmp_rgb_model
            opt.flow_model = tmp_flow_model

        else:
            # this step drop lr
            if epoch in opt.lr_step:
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                logger.write('Drop LR to ' + str(lr) + '\n')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
    if opt.auto_stop:
        print('best epoch is ', best_epoch)

    logger.close()


if __name__ == '__main__':
    os.system("rm -rf tmp")
    opt = opts().parse()
    main(opt)
