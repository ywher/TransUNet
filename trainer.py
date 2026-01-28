import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.classes import CLASSES

def _get_state_dict(model):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    class_names = CLASSES[args.dataset.lower()]
    logging.info(f"Class names: {class_names}")
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    logging.info("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    early_stop = False
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_camus(args, model, snapshot_path):
    from datasets.dataset_camus import CamusDataset, RandomGenerator, ValTransform

    list_path = os.path.join(args.list_dir, args.camus_split, 'labeled.txt')
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"CAMUS split file not found: {list_path}")
    val_list_path = os.path.join(args.list_dir, 'val.txt')
    if not os.path.exists(val_list_path):
        raise FileNotFoundError(f"CAMUS val file not found: {val_list_path}")

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    class_names = CLASSES[args.dataset.lower()]
    logging.info(f"Class names: {class_names}")
    batch_size = args.batch_size * args.n_gpu

    db_train = CamusDataset(root_path=args.root_path, list_path=list_path, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = CamusDataset(root_path=args.root_path, list_path=val_list_path, split="val",
                          transform=transforms.Compose(
                              [ValTransform(output_size=[args.img_size, args.img_size])]))
    logging.info("The length of train set is: {}".format(len(db_train)))
    logging.info("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                           worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_miou = -1.0
    epochs_without_improve = 0
    early_stop = False
    def evaluate():
        model.eval()
        num_cls = num_classes
        intersection = torch.zeros(num_cls, device='cuda')
        union = torch.zeros(num_cls, device='cuda')
        denom = torch.zeros(num_cls, device='cuda')
        with torch.no_grad():
            for sampled_batch in valloader:
                images, labels = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
                outputs = model(images)
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                for c in range(num_cls):
                    pred_c = preds == c
                    label_c = labels == c
                    inter = (pred_c & label_c).sum()
                    union_c = (pred_c | label_c).sum()
                    denom_c = pred_c.sum() + label_c.sum()
                    intersection[c] += inter
                    union[c] += union_c
                    denom[c] += denom_c
        valid_iou = union > 0
        ious = intersection[valid_iou] / union[valid_iou]
        miou = ious.mean().item() if valid_iou.any() else 0.0
        
        valid_dice = denom > 0
        dices = 2 * intersection[valid_dice] / denom[valid_dice]
        mdice = dices.mean().item() if valid_dice.any() else 0.0
        
        model.train()
        return miou, mdice, ious, dices

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        logging.info(f'\nStarting epoch {epoch_num + 1}/{max_epoch}')
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            if (i_batch + 1) % (len(trainloader) // 20) == 0:
                logging.info(f"iteration {iter_num} : loss : {loss.item():.3f}, loss_ce: {loss_ce.item():.3f}, loss_dice: {loss_dice.item():.3f}")

            if iter_num % 100 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        if (epoch_num + 1) % args.eval_interval == 0:
            miou, mdice, ious, dices = evaluate()
            writer.add_scalar('val/miou', miou, epoch_num + 1)
            writer.add_scalar('val/dice', mdice, epoch_num + 1)
            logging.info(f'Validation epoch {epoch_num + 1}: mIoU {miou:.4f}, Dice {mdice:.4f}')
            #class name + iou, dice
            for idx, class_name in enumerate(class_names):
                if idx < len(ious):
                    logging.info(f'  {class_name}: IoU {ious[idx]:.4f}, Dice {dices[idx]:.4f}')

            if miou > best_miou:
                best_miou = miou
                epochs_without_improve = 0
                best_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(_get_state_dict(model), best_path)
                logging.info(f"save best model to {best_path}")
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= args.early_stop_patience:
                logging.info("Early stopping: no improvement on mIoU")
                save_mode_path = os.path.join(snapshot_path, 'final_model.pth')
                torch.save(_get_state_dict(model), save_mode_path)
                logging.info("save final model to {}".format(save_mode_path))
                early_stop = True
                iterator.close()
                break

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(_get_state_dict(model), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'final_model.pth')
            torch.save(_get_state_dict(model), save_mode_path)
            logging.info("save final model to {}".format(save_mode_path))
            iterator.close()
            break

        if early_stop:
            break

    writer.close()
    return "Training Finished!"