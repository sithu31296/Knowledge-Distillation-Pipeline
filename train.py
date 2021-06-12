import argparse
import os
import time
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models.resnet import resnet18, resnet50

from utils.loss import KDLoss
from utils.metrics import accuracy
from utils.utils import fix_seeds, setup_cuda
from datasets.imagenet import ImageNet
from val import evaluate


def main(cfg):
    start = time.time()

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cuda()

    save_dir = Path(cfg['TRAIN']['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()

    best_top1_acc, best_top5_acc = 0.0, 0.0
    epochs = cfg['TRAIN']['EPOCHS']
    device = torch.device(cfg['DEVICE'])
    val_interval = cfg['TRAIN']['EVAL_INTERVAL']
    kd_dataset_subset = cfg['KD']['SUBSET']
    save_dir = Path(cfg['SAVE_DIR'])
    model_save_path = save_dir / f"{cfg['MODEL']['STUDENT']}_distilled.pth"

    student_model = resnet18()
    student_model = student_model.to(device)
    teacher_model = resnet50(pretrained=True)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    train_transform = transforms.Compose(
        transforms.RandomSizedCrop(cfg['TRAIN']['IMAGE_SIZE']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    val_transform = transforms.Compose(
        transforms.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    train_dataset = ImageNet(cfg['DATASET']['ROOT'], split='train', transform=train_transform)
    val_dataset = ImageNet(cfg['DATASET']['ROOT'], split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)
    
    if kd_dataset_subset < 1.0:
        trainset_size = len(train_dataset)
        indices = list(range(trainset_size))
        split = int(torch.floor(kd_dataset_subset * trainset_size))
        random.shuffle(indices)

        train_subset_sampler = SubsetRandomSampler(indices[:split])
        train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], sampler=train_subset_sampler,shuffle=True, num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True)

    loss_fn = KDLoss(cfg['KD']['ALPHA'], cfg['KD']['TEMP'])
    optimizer = SGD(student_model.parameters(), lr=cfg['TRAIN']['LR'])
    
    assert cfg['TRAIN']['STEP_LR']['STEP_SIZE'] < cfg['TRAIN']['EPOCHS'], "Step LR scheduler's step size must be less than number of epochs"
    scheduler = StepLR(optimizer, cfg['TRAIN']['STEP_LR']['STEP_SIZE'], cfg['TRAIN']['STEP_LR']['GAMMA'])

    iters_per_epoch = int(len(train_dataset)) / cfg['TRAIN']['BATCH_SIZE']
    writer = SummaryWriter(save_dir / 'logs')

    for epoch in range(1, epochs+1):
        student_model.train()
        pbar = tqdm(enumerate(train_loader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")
        train_loss = 0.0
        
        for iter, (img, lbl) in enumerate(train_loader):
            img = img.to(device)
            lbl = lbl.to(device)

            pred_student = student_model(img)

            with torch.no_grad():
                pred_teacher = teacher_model(img)

            loss = loss_fn(pred_student, pred_teacher, lbl).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = scheduler.get_last_lr()
            train_loss += loss
            scheduler.step()

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss:.8f}")

        train_loss /= iter + 1

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)

        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch % val_interval == 0) and (epoch >= val_interval):
            print('Evaluating...')

            top1_acc, top5_acc = evaluate(student_model, val_loader, device)

            print(f"Top-1 Accuracy: {top1_acc:>0.1f} Top-5 Accuracy: {top5_acc:>0.1f}")

            writer.add_scalar('val/Top1_Acc', top1_acc, epoch)
            writer.add_scalar('val/Top5_Acc', top5_acc, epoch)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(student_model.state_dict(), model_save_path)

                print(f"Best Top-1 Accuracy: {best_top1_acc:>0.1f} Best Top-5 Accuracy: {best_top5_acc:>0.5f}")

    # evaluating teacher model
    teacher_top1_acc, teacher_top5_acc = evaluate(teacher_model, val_loader, device)

    writer.close()
    pbar.close()

    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = [
        ['Teacher', teacher_top1_acc, teacher_top5_acc],
        ['Student', best_top1_acc, best_top5_acc],
    ]

    print(f"Total Training Time: {total_time}")
    print(tabulate(table, headers=['Top 1 Accuracy', 'Top 5 Accuracy'], numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)