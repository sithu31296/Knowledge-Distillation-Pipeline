import argparse
import yaml
import torch
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet50, resnet18

from datasets.imagenet import ImageNet
from utils.utils import setup_cuda
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    top1_acc, top5_acc = 0.0, 0.0

    for iter, (img, lbl) in enumerate(dataloader):
        img = img.to(device)
        lbl = lbl.to(device)

        pred = model(img)

        acc1, acc5 = accuracy(pred, lbl, topk=(1, 5))
        top1_acc += acc1
        top5_acc += acc5
        
    top1_acc /= iter + 1
    top5_acc /= iter + 1

    return 100*top1_acc, 100*top5_acc


def main(cfg):
    setup_cuda()
    device = torch.device(cfg['DEVICE'])
    save_dir = Path(cfg['SAVE_DIR'])
    model_save_path = save_dir / f"{cfg['MODEL']['STUDENT']}_distilled.pth"

    student_model = resnet18()
    student_model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
    student_model = student_model.to(device)
    teacher_model = resnet50(pretrained=True)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    val_transform = transforms.Compose(
        transforms.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    val_dataset = ImageNet(cfg['DATASET']['ROOT'], split='val', transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    student_top1_acc, student_top5_acc = evaluate(student_model, val_dataloader, device)
    teacher_top1_acc, teacher_top5_acc = evaluate(teacher_model, val_dataloader, device)

    table = [
        ['Teacher', teacher_top1_acc, teacher_top5_acc],
        ['Student', student_top1_acc, student_top5_acc],
    ]

    print(tabulate(table, headers=['Top 1 Accuracy', 'Top 5 Accuracy'], numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)
