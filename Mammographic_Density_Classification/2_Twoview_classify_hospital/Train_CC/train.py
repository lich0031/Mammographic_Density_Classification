import sys
sys.path.extend(['../../', '../'])
import os, time, argparse, random
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from Datasets import breast_classify_hospital, Compose, Resize, ToTensor, Normalize, RandomHorizontallyFlip, \
    RandomVerticallyFlip, RandomRotate
from Models import resnet50, resnet34, resnet18, resnet101, resnet152, resnet50se, resnet50_dilated, resnet50_dcse
from Models import resnet50noshare_dcse, resnet50noshare, resnet50_dcsegse, resnet50_dcseresgse
from Models import resnet50_dcseadd, resnet50_dcsemul
import torch.nn.functional as F
from Utils import PolyLR

def parse_args():
    parser = argparse.ArgumentParser(description='Classification for Breast',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='resnet50noshare_dcse', type=str, help='unet, ...')
    parser.add_argument('--pretrain', default=False, type=bool, help='...')
    parser.add_argument('--weighted_sampling', default=0, type=int, help='weighted sample')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--aug', default=2, type=int, help='data augmentation')
    parser.add_argument('--gpu_order', default='0', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='num epoch')
    parser.add_argument('--img_size', default=224, type=int, help='512')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--checkpoint', default='checkpoint/')
    parser.add_argument('--params_name', default='resnet50_params.pkl')
    parser.add_argument('--log_name', default='resnet50.log', type=str, help='log name')
    parser.add_argument('--history', default='history')
    parser.add_argument('--cudnn', default=0, type=int, help='cudnn')
    parser.add_argument('--split_order', default=1, type=int, help='...')
    args = parser.parse_args()
    return args

def record_params(args):
    localtime = time.asctime(time.localtime(time.time()))
    logging.info('Breast_density_classification(Data: {}) \n'.format(localtime))
    logging.info('**************Parameters***************')

    args_dict = args.__dict__
    for key, value in args_dict.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('**************Parameters***************\n')


def build_model(model_name, num_classes, pretrain):
    if model_name == 'resnet50':
        net = resnet50(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet18':
        net = resnet18(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet34':
        net = resnet34(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet101':
        net = resnet101(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet152':
        net = resnet152(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet50se':
        net = resnet50se(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet50dilated':
        net = resnet50_dilated(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet50dcse':
        net = resnet50_dcse(num_classes=num_classes, pretrain=pretrain)

    elif model_name == 'resnet50_dcsegse':
        net = resnet50_dcsegse(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet50_dcseresgse':
        net = resnet50_dcseresgse(num_classes=num_classes, pretrain=pretrain)

    elif model_name == 'resnet50noshare':
        net = resnet50noshare(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet50noshare_dcse':
        net = resnet50noshare_dcse(num_classes=num_classes, pretrain=pretrain)

    elif model_name == 'resnet50_dcseadd':
        net = resnet50_dcseadd(num_classes=num_classes, pretrain=pretrain)
    elif model_name == 'resnet50_dcsemul':
        net = resnet50_dcsemul(num_classes=num_classes, pretrain=pretrain)

    else:
        print('wait a minute')
    return net

def Train(train_root, train_csv, test_csv):

    # parameters
    args = parse_args()
    record_params(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_order
    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    np.random.seed(args.torch_seed)
    random.seed(args.torch_seed)

    if args.cudnn == 0:
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
        cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4
    net = build_model(args.model_name, num_classes, args.pretrain)

    # resume
    checkpoint_name_loss = os.path.join(args.checkpoint, args.params_name.split('.')[0]+'_loss.'+args.params_name.split('.')[-1])
    checkpoint_name_acc = os.path.join(args.checkpoint, args.params_name.split('.')[0]+'_acc.'+args.params_name.split('.')[-1])

    if args.resume != 0:
        logging.info('Resuming from checkpoint...')
        checkpoint = torch.load(checkpoint_name_loss)
        best_loss = checkpoint['loss']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        net.load_state_dict(checkpoint['net'])
    else:
        best_loss = float('inf')
        best_acc = 0.0
        start_epoch = 0
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    end_epoch = start_epoch + args.num_epoch

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # data
    img_size = args.img_size
    ## train
    train_aug = Compose([
        Resize(size=(img_size, img_size)),
        RandomHorizontallyFlip(),
        RandomVerticallyFlip(),
        RandomRotate(90),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5),
                  std=(0.5, 0.5, 0.5))])
    ## test
    # test_aug = train_aug
    test_aug = Compose([
        Resize(size=(img_size, img_size)),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5),
                  std=(0.5, 0.5, 0.5))])

    train_dataset = breast_classify_hospital(root=train_root, csv_file=train_csv, transform=train_aug)
    test_dataset = breast_classify_hospital(root=train_root, csv_file=test_csv, transform=test_aug)

    if args.weighted_sampling == 1:
        weights = torch.FloatTensor([1.0, 1.0, 1.5, 5.0]).to(device)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=4, shuffle=True)
    else:
        weights = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=4, shuffle=True)


    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
    #                           num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, shuffle=True)

    # loss function, optimizer and scheduler

    criterion = nn.NLLLoss(size_average=True, weight=weights).to(device)

    optimizer = Adam(net.parameters(), lr=args.lr, amsgrad=True)

    ## scheduler
    if args.lr_policy == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    if args.lr_policy == 'PolyLR':
        scheduler = PolyLR(optimizer, max_epoch=end_epoch, power=0.9)

    # training process
    logging.info('Start Training For Breast Density Classification')
    for epoch in range(start_epoch, end_epoch):
        ts = time.time()
        if args.lr_policy != 'None':
            scheduler.step()

        # train
        net.train()
        train_loss = 0.
        train_acc = 0.

        for batch_idx, (inputs1, inputs2, targets) in tqdm(enumerate(train_loader),
                                                           total=int(len(train_loader))):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            targets = targets.to(device)
            targets = targets.long()
            optimizer.zero_grad()
            outputs = net(inputs1, inputs2)
            loss = criterion(F.log_softmax(outputs, dim=1), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            accuracy = float(sum(outputs.argmax(dim=1) == targets))
            train_acc += accuracy

        train_acc_epoch = train_acc / (len(train_loader.dataset))

        train_loss_epoch = train_loss / (batch_idx + 1)
        history['train_loss'].append(train_loss_epoch)
        history['train_acc'].append(train_acc_epoch)

        # test
        net.eval()
        test_loss = 0.
        test_acc = 0.

        for batch_idx, (inputs1, inputs2, targets) in tqdm(enumerate(test_loader),
                                                 total=int(len(test_loader.dataset) / args.batch_size) + 1):
            with torch.no_grad():
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                targets = targets.to(device)
                targets = targets.long()
                outputs = net(inputs1, inputs2)
                loss = criterion(F.log_softmax(outputs, dim=1), targets)
                accuracy = float(sum(outputs.argmax(dim=1) == targets))

            test_acc += accuracy
            test_loss += loss.item()

        test_loss_epoch = test_loss / (batch_idx + 1)
        test_acc_epoch = test_acc / (len(test_loader.dataset))
        history['test_loss'].append(test_loss_epoch)
        history['test_acc'].append(test_acc_epoch)

        time_cost = time.time() - ts
        logging.info('epoch[%d/%d]: train_loss: %.3f | train_acc: %.3f | test_loss: %.3f | test_acc: %.3f || time: %.1f'
                     % (epoch + 1, end_epoch, train_loss_epoch, train_acc_epoch, test_loss_epoch, test_acc_epoch, time_cost))

        # save checkpoint
        if test_loss_epoch < best_loss:
            logging.info('Loss checkpoint Saving...')

            save_model = net
            if torch.cuda.device_count() > 1:
                save_model = list(net.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_loss_epoch,
                'acc': test_acc_epoch,
                'epoch': epoch + 1,
                'history': history
            }
            torch.save(state, checkpoint_name_loss)
            best_loss = test_loss_epoch

        if test_acc_epoch > best_acc:
            logging.info('Acc checkpoint Saving...')

            save_model = net
            if torch.cuda.device_count() > 1:
                save_model = list(net.children())[0]
            state = {
                'net': save_model.state_dict(),
                'loss': test_loss_epoch,
                'acc': test_acc_epoch,
                'epoch': epoch + 1,
                'history': history
            }
            torch.save(state, checkpoint_name_acc)
            best_acc = test_acc_epoch

args = parse_args()
if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)
if not os.path.exists(args.history):
    os.mkdir(args.history)
logging_save = os.path.join(args.history, args.log_name)
logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(logging_save)
                    ])

if __name__ == "__main__":
    args = parse_args()
    train_root = '../../JPG_Hospital_Adjusted_Multiview_Viewspecific'
    csv_root = '../../JPG_Hospital_Adjusted_Multiview_Viewspecific/Split_5fold_CC'
    train_csv = os.path.join(csv_root, 'train_data_split{}.csv'.format(args.split_order))
    test_csv = os.path.join(csv_root, 'val_data_split{}.csv'.format(args.split_order))
    Train(train_root, train_csv, test_csv)