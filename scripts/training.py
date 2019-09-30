import os
import time
import torch
import argparse
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from dataset import *
from backbone.swapnet_v2 import swap_net_v2
from backbone.MobileNet_V2 import MobileNetV2
from tensorboardX import SummaryWriter






def train():
    model.train()
    global_step = 1
    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()
        model.zero_grad()
        output = model(sample["img"].to(device))
        output = torch.sigmoid(output).to(device)
        loss = criterion(output, sample['target'].to(device)).to(device)
        output_np = output.detach().cpu().numpy()
        pos_ind = np.where(output_np >= 0.5)
        neg_ind = np.where(output_np < 0.5)
        output_np[pos_ind] = 1
        output_np[neg_ind] = 0
        # two dim
        # pos_ind = np.where((output_np[:, 1] >= 0.5) & (output_np[:, 0] < 0.5))
        # neg_ind = np.where((output_np[:, 1] < 0.5) & (output_np[:, 0] >= 0.5))
        # output_np[pos_ind] = [0, 1]
        # output_np[neg_ind] = [1, 0]
        acc = np.equal(output_np, sample['target'].detach().cpu().numpy()).mean()
        print(
            'Epoch {}, Iteration {}, total loss = {:.3f}, train_acc= {:.2%}'.format(
                epoch,
                i,
                loss,
                acc
            )
        )
        # writer.add_scalar('total_loss', loss, global_step=global_step)
        if global_step == 0:
            writer.add_graph(model.module, (sample['img']))

        global_step += 1
        loss.backward()
        optimizer.step()


def eval_training():
    test_loss = 0.0  # cost function error
    correct = 0.0
    model.eval()
    global best_acc

    for sample in test_loader:
        outputs = model(sample['img'].to(device))
        outputs = torch.sigmoid(outputs).to(device)
        loss = criterion(outputs, sample['target'].to(device)).to(device)
        test_loss += loss.item()
        output_np = outputs.detach().cpu().numpy()
        # one dim:
        pos_ind = np.where(output_np >= 0.5)
        neg_ind = np.where(output_np < 0.5)
        output_np[pos_ind] = 1
        output_np[neg_ind] = 0
        correct += np.equal(output_np, sample['target'].detach().cpu().numpy()).sum()
        i=0
    acc = correct / len(test_loader.dataset)
        # two dim:
        # pos_ind = np.where((output_np[:, 1] >= 0.5) & (output_np[:, 0] < 0.5))
        # neg_ind = np.where((output_np[:, 1] < 0.5) & (output_np[:, 0] >= 0.5))
        # output_np[pos_ind] = [0, 1]
        # output_np[neg_ind] = [1, 0]
    # acc = correct / 2 / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2%}'.format(
        test_loss / len(test_loader.dataset),
        acc
    ))

    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state_dict = model.module.state_dict()
        torch.save(state_dict, './checkpoint/'+str(args.distance)+'_'+args.day_or_night+'_model_best.pth')
        best_acc = acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_label_path', type=str,
                        default="/data/yzxia/projects/light_state_recognition/data/labels",
                        required=False,
                        help='path to the training label of dataset')
    parser.add_argument('--test_label_path', type=str,
                        default="/data/yzxia/projects/light_state_recognition/data/labels",
                        required=False,
                        help='path to the testing label of dataset')
    parser.add_argument('--day_or_night', type=str, default='night', required=False,
                        help='choose the day or night')
    parser.add_argument('--distance', type=int, default=30, required=False,
                        help='set the threshold of distance')
    parser.add_argument('--task', type=str, default='break', required=False,
                        help='break or turn')
    parser.add_argument('--model_scale', type=float, default=1, required=False,
                        help='set scale of model')
    parser.add_argument('--model_channels', type=list, default=[3, 8, 3], required=False,
                        help='set channels of model')
    parser.add_argument('--lr', type=float, default=0.01, required=False,
                        help='choose the learning rate')
    parser.add_argument('--nums_epochs', type=int, default=80, required=False,
                        help='set the nums of epoch')
    parser.add_argument('--gpus',  default=[0], help='gpus available for training')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='set the size of batch')
    parser.add_argument('--test_batch_size', type=int, default=8, required=False,
                        help='set the size of test batch')
    parser.add_argument('--model', type=str, default='swapnet', help='choose the back bone')
    parser.add_argument('--if_sever', type=bool, default=False)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device: ',device)

    if args.model == 'swapnet':
        model = swap_net_v2(scale=args.model_scale, n=args.model_channels)
    if args.model == 'mobilenet':
        model = MobileNetV2()
    if device == 'cuda':
        model = torch.nn.DataParallel(model,device_ids=args.gpus).cuda()
        print("Data paralleling...")
        cudnn.benchmark = True
    best_acc = 0
    global_step = 1
    train_dataset = Tail(label_path=args.train_label_path, day_or_night=args.day_or_night, distance=args.distance,
                         task=args.task, train=True, if_sever=args.if_sever)
    test_dataset = Tail(label_path=args.test_label_path, day_or_night=args.day_or_night, distance=args.distance,
                        task=args.task, train=False, if_sever=args.if_sever)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,num_workers=4 )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.BCELoss()
    writer = SummaryWriter()
    for epoch in range(1, args.nums_epochs + 1):
        train()
        print("Testing...")
        eval_training()
    state_dict = model.module.state_dict()
    torch.save(state_dict, './checkpoint/'+str(args.distance)+'_'+args.day_or_night+'_model_last.pth')
    writer.close()
