from backbone.swapnet_v2 import swap_net_v2
import os
import torch
import torch.utils.data
import torch.nn as nn
import cv2
from dataset.tail import Tail
import numpy as np
import torch.backends.cudnn as cudnn


def changename():
    file_list = os.listdir('C:\\Users\\DELL\\Desktop\\test\\')
    for ind, i in enumerate(file_list):
        os.rename('C:\\Users\\DELL\\Desktop\\gt\\' + i,
                  'C:\\Users\\DELL\\Desktop\\gt\\' + "{:03d}".format(ind) + '.jpg')
def eval_training(distance,day_or_night,car_type):
    model = swap_net_v2(scale=1, n=[3, 8, 3])
    correct=0
    # changename()
    state_dict = torch.load('./checkpoint/' + str(distance) + '_' + day_or_night + '_model_best.pth',
                            map_location='cpu')
    print("Loading checkpoint", './checkpoint/' + str(distance) + '_' + day_or_night + '_model_best.pth')
    # state_dict = torch.load('./model_best_night.pth',
    #                         map_location='cpu')
    # all_data = os.listdir("C:\\Users\\DELL\\Desktop\\gt")
    # all_data.sort()
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    data = Tail("D:\\light_state_recognition_data_20190909\\tail", day_or_night=day_or_night, distance=50,
                car_type=car_type, train=False)
    test_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    for sample in test_loader:
        outputs = model(sample['img'])
        outputs = torch.sigmoid(outputs)
        output_np = outputs.detach().cpu().numpy()
        # one dim:
        pos_ind = np.where(output_np >= 0.5)
        neg_ind = np.where(output_np < 0.5)
        output_np[pos_ind] = 1
        output_np[neg_ind] = 0
        correct += np.equal(output_np, sample['target'].detach().cpu().numpy()).sum()

    acc = correct / len(test_loader.dataset)
    # two dim:
    # pos_ind = np.where((output_np[:, 1] >= 0.5) & (output_np[:, 0] < 0.5))
    # neg_ind = np.where((output_np[:, 1] < 0.5) & (output_np[:, 0] >= 0.5))
    # output_np[pos_ind] = [0, 1]
    # output_np[neg_ind] = [1, 0]
    # acc = correct / 2 / len(test_loader.dataset)
    print('Test set:  Accuracy: {:.2%}'.format(
        acc
    ))


def vis(day_or_night,distance,car_type,thres=0.5):
    model = swap_net_v2(scale=1, n=[3, 8, 3])
    # changename()
    state_dict = torch.load('./checkpoint/'+str(distance)+'_'+day_or_night+'_model_best.pth',
                            map_location='cpu')
    print("Loading checkpoint",'./checkpoint/'+str(distance)+'_'+day_or_night+'_model_best.pth')
    # state_dict = torch.load('./model_best_night.pth',
    #                         map_location='cpu')
    # all_data = os.listdir("C:\\Users\\DELL\\Desktop\\gt")
    # all_data.sort()
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    result = {}
    data = Tail("D:\\light_state_recognition_data_20190909\\tail", day_or_night=day_or_night,distance=50, car_type=car_type, train=False)
    data_loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False)
    for i, sample in enumerate(data_loader):
        # for img_path in all_data:
        # input=cv2.imread("C:\\Users\\DELL\\Desktop\\gt"+'\\'+img_path)
        # input = cv2.resize(input, (112, 112), interpolation=cv2.INTER_CUBIC)
        # show = input.copy()
        # input = torch.from_numpy(input.transpose((2, 0, 1)).astype(np.float32))
        # input=torch.unsqueeze(input,0)
        # input = input.float().div(255).unsqueeze(0)
        # region = input.unsqueeze(0)

        show = sample["origin"].numpy().astype(np.uint8).copy()
        rect = sample['region']
        for i in range(8):
            try:
                show_new = show[i]
                output = model(sample["img"][i].unsqueeze(0).to('cpu'))

                # output = model(input)
                output = torch.sigmoid(output)

                # output = model(input)
                # output = nn.functional.sigmoid(output)
                output_np = output.detach().cpu().numpy()
                pos_ind = np.where(output_np >= thres)
                output_np[pos_ind] = 1
                # pos_ind = np.where((output_np[:, 1] >= 0.9) & (output_np[:, 0] < 0.1))
                # neg_ind = np.where((output_np[:, 1] < 0.5) & (output_np[:, 0] >= 0.5))
                # output_np[pos_ind] = [0, 1]
                # output_np[neg_ind] = [1, 0]
                # print(output_np[0])
                if (output_np == np.array([1])).all():
                    label = '2'
                else:
                    label = '1'
                print(label, sample['gt'][i])
                # result[img_path_list]=label
                rect_lt=(int(rect[0][i]),int(rect[1][i]))
                label_pos=(int(int(rect[0][i]) + int(rect[2][i])-60),int(rect[1][i]))
                rexy_rb=(int(int(rect[0][i]) + int(rect[2][i])), int(int(rect[1][i]) + int(rect[3][i])))
                cv2.rectangle(show_new, rect_lt,
                              rexy_rb, (0, 255, 0), 2)
                cv2.putText(img=show_new, text=label, org=rect_lt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                            color=(0, 0, 255), thickness=2)
                cv2.putText(img=show_new, text=sample['gt'][i], org=label_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                            color=(0, 255,0), thickness=2)
                cv2.imshow('test', show_new)
                # cv2.imshow('test', show)
                cv2.waitKey()
            except IndexError:
                break

if __name__ == '__main__':
    vis(day_or_night='night',distance=50,car_type='')
    # eval_training(day_or_night='night',distance=50,car_type='')
