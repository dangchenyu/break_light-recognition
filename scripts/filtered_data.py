import os
import numpy as np


def cal_pixel(distance, car_type, ratio=0.9):
    if car_type == 'bus':
        car_width = 250 * 14.2 / distance * ratio
    if car_type == 'sedan':
        car_width = 165 * 14.2 / distance * ratio
    if car_type == 'truck':
        car_width = 190 * 14.2 / distance * ratio
    if car_type == 'special':
        car_width = 280 * 14.2 / distance * ratio
    return car_width


def main(label_path, distance):
    list = os.listdir(label_path)
    list.sort()

    for index, i in enumerate(list):

        new_path = os.path.join(label_path, i)

        if os.path.isdir(new_path):
            main(new_path, distance)

        if os.path.isfile(new_path):
            if 'txt' in new_path:
                if 'bus' in new_path:
                    car_type = 'bus'
                elif 'sedan' in new_path:
                    car_type = 'sedan'
                elif 'truck' in new_path:
                    car_type = 'truck'
                elif 'special' in new_path:
                    car_type = 'special'
                else:
                    continue
                f = open(new_path)
                with open("D:\\light_state_recognition_data_20190909\\tail" + "\\" + "over50_night_test.txt",
                          'a+') as o:
                    while True:
                        line = f.readline()
                        if line:
                            changed_str = line.replace('/home/yzxia/projects/light_state_recognition/data',
                                                       'D:\\light_state_recognition_data_20190909')
                            changed_str = changed_str.replace('/', '\\')
                            line_list = changed_str.split()
                            thres_car_width = cal_pixel(distance, car_type)
                            if int(line_list[3]) < thres_car_width:
                                str_line = ','.join(line_list)
                                print("writing:", str_line)
                                o.write(str_line + '\n')
                        else:
                            o.close()
                            break


if __name__ == '__main__':
    main("D:\\light_state_recognition_data_20190909\\tail\\night\\labels_after_fusion_night\\test_labels\\", 50)
