import os


def main(path, distance, *car_type):
    list_label = os.listdir(path)
    for label in list_label:
        for i in car_type:
            f = open(path+label)
            break_pos = 0
            break_neg = 0
            turn_neg = 0
            turn_lift = 0
            turn_right = 0
            flash = 0
            lines = 0
            while True:
                line = f.readline()
                line = line.rstrip()

                if line:

                    if i in line:
                        lines += 1
                        line_list = line.split(',')
                        turn_light = int(line_list[6])
                        break_light = int(line_list[5])
                        if break_light == 2:
                            break_pos += 1
                        if break_light == 1:
                            break_neg += 1
                        if turn_light == 1:
                            turn_neg += 1
                        if turn_light == 2:
                            turn_lift += 1
                        if turn_light == 3:
                            turn_right += 1
                        if turn_light == 4:
                            flash += 1
                        if 'day' in line:
                            day_or_night = 'day'
                        if 'night' in line:
                            day_or_night = 'night'
                else:
                    break
            with open("D:\\light_state_recognition_data_20190909\\" + os.path.splitext(os.path.basename(label))[
                0] + '_num.txt', 'a+') as f:
                f.write("day_or_night:" + day_or_night + '    ' + "car_type:" + i + '    ' + 'total num:' + str(
                    lines) + '\n' + 'break_neg:' + str(
                    break_neg) + '    ' + str(
                    break_neg / lines * 100) + '\n' + 'break_pos:' + str(
                    break_pos) + '    ' + str(
                    break_pos / lines * 100) + '\n' + 'turn_lift:' + str(turn_lift) + '    ' + str(
                    turn_lift / lines * 100) + '\n' + 'turn_right:' + str(turn_right) + '    ' + str(
                    turn_right / lines * 100) + '\n' + 'flash:' + str(
                    flash) + '    ' + str(
                    flash / lines * 100) + '\n' + 'turn_neg:' + str(turn_neg) + '    ' + str(
                    turn_neg / lines * 100) + '\n')


if __name__ == '__main__':
    main("C:\\Users\\DELL\\Desktop\\labels\\", 30,  'bus', 'truck', 'sedan', 'special', '')

    print("file generated")
