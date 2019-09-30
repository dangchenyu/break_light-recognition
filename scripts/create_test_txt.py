import os


def main(label_path):
    num = 0
    with open(label_path) as f:
        while True:
            lines = f.readline()
            if lines:
                line_list=lines.split(',')
                if line_list[5]=='2':
                    o = open("D:\\light_state_recognition_data_20190909\\tail\\30_train.txt", 'a+')
                    o.write(lines)


                print("writing:", num)
            else:
                o.close()
                break


if __name__ == '__main__':
    main("D:\\light_state_recognition_data_20190909\\tail\\30_train.txt")
