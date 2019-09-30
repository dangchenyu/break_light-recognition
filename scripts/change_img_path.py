import os


def main(label_path):
    num = 0
    with open(label_path) as f:
        while True:
            lines = f.readline()
            if lines:
                num += 1
                new = lines.replace('D:\\light_state_recognition_data_20190909',
                                    '/data/yzxia/projects/light_state_recognition/data')
                new = new.replace('\\', '/')
                o = open('C:\\Users\\DELL\\Desktop\\labels\\50_day_sever_test.txt', 'a+')
                o.write(new)
                print("writing:", num)
            else:
                o.close()
                break


if __name__ == '__main__':
    main("D:\\light_state_recognition_data_20190909\\tail\\50_day_test.txt")
