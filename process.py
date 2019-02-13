import cv2
import numpy as np


def makecsvfile(num_images):
    filename = './' + subject_name + '.csv'
    myfile = open(filename, 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    count = 0
    while(count < num_images):
        string = 'faces/' + subject_name + '/' + \
            str(count) + '.pgm;' + str(subject_id)
        wr.writerow(string)
        count = count + 1
    myfile.close()


def getdata(sizex=30, sizey=30):
    basename = '/Users/Jaiyam/Python_problems/OpenCV practice/faces/'
    datafor = ['Ambai', 'Hikaru', 'Jaiyam', 'Miyashita', 'Taisuke', 'Yukino']
    all_ims = []
    all_labels = []

    for name in datafor:
        filename = basename + name + '.csv'
        myfile = open(filename, 'rb')
        # read lines of file
        print 'Getting data for ', name
        for line in myfile:
            line = line.replace('\"', '')

            path, label = line.split(';')
            path = path.replace(',', '')
            path = path.replace('\\', '')
            # print label
            label = int(label.replace(',', ''))
            image = cv2.imread('./' + path, 0)  # read as grayscale
            # newim=cv2.resize(image,(sizex,sizey))
            all_ims.append(np.reshape(image, sizex * sizey))
            all_labels.append(label)

        myfile.close()

    all_ims = np.array(all_ims)
    all_labels = np.array(all_labels)
    print all_ims.shape
    print all_labels.shape
    return [all_ims, all_labels]

if __name__ == "__main__":
    getdata()
