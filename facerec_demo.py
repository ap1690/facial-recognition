import cv2
import os
import sys
import numpy as np

#from generate_facedata import names
names = ['Jaiyam', 'Kadri', 'Hikaru', 'Miyashita',
         'Taisuke', 'Ambai', 'Kato', 'OnoD', 'Yukino']


def read_images(path, sz=None):
    # print 'read_images called'
    c = 0
    X, y = [], []

    root_path = path + '/faces/'

    for name in names:
        subject_path = root_path + name

        if os.path.exists(subject_path):
            print 'Found data for ', name
            print '%s has data label: %g' % (name, c)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = subject_path + '/' + filename
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    if sz is not None:
                        im = cv2.resize(im, sz)

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error ({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected erro: ", sys.exc_info()[0]
                    raise
        else:
            print 'Directory does not exist for ', name
            # print subject_path
            c = c + 1
            continue
        print len(X), len(y)
        c = c + 1

    return [X, y]


def face_rec():

    # if len(sys.argv) <2:
    # 	print "USAGE: facerec_demo.py"
    # 	print sys.argv
    # 	sys.exit()
    images_path = '/Users/Jaiyam/Python_problems/OpenCV practice'

    [X, y] = read_images(images_path)

    y = np.asarray(y, dtype=np.int32)
    print y
    # if len(sys.argv) ==3:
    # 	out_dir=sys.argv[2]

    model = cv2.face.createLBPHFaceRecognizer()
    print 'Training model...'
    model.train(X, y)
    print 'Training finished'
    return model
        camera = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(
            './haarcascades/haarcascade_frontalface_default.xml')
        while(True):
            read, img = camera.read()
            faces = face_cascade.detectMultiScale(img, 1.1, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(
                    img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi = cv2.cvtColor(img[x:x + w, y:y + h], cv2.COLOR_BGR2GRAY)

                try:
                    roi = cv2.resize(
                        roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi)
                    print "Label: %s, Confidence: %.2f" % (params[0], params[1])
                    cv2.putText(
                        img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                except:
                    continue

            try:
                cv2.imshow('camera', img)
            except:
                continue

            if(cv2.waitKey(1) == ord("q")):
                break
            cv2.destroyAllWindows()
            camera.release()


if __name__ == "__main__":
    face_rec()
