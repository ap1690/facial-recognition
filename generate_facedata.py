import cv2
import time
import csv
import os
from facerec_demo import names

print 'The list of available names and ids is:'
print 'Name : id'

for id in range(len(names)):
    print names[id] + ' : ' + str(id)


subject_id = int(raw_input('Please enter your id: '))

subject_name = names[subject_id]

directory = './faces/' + subject_name

if not os.path.exists(directory):
    os.makedirs(directory)
else:
    print 'There may already be data for {} which will be overwritten if you continue.'.format(subject_name)
    action = raw_input('Do you want to continue? [y]/[n]')
    if action != "y":
        print 'Execution terminated'
        exit()
    else:
        print 'Continuing'


def generate():
    face_cascade = cv2.CascadeClassifier(
        './haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    time.sleep(5)

    while(count < 500):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y + h, x:x + w], (30, 30))
            cv2.imwrite(directory + '/%s.pgm' % str(count), f)
            # print count
            count += 1
        cv2.imshow('camera', frame)

        if(cv2.waitKey(1) == ord("q")):
            break
    camera.release()
    cv2.destroyAllWindows()
    return count


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

if __name__ == "__main__":
    num_images = generate()
    makecsvfile(num_images)
