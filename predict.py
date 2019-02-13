import cv2
import numpy as np
from pre_process import sigmoid
import time
from facerec_demo import names

sizex=30
sizey=30

def predict_single(input_images, wt_i_h,wt_h_o): #suppose 1 image is given
	#names=['Jaiyam','Hikaru']
	hiddenl_i=np.dot(input_images,wt_i_h)
	hiddenl_o=sigmoid(hiddenl_i) #produce a Nx10 matrix whose
										#[i,j] represents output for image i from neuron j
	outputl_i=np.dot(hiddenl_o,wt_h_o)
	probabilities=sigmoid(outputl_i) #produces a Nx2 output
	
	#print np.sum(probabilities)

	probabilities/=np.sum(probabilities,axis=1)[:,None]
	print probabilities

	predictions= np.argmax(probabilities) #find index with max probability
	
	# if probabilities[0][predictions]<0.7: #this assumes a single image
	# 	predictions=None
	#print predictions
	return predictions

means=np.load('means.npy')
stds=np.load('stds.npy')
wt_i_h=np.load('weights_i_h20.npy')
wt_h_o=np.load('weights_h_o20.npy')
for index in range(371,372):
	image=cv2.imread('./faces/Jaiyam/'+str(index)+'.pgm',0)
#image=cv2.imread('circle.png',0)

	if image is None:
		print 'Could not read image'

	else:
		
		newim=cv2.resize(image,(sizex,sizey))
		newim=np.reshape(newim,(1,sizex*sizey))
		normed=(newim-means)/(stds)
		index=predict_single(normed,wt_i_h,wt_h_o)
		#print 'Predicted person is', names[predict(normed,wt_i_h,wt_h_o)[0]]

face_cascade=cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

camera=cv2.VideoCapture(0)
count=0
#time.sleep(5)

while(count<200):
	ret,frame=camera.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces=face_cascade.detectMultiScale(gray,1.1,5)

	for (x,y,w,h) in faces:
		img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		if w>50 and h>50:
			f=cv2.resize(gray[y:y+h,x:x+w],(sizex,sizey))
			count+=1
			newim=np.reshape(f,(1,sizex*sizey))
			normed=(newim-means)/(stds)
			start_time=time.time()
			index=predict_single(normed,wt_i_h,wt_h_o)
			elapsed=time.time()-start_time
			if index is not None:
				person=names[index]
				cv2.putText(frame,person, (x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
			print 'Prediction time is ', elapsed
			
	cv2.imshow('camera',frame)
	
	if(cv2.waitKey(1)== ord("q")):
		break

camera.release()
cv2.destroyAllWindows()