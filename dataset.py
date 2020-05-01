import csv
import glob
import _pickle as pickle
import copyreg
import cv2

data=[]
j=0
count=0
csvData=[]
csvData.append(['country','denomination','currency','side'])
for x in glob.glob('files/*.jpg'):
	y = x.strip('.jpg')
	z = y.strip('files/')
	data= [z.split('_')[0], z.split('_')[1], z.split('_')[2], z.split('_')[3]]
	csvData.append(data)
	
with open('dataset.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)


training_set=[img for img in glob.glob('files/*.jpg')]
sift = cv2.xfeatures2d.SIFT_create()
i=0
kpTrain = []
desTrain = []
while i<len(training_set):
	train=cv2.imread(training_set[i])
	grey_img=cv2.cvtColor(train,cv2.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(grey_img,None,useProvidedKeypoints = False)
	kpTrain.append(kp)
	desTrain.append(des)
	
	i = i + 1




