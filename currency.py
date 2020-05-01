import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import csv
import json
from tkinter import *
import json, requests
from forex_python.converter import CurrencyRates
from PIL import Image, ImageTk
import re
import time
import _pickle as pickle
from dataset import kpTrain, desTrain

f= open("log.txt","w+")
c = CurrencyRates(force_decimal=False)
CurrenyCode_list = ["[AUD] Australian Dollar", "[BGN] Bulgerian Lev", "[BRL] Brazilian Real", "[CAD] Canadian Dollar", "[CHF] Swiss Franc", "[CNY] Chinese Yuan", "[CZK] Czech Koruna", "[DKK] Danish Krone", "[EUR] European Euro", "[GBP] British Pound", "[HKD] Hong Kong Dollar", "[HRK] Croatian Kuna", "[HUF] Hungarian Forint", "[IDR] Indonesian Rupiah", "[ILS] Israeli Shakal", "[INR] Indian Rupee", "[JPY] Japanese Yen", "[MXN] Mexican Peso", "[MYR] Malaysian Ringgit", "[NOK] Norwegian Kroner", "[NZD] New Zealand Dollar", "[PHP] Philippine Peso", "[PLN] Polish Zloty", "[RON] Romanian Lei", "[RUB] Russian Rouble", "[SEK] Swedish Krona", "[SGD] Singapore Dollar", "[KRW]South Korean Won", "[TRY] Turkish Lira", "[THB] Thai Baht", "[USD] US Dollar", "[ZAR] South African Rand"] 
createDict = {}
training_set=[img for img in glob.glob('files/*.jpg')]

class mclass:
	def __init__(self,  root, variable2, var1, var2, var6, var4, var5):
		self.variable2 = variable2
		self.var1 = var1
		self.var2 = var2
		self.var6 = var6
		self.var4 = var4
		self.var5 = var5
		self.root = root
		
		self.img = ImageTk.PhotoImage(Image.open("compare.jpg"))
		self.panel = Label(root, image = self.img)
		self.panel.place(x=30, y=80)
		self.headlabel = Label(root, text = 'Currency Recognition System', fg = 'black', bg = "yellow", font=("Courier", 28)).place(x=600, y=5)     
		self.label1 = Label(root, text = "Amount :", fg = 'black', bg = 'light blue', font=("Courier", 14)).place(x=80, y=650)   
		self.label2 = Label(root, text = "From Currency :", fg = 'black', bg = 'light blue', font=("Courier", 14)).place(x=80, y=720)     
		self.label3 = Label(root, text = "To Currency :", fg = 'black', bg = 'light pink', font=("Courier", 14)).place(x=80, y=790)  
		self.label4 = Label(root, text = "Converted Amount :", fg = 'black', bg = 'light blue', font=("Courier", 14)).place(x=80, y=950)
		self.label4 = Label(root, text = "History", fg = 'black', bg = 'light pink', font=("Courier", 14)).place(x=1550, y=90)       
		 
		self.label5 = Label( root, textvariable=var1)
		self.label5.place(x=300, y=650) 
		self.label6 = Label( root, textvariable=var2)
		self.label6.place(x=300, y=720)   
		
		self.label8 = Label( root, textvariable=var6, font=("Courier", 14))
		self.label8.place(x=1650, y=1000)
		self.label9 = Label( root, textvariable=var4, font=("Courier", 14))
		self.label9.place(x=800, y=450)
		self.label10 = Label( root, textvariable=var5, font=("Courier", 14))
		self.label10.place(x=800, y=500)   
		
		self.ToCurrency_option = OptionMenu(root, self.variable2, *CurrenyCode_list)
		self.ToCurrency_option.place(x=300, y=790)
				
		self.button1 = Button(root, text = "Convert", bg = "light green", fg = "black", font=("Courier", 16), command = self.conversion).place(x=200, y=860)        
		self.button2 = Button(root, text = "Add", bg = "light green", fg = "black", font=("Courier", 16), command = self.nextimage).place(x=850, y=650)  
		self.button3 = Button(root, text = "Exit", bg = "light green", fg = "black", font=("Courier", 16), command = self.exit).place(x=1000, y=650) 

	def nextimage(self):
		self.label5.destroy()
		self.label6.destroy()
		try:
			self.label7.destroy()
		except (NameError, AttributeError):
			pass
		self.label8.destroy()
		self.label9.destroy()
		self.label10.destroy()
		self.panel.destroy()
		self.ToCurrency_option.destroy()
		detect(self.root)
	
	def conversion(self):
		self.var3 = re.search(r"\[([A-Z]+)\]", self.variable2.get())
		print(str(self.var3.group(1)))
		print(str(self.var1.get()))
		print(float(self.var2.get()))
		self.output = c.convert(str(self.var1.get()), str(self.var3.group(1)), float(self.var2.get()))
		print(self.output)
		out = DoubleVar()
		out.set(self.output)
		self.label7 = Label( root, textvariable=out)
		self.label7.place(x=300, y=950) 
		

	def exit(self):
		exit(0)
 
		

def process(cap, sift, l):
	
	ret, test_img=cap.read()
	f.write("captured frame ret\n " + str(ret) + "\n")
	f.write("captured frame image\n " + str(test_img) + "\n")
	if cv2.waitKey(1)& 0xFF==ord('q'):
		return
	import time
	start = time.time()
	f.write("Start time\n " + str(start) + "\n")
	j=0
	kp1, des1 = sift.detectAndCompute(test_img,None)
	f.write("k of input image\n " + str(kp1) + "\n")
	f.write("descriptors of input image\n " + str(des1) + "\n")
	
	while j<len(training_set):
		kp2 = kpTrain[j]		
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		f.write("parameters of index\n " + str(index_params) + "\n")
		search_params = dict(checks=60) 
		f.write("Parameters of comparitive data\n " + str(search_params) + "\n")

		flann = cv2.FlannBasedMatcher(index_params,search_params)
		f.write("Flann based parameter matching\n " + str(flann) + "\n")

		matches = flann.knnMatch(des1,desTrain[j],k=2)
		f.write("FLann based descriptor matching\n " + str(matches) + "\n")
		matchesMask = [[0,0] for i in range(len(matches))]
		f.write("Mask for matching initialization\n " + str(matchesMask) + "\n")

		for i,(m,n) in enumerate(matches):
			f.write("i, (m,n)\n " + str(i) + "\n"+ str(m) + " " + str(n)+ "\n")
			if m.distance < 0.7*n.distance:
				matchesMask[i]=[1,0]
				f.write("Mask for matching\n " + str(matchesMask[i]) + "\n")

		l[j]=(np.sum(matchesMask))
		f.write("Sum of matching mask\n " + str(l[j]) + "\n")
		j=j+1
	end = time.time()
	f.write("End time\n " + str(end) + "\n")
	#print(end-start)
	with open('time.txt', 'w+') as outfile:  
			json.dump(end-start, outfile)
	f.write("Total time for processing\n " + str(end-start) + "\n")

	temp2=l[:]
	l.sort()
	f.write("Sorted training set array\n " + str(l) + "\n")
	if l[len(l)-1]-l[len(l)-2]>15:
		draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
		img3 = cv2.drawMatchesKnn(test_img,kp1,cv2.imread(training_set[temp2.index(l[len(l)-1])]),kp2,matches,None,**draw_params)
		
		c=0		
		with open("dataset.csv", 'r') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				createDict[c] = dict(row)
				c=c+1
		#print(createDict)

		plt.imsave('compare.jpg', img3)
		
		f.write("Dataset\n " + str(createDict) + "\n")
		#print(createDict[temp2.index(l[len(l)-1])])
		
		with open('storedData.txt', 'w+') as outfile:  
			json.dump(createDict[temp2.index(l[len(l)-1])], outfile)
		
		with open('history.txt', 'a+') as dumpFile:
			dumpFile.write(str(createDict[temp2.index(l[len(l)-1])]) + "\n")
    					
		f.write("Output\n " + str(createDict[temp2.index(l[len(l)-1])]) + "\n")
		f.write("\n\n\n\n")
		
		return  
	
	f.write("Not found \n\n\n\n")
	process(cap, sift, l)

def detect(root):
	cap=cv2.VideoCapture(0)
	l=[0]*(len(training_set)) 
	sift = cv2.xfeatures2d.SIFT_create()
	process(cap, sift, l)
	cap.release()
	cv2.destroyAllWindows()
	with open('storedData.txt') as json_file:  
		loaded_r = json.load(json_file) 
		print(str(loaded_r))
	variable2 = StringVar(root)	 
	variable2.set("Choose") 
	var1 = StringVar() 
	var2 = StringVar()
	var1.set(loaded_r["currency"]) 
	var2.set(loaded_r["denomination"])
	with open('time.txt') as json_file:  
		loaded_t = json.load(json_file) 
		print(str(loaded_t))


	var6 = StringVar()
	var4 = StringVar()
	var5 = StringVar()
	var6.set(loaded_t)
	var4.set(loaded_r["country"])
	var5.set(loaded_r["side"])
	start= mclass (root, variable2, var1, var2, var6, var4, var5)  	
	
	history = Listbox(root, height=40, width=65)
	with open("history.txt") as fp:
		line = fp.readline()
		cnt = 1
		while line:
			history.insert(END, line)
			line = fp.readline()
			cnt += 1
	history.place(x=1300, y=125)
	
root = Tk() 
root.configure(background = 'dark blue')  
root.geometry("1800x1125") 
root.title("Digital Image Processing") 
detect(root)	
root.mainloop()  

