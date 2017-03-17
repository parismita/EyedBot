import PIL,numpy,random
from PIL import Image
from scipy.misc import imread
from os import listdir
from os.path import isfile, join

a = [] #trainX
c = [] #trainY
data = ['human','object']
for i in range(2):
	mypath="/home/parismita/machine learning/techevince/dataset/"+str(data[i])
	#image_files="WP.jpg"
	image_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	#print image_files
	width = []
	height = []
	b = [] #class i
	for fr in image_files:
		#fi = image_files
		fi = "dataset/"+str(data[i])+"/"+fr
		im = imread(fi,mode = 'RGB')
		#print im.shape
		a.append(im)
		b.append(im)

		image = Image.open(fi).convert('RGB')
		w,h = image.size
		#print image.size
		#image = image.resize((int(250),int(250)),PIL.Image.ANTIALIAS)
		#image.save(fi)
		'''if w == h:
			#print image.size
			image = image.resize((120,120),PIL.Image.ANTIALIAS)
			#print image.size
			image.save("class/"+str(i)+"/"+fr)
		elif w>h:
			new_size = (w,w)
			new_im = Image.new("RGB", (new_size))
			new_im.paste(image, (0,0))
			new_im = new_im.resize((120,120),PIL.Image.ANTIALIAS)
			new_im.save("class/"+str(i)+"/"+fr)
		elif w<h:
			new_size = (h,h)
			new_im = Image.new("RGB", (new_size))
			new_im.paste(image, (0,0))
			new_im = new_im.resize((120,120),PIL.Image.ANTIALIAS)
			new_im.save("class/"+str(i)+"/"+fr)
		#print f,":",width,height'''
		
	if i==1:
		d = [i for x in range(len(b))]
	else:
		d = [i for x in range(len(a)-len(b),len(a))]
	c =numpy.concatenate((c,d))
	#print c.shape,c[0],c[len(a)-1]

from sklearn.utils import shuffle
a,c = shuffle(a,c)
trainX = numpy.asarray(a)
print trainX.shape
trainY = numpy.asarray(c)
trainX = numpy.transpose(trainX,(0,3,2,1))
testX = trainX[7200:]
testY = trainY[7200:]
trainY = trainY[:7200]
trainX = trainX[:7200]
print "trainX",trainX.shape,"trainY",trainY.shape
print "testX",testX.shape,"testY",testY.shape
print testY
