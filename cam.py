import PIL,numpy,os,cifar,time,cv2
from PIL import Image
from cifar import model
import matplotlib.pylab as plt


'''create a folder called pics in your working directory and change the path to that folder'''
path=r"/home/parismita/machine learning/techevince/pics"

capture = cv2.VideoCapture(0)
objects = ['human','object']
i = 2500
print time.asctime( time.localtime(time.time()) )
while 1:
    #time sleep delays time by 5...not used as it still take succesive video frames not 5th video frame making the vidoe slow
    #time.sleep(5)
    img = capture.read()
    test_x = img[1]
    print test_x.shape,i

    #resizing the video decreases the picture quality
    a=cv2.resize(test_x,(70,70),interpolation=cv2.INTER_CUBIC)

    #prediction
    test = numpy.reshape(a,(1,3,70,70))
    prediction = model.predict(test)
    #prediction2 = model.predict(test2)
    print "predict", prediction , prediction.shape





    if prediction[0][0]>prediction[0][1]:
        print 'human'
    else:
        print 'object'




    #index = numpy.argmax(prediction)
    '''print objects[index]
    text_file = open("Trial.txt", "a")
    text_file.write(str(index))
    text_file.write("\n")'''

    #saving images (not required) as succesive testing would be done
    #cv2.imwrite('pics/pic{:>05}.jpg'.format(i),a)
    #i=i+1
    #print numpy.shape(a)

    cv2.imshow('original',img[1])
    cv2.imshow('resized',a)
    #close loop
    if cv2.waitKey(33) == ord('a'):
        print "pressed a"
        break
#text_file.close()
print time.asctime( time.localtime(time.time()) )

