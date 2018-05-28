import cv2
import numpy as np
import argparse
import datetime
import imutils
import math
from imutils.object_detection import non_max_suppression
from imutils import paths

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-vid','--video',required=True,default="Input/2.mp4",help="Video File Path")
parser.add_argument('-roi','--roi creation mode',required=False,default="manually",help="Create region of interest-do it 'manually'," + 
					"or use the 'pre-tested' one which gives good results")
args = vars(parser.parse_args())

video=args['video']
roi_mode=args['roi creation mode']

width = 800
height= 640
videopath,__=video.split(".")
__,videoname=videopath.split('/',-1)
counter,textIn,textOut=0,0,0
imageset=[]
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def feature_matching(img1,img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    return (len(good))

def create_roi(videopath,roi):
	x=input("Please type 'Y' or 'N' ")

	if x.lower()=='y':
		f= open("Input/"+videoname+"_pre-testedROI.txt","w+")
		f.write(str(roi))
		f.close()
	elif x.lower()=='n':
		pass 
	else:
		create_roi(videoname,roi)

camera = cv2.VideoCapture(video)
grabbed, frame = camera.read()

# resize the frame, convert it to grayscale, and blur it
frame = imutils.resize(frame, width=width,height=height)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)

if roi_mode=='manually':
	roi = cv2.selectROI(frame,showCrosshair=False)
elif roi_mode=='pre-tested':
	try:
		roi_file=open(videopath+'_pre-testedROI.txt','r')
		rois=roi_file.read()
		rois=rois[1:-1]
		roi=rois.split(", ")
		for i in range(len(roi)): 
			roi[i]=int(roi[i])
	except:
		print("The pre-tested Region of Interest file does not exist yet. Please create it manually.")
		roi = cv2.selectROI(frame,showCrosshair=False)
	
cv2.destroyWindow('ROI selector')

# loop over the frames of the video
st=0.0
cs=[0,0]
tc,tb,cts=0,0,0
while (camera.isOpened()):
    # grab the current frame and initialize the occupied/unoccupied
    # text
    grabbed, frame = camera.read()  
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
    
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=width,height=height)
    forig=frame
    frame=frame[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    ori=frame.copy()
    f=0
    (rects, weights) = hog.detectMultiScale(gray, winStride=(3, 3),padding=(8, 8), scale=1.2)    
    for (a, b, c, d) in rects:
        cv2.rectangle(ori, (a, b), (a + c, b + d), (255,0, 0), 2)
    rects = np.array([[a, b, a + c, b + d] for (a, b, c, d) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.75)
    #pick = non_max_suppression_fast(rects,overlapThresh=0.85)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        rect=cv2.rectangle(frame, (xA, yA), (xB, yB), (255,0,0), 2)
        rectf=frame[yA:yA+yB,xA:xA+xB]
        imageset2=np.asarray(imageset)
        xc=(xA + xB) /2
        yc=(yA + yB) /2
        rectangleCenterPoint = (int(xc), int(yc))
        cv2.circle(frame, rectangleCenterPoint, 1, (255,0,0), 2)
        if counter==0:
            counter+=1
            pimg='Output_Images/'+videoname+'_person'+str(counter)+'.jpg'
            img=cv2.imwrite(pimg,rectf)
            imageset.append(rectf)
            continue
        u=0
        if (cts==3 or cts==10) and (textIn+textOut)!=counter:
            if cs[counter]>0:
                textIn+=1
                f=1
            elif cs[counter]<0:
                textOut+=1
                f=-1	
            cts+=1
        if (cts==10) and (cs[counter]*f)<0:
            if cs[counter]>0:
                textIn+=1
                textOut-=1
            elif cs[counter]<0:
                textOut+=1
                textIn-=1       	    
        for t in imageset2:
            if feature_matching(rectf,t)>2:
                u=0
                break
            else:
                u+=1

        if u==0:
            if yc>tc and yB>tb:
                cs[counter]+=1
            elif yc<tc and yB<tb:
                cs[counter]-=1
            tc=yc
            tb=yB
            cts+=1
        if u>0:  
            cs.append(0)
            cts=0
            counter+=1
            pimg='Output_Images/'+videoname+'_person'+str(counter)+'.jpg'
            img=cv2.imwrite(pimg,rectf)
            imageset.append(rectf)
    	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.rectangle(forig, (roi[0], roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0, 255, 0),1)
    cv2.putText(forig, "In: {}".format(str(textIn)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(forig, "Out: {}".format(str(textOut)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(forig, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, forig.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Security Feed", forig)


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
print("No. of images created=",counter)
if roi_mode=='manually':
	print("Do you wish to save the created roi to be used as an optional pre-tested file for this video next run onwards (if it gave good results)")
	wr=create_roi(videoname,roi)
