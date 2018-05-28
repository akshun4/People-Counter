# People Counter

The People Counter program allows you to count the number of people in a video passing through a Region of Interest, which is created by the user. To use the program, you'll need Python 3 and OpenCV 3.4. 

It is built for security footage, and it detects people using HOG + Linear SVM classifier (Histogram of Oriented Gradients for Objection Detection), along with Non-maximum Suppression(NMS); and it then outputs distinct pictures of the detected people using SIFT (Scale-invariant feature transform). 
The program counts number of people coming in towards the camera as coming 'In' and people going away from the camera as 'Out'. 

(Some modules such as SIFT do not come with OpenCV versions 3.x (unlike opencv 2.x), so you'll have to install the  'opencv_contrib' package seperately)

# Running the program

```shell
python peoplecounter.py -vid Input/2.mp4 -roi manually
``` 

This will take the video 2.mp4 from Input folder and the distinct peoples' images will be stored in the Output_Images folder automatically. You'll have to create the Region of Interest (RoI) manually (by default). However, to minimize repeated efforts for drawing the RoI, you can create it once and test to see if the results are good (preferred to draw a tall RoI which will encompass the entire length of a person; this ensures that different parts of the person recognized do not output as different people). If you have a well created and tested RoI, the program will ask you whether you want to save the RoI for future testing. To use the pre-tested RoIs, just use the argument "pre-tested" instead of "manually" in the command line.  

# Input Examples

The Input folder has currently 3 video examples, taken from the internet. These videos are very short and only for testing. 2 of these videos - 1.mp4 and 2.mp4 have pre-tested RoI text files already created, so you can run the command line with RoI creation argument as "pre-tested", or "manually" if you prefer. 

There have been some parameters used throughout the program, such as HOG Descriptor parameters, NMS parameters, Feature Matching and SIFT parameters, which have their values as a result of repeated testing. However, these are not hard and fast parameters, as in they may not be the best parameters for all possible inputs, so the users can change them from within the program. 