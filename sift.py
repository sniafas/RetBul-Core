#!/usr/bin/env python

'''
SIFT features detection & extraction + BruteForce Matching + RANSAC homography 

A simple example of feature extraction techniques using SIFT features,
					image matching using BruteForce Matching combined with Lowe filter
					and RANSAC homography.
Experiments were applied on Vironas database exclusively.

Source code is inspired from OpenCV libraries.
==================

Usage:
------
    sift.py <img1> <img2>  --nF <nFeatures value> --nL <nOctaveLayers value> --cT <contrastThres value> --eT <edgeThresshold value> --sG <Sigma> --u --e --kpfixed --v --o"

    nF: Number of Features to retain
    nL: Number of Octave Layers
    cT: Contrast Threshold weak feature filter factor
    eT: Edge Threshold edge-like feature factor
    sG: Sigma of the Gaussian
    kpfixed: Fixed feature keypoint colour
    v: images to files
    o: data to files
'''

import datetime,time
import os,sys,getopt
import numpy as np
import cv2
import subprocess
import math
from json_tricks.np import dump, dumps, load, loads, strip_comments

queryImg = ''
nFeatures = 0
nOctaveLayers = 3
contrastThres = 0.08
edgeThres = 10
sigma = 1.6
verbose = 0
kpfixed = 0
output = 0

def writeLogsQuery(timestampFolder,d1,kp1):

	try:
		desc1_log = open(timestampFolder + '/descriptors1.txt', 'w')
		desc1_len_log = open(timestampFolder + '/descriptors1_len.txt', 'w')

		kp1_len_log = open(timestampFolder + '/kp1_len.txt', 'w')
		kp1_angle_log = open(timestampFolder + '/kp1_angle.txt', 'w')
		kp1_pt_log = open(timestampFolder + '/kp1_pt.txt', 'w')
		kp1_octave_log = open(timestampFolder + '/kp1_octave.txt', 'w')
		kp1_size_log = open(timestampFolder + '/kp1_size.txt', 'w')

	except (RuntimeError, TypeError, NameError):
		print "Cannot create log files"


	desc1_len_log.write('%s' % len(d1))
	kp1_len_log.write('%s' % len(kp1) )


	for d1log in d1:
		desc1_log.write("%s\n" % d1log)


	for kp1log in kp1:
		kp1_angle_log.write("%s\n" % kp1log.angle)
		kp1_pt_log.write("%s %s\n" % kp1log.pt)
		kp1_octave_log.write("%s\n" % kp1log.octave)
		kp1_size_log.write("%s\n" % kp1log.size)

	desc1_log.close()
	desc1_len_log.close()		

	kp1_len_log.close()
	kp1_angle_log.close()
	kp1_pt_log.close()
	kp1_octave_log.close()
	kp1_size_log.close()

def writeLogsTrain(save_path,d2,kp2):

	try:	
		desc2_log = open(save_path + '/descriptors2.txt', 'w')
		desc2_len_log = open(save_path + '/descriptors2_len.txt', 'w')

		kp2_len_log = open(save_path + '/kp2_len.txt', 'w')
		kp2_angle_log = open(save_path + '/kp2_angle.txt', 'w')
		kp2_pt_log = open(save_path + '/kp2_pt.txt', 'w')
		kp2_octave_log = open(save_path + '/kp2_octave.txt', 'w')
		kp2_size_log = open(save_path + '/kp2_size.txt', 'w')

	except (RuntimeError, TypeError, NameError):
		print "Cannot create log files"		

	desc2_len_log.write('%s' % len(d2))
	kp2_len_log.write('%s' % len(kp2))

	for d2log in d2:
		desc2_log.write("%s\n" % d2log)

	for kp2log in kp2:
		kp2_angle_log.write("%s\n" % kp2log.angle)
		kp2_pt_log.write("%s %s\n" % kp2log.pt)
		kp2_octave_log.write("%s\n" % kp2log.octave)
		kp2_size_log.write("%s\n" % kp2log.size)				

	desc2_log.close()
	desc2_len_log.close()

	kp2_angle_log.close()
	kp2_pt_log.close()
	kp2_octave_log.close()
	kp2_size_log.close()		
	kp2_len_log.close()	

def hsLog(save_path,Homography,status):

	H = open(save_path + '/h.txt','w')
	st = open(save_path + '/status.txt','w')

	H.write("%s\n" % Homography)
	st.writelines("%s\n" % str(status).split('\n') )

	st.close()
	H.close()

def filter_rawMatches(kp1, kp2, matches, ratio = 0.75):

	mkp1, mkp2 = [], []
	
	for r in range(len(matches)-1):
		#print matches[r].distance
		#print matches[r+1].distance

		if matches[r].distance < ratio * matches[r+1].distance:
			m = matches[r]
			mkp1.append(kp1[m.queryIdx])
			mkp2.append(kp2[m.trainIdx])
	
	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])
	kp_pairs = zip(mkp1, mkp2)	

	return p1,p2,kp_pairs	

def drawKeypoint(img, p):
	
	for i in range(0,len(p)):
		
		x = int(round(p[i].pt[0]))
		y = int(round(p[i].pt[1]))
		center = (x,y)
			
		radius = round(p[i].size/2) # KeyPoint::size is a diameter
		
		
		#draw the circles around keypoints with the keypoints size
		cv2.circle( img, center, int(radius), (0,0,100), 1)
		
		#draw orientation of the keypoint, if it is applicable		
		if p[i].angle != -1 :
			
			srcAngleRad = p[i].angle * 3.14159/180;
			orient1 = int(round(math.cos(srcAngleRad)*radius ))
			orient2 = int(round(math.sin(srcAngleRad)*radius ))
			cv2.line( img, center, (x+orient1,y+orient2), (0,150,0), 1);

	return img
		
def mkExpDir():

	## Prepare Experiment Folders ##
	## create new timestamp folder or use the existing for the current day experiments
	ts = time.time()

	listExpsTimeStmps = (subprocess.check_output(["ls ../exps/", "-x"], shell=True))
	dateTimeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') 	#current date

	if dateTimeStamp in listExpsTimeStmps: 					# if there are previous xperiments for the current date, populate
		
		try:
			curTimestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

			timestampFolder = "../exps/" + dateTimeStamp + "/exp_" + curTimestamp
			os.mkdir(timestampFolder , 0775)
		except (RuntimeError, TypeError, NameError):
			print "Can't create experiment folder"

	else:																		# create current date timestamp if not exist
		curTimestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

		os.mkdir("../exps/" + dateTimeStamp , 0775)
		timestampFolder = "../exps/" + dateTimeStamp + "/exp_" + curTimestamp
		os.mkdir(timestampFolder , 0775)
	
	return timestampFolder					

if __name__ == '__main__':

	#image counter
	n = 0 

	# read shell arguments 
	try:
		
		opts, args = getopt.getopt(sys.argv[3:], '', ['h=' , 'nO=' , 'nL=' , 'sG=' , 'e' , 'u', 'kpfixed', 'v', 'o' ])
	except getopt.GetoptError as e:
		print (str(e))
		print "Usage: <img1> <img2>  --nF <nFeatures value> --nL <nOctaveLayers value> --cT <contrastThres value> --eT <edgeThresshold value> --sG <Sigma> --u --e --kpfixed --v --o"
		sys.exit(2)
	

	for o, a in opts:	
		
		if o == '--nF':				# number of features
			nFeatures = a
		elif o == '--nL':			# number of octave Layers
			nOctaveLayers = a
		elif o == '--cT':			# contrast Threshold
			contrastThres = a			
		elif o == '--eT':			# edge Threshold
			edgeThres = a
		elif o == '--sG':			# sigma of Gaussian
			sigma = a			
		elif o == '--kpfixed':		# fixed keypoint colour on/off
			kpfixed = 1
		elif o == '--v':			# save images to files on/off
			verbose = 1
		elif o == '--o':			# output data to files on/off
			output = 1											
		else:
			print "Usage: <img1> <img2>  --nF <nFeatures value> --nL <nOctaveLayers value> --cT <contrastThres value> --eT <edgeThresshold value> --u --e --kpfixed --v --o"

	# Experiment Folder Routine
	timestampFolder = mkExpDir()		

	## #----------------- # ##
	## Read, Resize, Convert Query Image ##

	img1 = cv2.imread(sys.argv[1], 1)
	img1Res = cv2.resize(img1, (480, 640))
	gray1 = cv2.cvtColor(img1Res, cv2.COLOR_RGB2GRAY)

	print "\nFeatures: %d, OctaveLayers: %d, ContrastThres: %f, EdgeThres: %f, Sigma: %f" % (nFeatures,contrastThres,nOctaveLayers,edgeThres,sigma)

	## SIFT features and descriptor
	sift = cv2.xfeatures2d.SIFT_create(int(nFeatures),int(nOctaveLayers),float(contrastThres),int(edgeThres),float(sigma))
	
	kp1, d1 = sift.detectAndCompute(gray1, None)

	## # open, write, close logging files from query Image # ##
	if output:
		writeLogsQuery(timestampFolder,d1,kp1)

	if verbose:
		
		if kpfixed:
			img1Res1 = drawKeypoint(img1Res,kp1)
		else:
			cv2.drawKeypoints(img1Res,kp1,img1Res,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		cv2.imwrite(timestampFolder + '/sift_keypoints1.jpg', img1Res)

	## #----------------- # ##
	## Read, Resize, Convert Train Image ##	

	img2 = cv2.imread(sys.argv[2],1)
	img2Res = cv2.resize(img2,(480,640))
	gray2 = cv2.cvtColor(img2Res,cv2.COLOR_RGB2GRAY )	

	print "\n================"
	print "\nProcessing.. \n"

	## -Compute SURF Descriptors- ##
	computeTime = time.time()

	kp2, d2 = sift.detectAndCompute(gray2,None)

	print "Detect and Compute Train Descriptors : ",np.float16(time.time() - computeTime), " seconds \n"

	print '#Descriptors in image1: %d, image2: %d' % (len(d1), len(d2))
	print '#Keypoints in image1: %d, image2: %d \n' % (len(kp1), len(kp2))

	## # open, write, close logging files from trainImage # ##
	if output:
		writeLogsTrain(timestampFolder,d2,kp2)

	if verbose:
		
		if kpfixed:
			img2Res2 = drawKeypoint(img2Res,kp2)
		else:
			cv2.drawKeypoints(img2Res,kp2,img2Res,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		cv2.imwrite(timestampFolder + '/sift_keypoints2.jpg', img2Res)

	## # ----------------# ##
	## # Matching & Homography # ##


	## # Use simple matching for all matches # ##
	bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
	raw_matches = bf.match(d1,d2)
	p1, p2, kp_pairs = filter_rawMatches(kp1,kp2,raw_matches)	

	if verbose:				
		img_match = cv2.drawMatches(img1Res,kp1,img2Res,kp2,raw_matches,None, flags=2)
		cv2.imwrite(timestampFolder + '/bf_match.jpg', img_match)
	
	print 'Matching tentative points in image1: %d, image2: %d' % (len(p1), len(p2))
	
	## # ----------------# ##
	## # Homography # ##

	print 'Homography\n'

	if len(kp_pairs) > 4:
		
		Homography, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
		inliers = np.count_nonzero(status)
		percent = float(inliers) / len(kp_pairs)
	
		print "# Inliers %d out of %d tentative pairs" % (inliers,len(kp_pairs))
		print '{percent:.2%}'.format(percent= percent )

		
		## # open,save,close Homography results
		if output:
			hsLog(timestampFolder,Homography, status)	

		img1Res = cv2.resize(img1,(480,640))
		img2Res = cv2.resize(img2,(480,640))		
		try:

			h1, w1, z1 = img1Res.shape[:3]
			h2, w2, z2 = img2Res.shape[:3]
			img3 = np.zeros((max(h1, h2), w1+w2,z1), np.uint8)
			img3[:h1, :w1, :z1] = img1Res
			img3[:h2, w1:w1+w2, :z2] = img2Res

			p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
			p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

			# plot the matches
			color = (0,250,0)

			for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
				if inlier:
					cv2.circle(img3, (x1, y1), 2, color, 5)
					cv2.circle(img3, (x2, y2), 2, color, 5)
					cv2.line(img3, (x1, y1), (x2, y2), (255,100,0),2)
				else:
					col = (0, 0, 255)
					r = 2
					thickness = 3
					cv2.line(img3, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
					cv2.line(img3, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
					cv2.line(img3, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
					cv2.line(img3, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

			if verbose:
				cv2.imwrite(timestampFolder + '/sift_match.jpg', img3)

		except (RuntimeError, TypeError, NameError):
			print "Not enough Inliers"

	else:
		print "Not enough tentative correspondenses"
	
	mdata = {	'Query': queryImg, 'Descriptor' : 'sift' , 'nFeatures': nFeatures , 
				'nOctaveLayers' : nOctaveLayers , 'contrastThres' : contrastThres,
				'edgeThres' : edgeThres
			}

	with open(timestampFolder + '/mdata.json','w') as mdataFile:
		dump( {'Metadata': mdata },mdataFile )