#!/usr/bin/env python

import cv2
import pandas as pd
import numpy as np
import argparse,sys
import math
import glob
import subprocess
from openpyxl import Workbook, load_workbook
from json_tricks.np import dump, load
from utilities import Utilities

def filter_rawMatches(kp1, kp2, matches, ratio = 0.75):

	mkp1, mkp2 = [], []

	for r in range(len(matches)-1):
		if matches[r].distance < ratio * matches[r+1].distance:
			m = matches[r]
			mkp1.append(kp1[m.queryIdx])
			mkp2.append(kp2[m.trainIdx])
	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])
	kp_pairs = zip(mkp1, mkp2)
	
	return p1,p2,kp_pairs

def rankingList(index,image_id,n_inliers,percent):

	resList[index][0] = index
	resList[index][1] = image_id
	resList[index][2] = n_inliers
	resList[index][3] = percent	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-dataset', action='store', dest='dataset',
	                    help='Set dataset Path ending with / ')
	parser.add_argument('-H', action='store', dest='hss', type=int,
						default=200, help='Hessian Matrix <0-5000>')
	parser.add_argument('-nO', action='store', dest='nO', type=int,
						default=3, help='Number of Octaves <1-4>')
	parser.add_argument('-nL', action='store', dest='nL', type=int,
						default=3, help='Number of Octave Layers <3-6>')
	parser.add_argument('-u', action='store_true', default=False,
						dest='upright', help='SURF Upright')
	parser.add_argument('-e', action='store_true', default=False,
						dest='e', help='64/128 bit Length')
	parser.add_argument('-kpfixed', action='store_true',default=False,
	                    dest='kpfixed',help='Fixed Keypoint Colour')
	parser.add_argument('-v', action='store_true',default=False,
	                    dest='v',help='Save image to file')
	parser.add_argument('-wdata', action='store_true',default=False,
	                    dest='wdata',help='Save data to CSV')
	parser.add_argument('-wimages', action='store_true',default=False,
	                    dest='wimages',help='Save images in query folder')
	arguments = parser.parse_args()

	if arguments.dataset:
		datasetPath = arguments.dataset
	else:
		parser.print_help()
		print("-img1: Query Image")
		sys.exit(1)
	
	util = Utilities()		
	
	# experiment directory
	expDir = "surf_experiments"
	qBuildings = ['01','02']	
	## Prepare Dataset ##
	dataset, queryList = util.createDataset(datasetPath,qBuildings)
	#creating a list of (<image>,#inliers) pairs
	resList = np.zeros( len(dataset) , [('idx', 'int16'), ('imageId', 'a28'), ('inliers', 'int16'), ('percent', 'float') ])
	
	print("\n================")
	print("Hessian", arguments.hss)
	print("Octaves", arguments.nO)
	print("Layers", arguments.nL)
	print("================")

	## SURF features and descriptor
	surf = cv2.xfeatures2d.SURF_create(arguments.hss, arguments.nO, arguments.nL, arguments.e, arguments.upright)

	for img1Path in queryList:

		## #----------------- # ##
		## Read, Resize, Grayscale Query Image ##
		n = 0
		img1 = cv2.resize(cv2.imread(datasetPath + img1Path, 1), (480, 640))
		img1Gray = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY )
		kp1, d1 = surf.detectAndCompute(img1Gray, None)

		for img2Path in dataset:
			if img1Path != img2Path:

				print("\nProcessing..")
				print("Test Image:%s (%d/%d) \n" % (img2Path,n+1,len(dataset)))
				
				## #----------------- # ##
				## Read, Resize, Grayscale Test Image ##
				img2 = cv2.resize(cv2.imread(datasetPath + img2Path, 1), (480, 640))
				img2Gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY )
				kp2, d2 = surf.detectAndCompute(img2Gray, None)

				## # Use BFMatcher, Euclidian distance, Eliminate Multiples # ##
				bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
				raw_matches = bf.match(d1,d2)
				src_points, dst_points, kp_pairs = filter_rawMatches(kp1,kp2,raw_matches)

				print('Matching tentative points in image1: %d, image2: %d' % (len(src_points), len(dst_points)))

				## # ----------------# ##
				## # Homography # ##
				print('#----------------#')
				print('Homography')
				print('#----------------#')
				if len(kp_pairs) > 4:
					Homography, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
					inliers = np.count_nonzero(status)
					percent = float(inliers) / len(kp_pairs)

					print("# Inliers %d out of %d tentative pairs" % (inliers,len(kp_pairs)))
					rankingList(n,img2Path,inliers,percent)
				else:
					rankingList(n,img2Path,0,0)
					print("Not enough correspondenses")

				n = n+1
				## Verbose OR WriteOutput
				if arguments.v or arguments.wimages:
					img1kp = img1
					img2kp = img2

					if arguments.kpfixed:
						img1kp = util.drawKeypoint(img1kp,kp1)
						img2kp = util.drawKeypoint(img2kp,kp2)
					else:
						cv2.drawKeypoints(img1kp,kp1,img1kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
						cv2.drawKeypoints(img2kp,kp2,img2kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
						print("Default Keypoints")

					try:
						h1, w1, z1 = img1.shape[:3]
						h2, w2, z2 = img2.shape[:3]
						img3 = np.zeros((max(h1, h2), w1+w2,z1), np.uint8)
						img3[:h1, :w1, :z1] = cv2.resize(cv2.imread(datasetPath + img1Path, 1), (480, 640))
						img3[:h2, w1:w1+w2, :z2] = cv2.resize(cv2.imread(datasetPath + img2Path, 1), (480, 640))
				
						p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
						p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

						for (x1, y1), (x2, y2), inlier in zip(p1,p2, status):
							if inlier:
								cv2.circle(img3, (x1, y1), 2, (0,250,0), 5)
								cv2.circle(img3, (x2, y2), 2, (0,250,0), 5)
								cv2.line(img3, (x1, y1), (x2, y2), (255,100,0),2)
							else:
								cv2.line(img3, (x1-2, y1-2), (x1+2, y1+2), (0, 0, 255), 3)
								cv2.line(img3, (x1-2, y1+2), (x1+2, y1-2), (0, 0, 255), 3)
								cv2.line(img3, (x2-2, y2-2), (x2+2, y2+2), (0, 0, 255), 3)
								cv2.line(img3, (x2-2, y2+2), (x2+2, y2-2), (0, 0, 255), 3)

					except (RuntimeError, TypeError, NameError):
						print("Not enough Inliers")

					if arguments.v:
						imgTentMatches = cv2.drawMatches(img1kp,kp1,img2kp,kp2,raw_matches,None, flags=2)
						cv2.imshow('Query',img1kp)
						cv2.imshow('Test',img2kp)
						cv2.imshow('Tentative Matches',imgTentMatches)						
						cv2.imshow('SURF Match + Inliers',img3)
						cv2.waitKey(0)
						cv2.destroyAllWindows()

					if arguments.wimages:
						## Write Matching Image ##
						util.mkFolder(expDir + "/" + img1Path[:-4] + '/' +img2Path[:-4])
						cv2.imwrite(expDir + "/" + img1Path[:-4] + '/' +img2Path[:-4] + '/sift_match.jpg', img3)						

				#Output CSV
				if arguments.wdata:
					util.mkFolder(expDir + "/" + img1Path[:-4])
					util.initWrite()
					util.writeFile(kp2,d2,img1Path,img2Path,inliers,percent,len(kp2))
					util.closeWrite(expDir + "/",img2Path,'surf')
			else:
				pass				
				rankingList(n,img2Path,0,0)
				n+=1

		print("\n#### Ranking ####")
		rList = np.sort(resList, order= 'inliers')[::-1]
		for bestPair in range(10):
			print '#%d: %s -> Inliers: %d' % (bestPair + 1, rList[bestPair][1], rList[bestPair][2])
			print '{percent:.2%}'.format(percent= rList[bestPair][3] )

		## # Results and Experimental Values Logging # ##
		if arguments.wdata:
			try:
				cmd = "sed -e '!d' surf_experiments/surf*.csv >> surf_experiments/" + img1Path[:-4] + "/data_merged.csv"
				rm = "rm -r surf_experiments/*.csv surf_experiments/*.xls"

				subprocess.check_output([cmd], shell=True)
				subprocess.check_output([rm], shell=True)
				jList = rList.reshape((n,1))
				with open(expDir + "/" + img1Path[:-4] + '/results.json','w') as resultFile:
					dump({'Results': jList },resultFile)

			except (RuntimeError, TypeError, NameError):
				print("Internal Structure Error")