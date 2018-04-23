#!/usr/bin/env python

import os,sys
import subprocess
import json
import numpy as np
from json_tricks.np import dump, dumps, load, loads, strip_comments
from pprint import pprint 


if __name__ == '__main__':

	## #----------------- # ##
	buildings = []	
	bestQuery = ''
	expBuildings = (subprocess.check_output(["ls surf_experiments/", "-x"], shell=True))
	exp_path = "surf_experiments/"

	### Append available experimental buildings ###
	for j in expBuildings.splitlines():
		buildings.append(j)

	for inlier_thres in xrange(5,16):
		# Initialise variables
		validQueries = 0		
		Precision = 0
		Recall = 0
		F = 0		
		totalP = 0 
		totalR = 0
		totalF = 0
		meanPrecision = 0
		meanRecall = 0
		meanF = 0

		for query in buildings:
			# for every individual experiment
			print "Opening %s" % query
			with open(exp_path + query + '/results.json') as s:
				results = json.load(s)

			N_results = 0
			C_results = 0
			for resultValid in results['Results']['__ndarray__']:
				# Matches >= the inlier threshold are counted
				if resultValid[0][2] >= inlier_thres:
					N_results +=1

					if resultValid[0][1][8:10] == query[8:10]:
						# Query Build Class == current building
						C_results +=1
			print N_results
			print C_results

			if N_results > 0 and C_results > 0 :

				Precision = float(C_results)/N_results
				totalP = totalP + Precision

				Recall = float(C_results) / 14
				totalR = totalR + Recall
			
				F = float((2 * Precision * Recall)) / (Precision + Recall)
				totalF = totalF + F

				print "Total Precision %.3f" % totalP
				print "Total Recall %.3f" % totalR
				print "Total F %.3f \n" % totalF		
				validQueries += 1
			else: 
				pass
		print validQueries

		meanPrecision = totalP / validQueries
		meanRecall = totalR / validQueries
		meanF = totalF / validQueries

		print "Mean Precision %.3f" % meanPrecision
		print "Mean Recall %.3f" % meanRecall
		print "Mean F %.3f" % meanF
		log = open("surf_results/prf/surf_results_" + str(inlier_thres),'w')
		log.write("%.3f,%.3f,%.3f,%d" % (meanPrecision,meanRecall,meanF,validQueries))
		log.close()