#!/usr/bin/env python
import os,sys
import subprocess
import json
import numpy as np
from json_tricks.np import dump, dumps, load, loads, strip_comments
from scipy.interpolate import interp1d

if __name__ == '__main__':

	## #-----------------# ##
	Precision = 0
	c = 15
	totalP = 0
	totalR = 0
	totalF = 0
	expCounter = 0
	Precision = np.zeros(14)
	Recall = np.zeros(14)
	F = 0
	totalP = np.zeros(14)
	totalR = 0
	totalF = 0
	meanPrecision = 0
	meanRecall = 0
	meanF = 0
	totalPr11 = np.zeros(11)
	totalRe = np.zeros(11) 
	buildings = []
	x = np.linspace(0, 1, num=11, endpoint=True)	
	expBuildings = (subprocess.check_output(["ls sift_experiments/", "-x"], shell=True))
	exp_path = "sift_experiments/"
	## #-----------------# ##

	### Append available experimental buildings ###
	for j in expBuildings.splitlines():
		buildings.append(j)

	for query in buildings:
		with open(exp_path + query + '/classRank.json') as cR:
			classResults = json.load(cR)

		C_results = 1
		for classValid in classResults['ClassRank']['__ndarray__']:
			if C_results <=14:
				Precision[C_results-1] = float(C_results) / int(classValid[0][0])
				Recall[C_results-1] = float(C_results) / 14
				C_results +=1
			else:
				pass
		Pr = interp1d(x, Precision)
		Re = interp1d(x, Recall)
		totalPr11 = Pr(x) + totalPr11
	meanPr11 = totalPr11/len(buildings)
	meanRe = totalRe/len(buildings)
	np.savetxt('sift_pr11_a', meanPr11, delimiter=',', fmt='%.4e')