#!/usr/bin/env python

import subprocess
import json
from json_tricks.np import load

if __name__ == '__main__':

	## #----------------- # ##
	buildings = []	
	bestQuery = ''
	expBuildings = (subprocess.check_output(["ls surf_experiments/", "-x"], shell=True))
	exp_path = "surf_experiments/"
	# Inlier Threshold acquired from the F1 score peak given from the PRF plot
	inlier_thres = 10
	### Append available experimental buildings ###
	for j in expBuildings.splitlines():
		buildings.append(j)

	#experimental building ids
	ids = ['03','13','15','22','39','60']

	# Parsing for every building
	for buildingIdx in ids:
		# Initialise variables
		expCounter = 0			
		Precision = 0
		Recall = 0
		F = 0
		
		totalP = 0 
		totalR = 0
		totalF = 0
		bestP = 0
		bestR = 0
		bestF = 0			

		meanPrecision = 0
		meanRecall = 0
		meanF = 0

		for query in buildings:

			print "Opening %s" % query
			with open(exp_path + query + '/results.json') as s:
				results = json.load(s)
			with open(exp_path + query + '/classRank.json') as cR:
				classResults = json.load(cR)

			# Query Build Class == current building
			if query[8:10] == buildingIdx:
				N_results = 0
				for resultValid in results['Results']['__ndarray__']:
					# Matches >= the inlier threshold are counted
					if resultValid[0][2] >= inlier_thres:
						N_results +=1

				if N_results > 0:
					C_results = 0
					# Retrieved Images from the same class query building					
					for classValid in classResults['ClassRank']['__ndarray__']:
						if classValid[0][2] >= inlier_thres:
							C_results +=1
			
					if N_results > 0 and C_results > 0 :
						#Precision, Recall, Fscore metrics
						Precision = float(C_results) / N_results
						totalP = totalP + Precision
						print "\tExperiment Precision %.3f" % Precision

						# 15 images/building, query image is omitted
						Recall = float(C_results) / 14
						totalR = totalR + Recall
						print "\tExperiment Recall %.3f" % Recall
					
						F = float((2 * Precision * Recall)) / (Precision + Recall)
						print "\tExperiment F Measure %.3f" % F
						totalF = totalF + F

						print "Total Precision %.3f" % totalP
						print "Total Recall %.3f" % totalR
						print "Total F %.3f \n" % totalF		
						expCounter += 1

						if F > bestF:
							bestF = F
							bestR = Recall
							bestP = Precision
							bestQuery = query
					else: 
						pass
				print "Best F %.3f" % bestF
				print "Best Query %s" % bestQuery

		log = open("surf_results/fmeasure/surf_fmeasure_" + str(buildingIdx),'w')
		log.write("%.3f,%.3f,%.3f,%s" % (bestP,bestR,bestF,str(buildingIdx)))
		log.close()

