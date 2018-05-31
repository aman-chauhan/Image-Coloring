from skimage import color
import numpy as np
import sys
import os


def main(dest,source):
	g=None
	for root,subdirs,files in os.walk(source):
		g=[for x in subdirs if x!='outliers']
		break
	labels=[]
	for gd in g:
		for root,subdirs,files in os.walk(os.path.join(source,gd)):
			



if __name__=='__main.py__':
	main(sys.argv[1],sys.argv[2],sys.argv[3])