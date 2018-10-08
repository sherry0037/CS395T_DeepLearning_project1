import glob
from random import shuffle
import os
import argparse
import numpy as np

import re
yb_r = re.compile("(\d\d\d\d)_(.*)_(.*)_(.*)_(.*)")
sv_r = re.compile("([+-]?\d*\.\d*)_([+-]?\d*\.\d*)_\d*_-004")

# Get the label for a file
# For yearbook this returns a year
# For streetview this returns a (longitude, latitude) pair
def label(filename):
  m = yb_r.search(filename)
  if m is not None: return int(m.group(1))
  m = sv_r.search(filename)
  assert m is not None, "Filename '%s' malformatted"%filename
  return float(m.group(2)), float(m.group(1))

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--out_file", dest="out_file",
	                    help="Output file name", metavar="FILE", required=False)

	args = parser.parse_args()
	image_names =[]

	for img in glob.glob('F/*.png'):
		image_names.append(img)

	shuffle(image_names)
	count = 1
	input = open(args.out_file, 'w')
	for image in image_names:
		new_image_name = 'F/' + '%06d'%count+'.png'
		print new_image_name
	   	#os.rename(image, new_image_name)
		count += 1
		#input.write(new_image_name + '\t' + str(label(image)) + '\n')
		input.write(new_image_name + '\n')
	input.close()

