# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher
import matplotlib.pyplot as plt
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required=True,
                help="Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required=True,
                help="Path to the query image")
ap.add_argument("-r", "--result-path", required=True,
                help="Path to the result path")
args = vars(ap.parse_args())

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query)

# perform the search
searcher = Searcher(args["index"])
results = searcher.search(features)

# display the query
plt.imshow(plt.imread(args["query"]))

fig = plt.figure(figsize=(200, 100))
columns = 5
rows = int(len(results) / 5) + 1

for i in range(len(results)):
    img = plt.imread(args["result_path"] + "/" + results[i][1])
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(img)
    plt.axis("off")

plt.show()
