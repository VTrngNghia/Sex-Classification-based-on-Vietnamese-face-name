import csv
import os.path
import sys

import requests

this_file, dataset = sys.argv

with open(dataset + '.csv') as csvfile:
    reader = list(csv.reader(csvfile))
    for index, row in enumerate(reader[1:]):
        [sex, img_url] = row
        img_filename = img_url.split('/')[-1]
        img = requests.get(img_url).content
        print(str(index) + '.', img_url, '\n')
        with open(os.path.join(dataset, sex, img_filename), 'wb') as downloader:
            downloader.write(img)