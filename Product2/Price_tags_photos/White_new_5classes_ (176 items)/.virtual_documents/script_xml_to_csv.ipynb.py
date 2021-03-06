import os
import shutil
import zipfile
import urllib
import xml.etree.ElementTree as ET
import numpy as np
import csv
import pandas


# all files: jpeg and xml are in the same folder
os.listdir()


# Define the necessary variables
DATASET_DIR = 'test'
ANNOTATIONS_FILE = 'annotations_test.csv'
CLASSES_FILE = 'classes_test.csv'


# Just run this script
annotations = []
classes = set([])

for xml_file in [f for f in os.listdir(DATASET_DIR) if f.endswith(".xml")]:
    tree = ET.parse(os.path.join(DATASET_DIR, xml_file))
    root = tree.getroot()

    file_name = None

    for elem in root:
        if elem.tag == 'filename':
            file_name = os.path.join(DATASET_DIR, elem.text)

        if elem.tag == 'object':
            obj_name = None
            coords = []
            for subelem in elem:
                if subelem.tag == 'name':
                    obj_name = subelem.text
                if subelem.tag == 'bndbox':
                    for subsubelem in subelem:
                        coords.append(subsubelem.text)
            item = [file_name] + coords + [obj_name]
            annotations.append(item)
            classes.add(obj_name)

with open(ANNOTATIONS_FILE, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(annotations)

with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(classes):
        f.write('{},{}\n'.format(line,i))












