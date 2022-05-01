#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
There are currently 4 different SR models supported in the module.
They can all upscale images by a scale of 2, 3 and 4. LapSRN can even
upscale by a factor of 8. They differ in accuracy, size and speed.
- EDSR [1]. This is the best performing model. However, it is also the
            biggest model and therefor has the biggest file size and slowest inference.
            You can download it here.
- ESPCN [2]. This is a small model with fast and good inference. It can do real-time video upscaling (depending on image size). You can download it here.
- FSRCNN [3]. This is also small model with fast and accurate inference. Can also do real-time video upscaling. You can download it here.
- LapSRN [4]. This is a medium sized model that can upscale by a factor as high as 8. You can download it here.

For more information and implementations of these models see the module’s
GitHub README file. For extensive benchmarks and comparisons go here.

si usa come:
 ./enhance.py immagine

scrive nella stessa directory del file originale e produce un file .tif

"""
import cv2
from cv2 import dnn_superres
import sys
import os
from datetime import datetime

startTime = datetime.now()

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# controlla numero di parametri, se più di uno errore
if len(sys.argv) > 2:
    print('Inserire solo il file da trasformare...')
    sys.exit()

# legge il nome del file da trasformare
file_name_full_path = sys.argv[1]

# estrae il solo nome del file senza estensione
file_name_full_path_no_extension = os.path.splitext(file_name_full_path)[0]

# divide il full path in nome e path
file_path, file_name = os.path.split(file_name_full_path_no_extension)

# Read image
image = cv2.imread(file_name_full_path)

# Read the desired model
# path = "FSRCNN_x2.pb"  # ok
# path = "ESPCN_x4.pb"  # ok
path = "EDSR_x2.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
# sr.setModel("fsrcnn", 2)
# sr.setModel("espcn", 4)
sr.setModel("edsr", 2)

# Upscale the image
result = sr.upsample(image)

# nuovo file in directory originale
new_file = file_path + '/' + file_name + 'x2_edsr.tif'

print("New file: ", new_file)

# Save the image
cv2.imwrite(new_file, result)

print("Time elapsed: ", datetime.now() - startTime)


