from flask import Flask, request, jsonify, render_template, send_file,redirect
from werkzeug.utils import secure_filename
import requests
import zipfile
import os
import cv2
import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras import models

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import re
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import time
import math
import albumentations as A
import pytesseract

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DOWNLOAD_FOLDER = 'downloads/'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

model_path_1 = 'weights/resnet50_csv_10.h5'
model_path_2 = 'weights/resnet50_csv_20.h5'

def load_model_weights(model_path_1):
    # Load weights for first model price tag recognition
    model_1 = models.load_model(model_path_1, backbone_name='resnet50')
    model_1 = models.convert_model(model_1)

    # Load weights for first model price tag recognition


    return model_1


def model_tag_recognition(image):
    # Copy of image to crop
    crop = image.copy()
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model_1.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    # take our box
    box, score, label = boxes[0][0], scores[0][0], labels[0][0]

    # Make int coordinates
    b = box.astype(int)

    image_croped = crop[b[1]:b[3], b[0]:b[2]]
    image_croped = cv2.cvtColor(image_croped, cv2.COLOR_BGR2RGB)

    return image_croped

# Rotation functions
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size

    #print(nlines)
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        #print(ang)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))

# Function to crop price_id, price_lei and price_bani from price_tag
def model_elements_of_price_tag(image):
    # Prepare test images to crop from 
    crop = image.copy()
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Define a dataframe  coordinates
    df = pd.DataFrame(columns = ['class_id','score','x_min','x_max','y_min','y_max'])

    # # preprocess image for network
    # image = preprocess_image(image)
    # image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model_2.predict_on_batch(np.expand_dims(image, axis=0))
  
  # # correct for image scale
  # boxes /= scale
  
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.3:
            break

        # Make coordinates as int type
        b = box.astype(int)

        df = df.append({'class_id':label,
                'score':score,
                'x_min':b[1],
                'x_max':b[3],
                'y_min':b[0],
                'y_max':b[2]}, ignore_index=True)


    # Calculate best bounding box coordinates for each class ={1,2,3}
    # Class 1 - product_id
    class_1_best_score = df[df.class_id == 1].score.max()
    class_1_x_min = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].x_min)
    class_1_y_min = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].y_min)
    class_1_x_max = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].x_max)
    class_1_y_max = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].y_max)

    # Class 2 - price_lei
    class_2_best_score = df[df.class_id == 2].score.max()
    class_2_x_min = int(df[(df.class_id == 2)&(df.score == class_2_best_score)].x_min)
    class_2_y_min = int(df[(df.class_id == 2)&(df.score == class_2_best_score)].y_min)
    class_2_x_max = int(df[(df.class_id == 2)&(df.score == class_2_best_score)].x_max)
    class_2_y_max = int(df[(df.class_id == 2)&(df.score == class_2_best_score)].y_max)

    # Class 3 - price_bani
    class_3_best_score = df[df.class_id == 3].score.max()
    class_3_x_min = int(df[(df.class_id == 3)&(df.score == class_3_best_score)].x_min)
    class_3_y_min = int(df[(df.class_id == 3)&(df.score == class_3_best_score)].y_min)
    class_3_x_max = int(df[(df.class_id == 3)&(df.score == class_3_best_score)].x_max)
    class_3_y_max = int(df[(df.class_id == 3)&(df.score == class_3_best_score)].y_max)

    crop_id = crop[class_1_x_min:class_1_x_max, class_1_y_min:class_1_y_max]
    crop_lei = crop[class_2_x_min:class_2_x_max, class_2_y_min:class_2_y_max]
    crop_bani = crop[class_3_x_min:class_3_x_max, class_3_y_min:class_3_y_max]

    return crop_id, crop_lei, crop_bani

# Text from product_name
def text_from_name(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_list = pytesseract.image_to_string(img_rgb).split('\n')


    # using join() + generator to remove special characters
    special_char = '@_!#$^&*()<>?/\|}{~:;.[]'
    out_list = [''.join(x for x in string if not x in special_char) for string in test_list]

    # trim spaces
    out_list = [elem.strip() for elem in out_list]

    # remove Null elements
    out_list = list(filter(None, out_list))

    if len(out_list) == 0:
        out_list.append('')
    return out_list[0]

# Text from croped_id
def text_from_id(image):  
    # Set config for tesseract
    config=' --psm 13 --oem 3 -c tessedit_char_whitelist=0123456789'

    img = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    text_list = str(pytesseract.image_to_string(img, config=config).split('\n')[0])

    # using join() + generator to remove special characters
    special_char = '@_-!#$^&*()<>?/\|}{~:;.[]'
    out_list = [''.join(x for x in string if not x in special_char) for string in text_list]

    # trim spaces
    out_list = [elem.strip() for elem in out_list]

    # remove Null elements
    out_list = list(filter(None, out_list))

    out_list = ''.join(out_list)

    return out_list

def text_from_price_lei(image):

    img = cv2.resize(image, None, fx=1.5 , fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config=' --psm 13 --oem 1 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(img, config=config).split('\n')

    return text[0]

def text_from_price_bani(image):

    img = cv2.resize(image, None, fx=1.5 , fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Set config for tesseract
    config=' --psm 13 --oem 3 -c tessedit_char_whitelist=0123456789'
    text = str(pytesseract.image_to_string(img, config=config)).split('\n')[0]

    return text

def price_tag_detector(file):
    image_loaded = read_image_bgr(file)
    image_price_tag = model_tag_recognition(image_loaded)
    image_deskewed = deskew(image_price_tag)

    croped_id, croped_lei ,croped_bani = model_elements_of_price_tag(image_deskewed)

    text_name = text_from_name(image_deskewed)

    text_id = text_from_id(croped_id)

    text_lei = text_from_price_lei(croped_lei)

    text_bani = text_from_price_bani(croped_bani)

    return text_name, text_id, text_lei, text_bani



model_1 = load_model_weights(model_path_1)
model_2 = load_model_weights(model_path_2)




@app.route("/", methods=["POST", "GET"])
def get_files():
    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        args["method"] = "POST"
        filename = secure_filename(file.filename)
        zipp = zipfile.ZipFile(file)
        zipp.extractall(UPLOAD_FOLDER)
        df_text = pd.DataFrame(columns = ['File','Product_name','Product_ID','Price'])
        for file in os.listdir(UPLOAD_FOLDER):
            

            text_name, text_id, text_lei, text_bani = price_tag_detector(UPLOAD_FOLDER+file)
            df_text = df_text.append({'File':file,
                          'Product_name':text_name,
                          'Product_ID':text_id,
                          'Price': text_lei + '.' + text_bani}, ignore_index=True)


        df_text.to_csv(DOWNLOAD_FOLDER + 'result.csv', index=False)
        return redirect('/downloadfile/'+ 'result.csv')
    return render_template("upload_form.html", args=args)


@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html', value=filename)

@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = DOWNLOAD_FOLDER + filename
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=False)