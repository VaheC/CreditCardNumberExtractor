import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
# import streamlit as st

def get_card_xy(model_path, image_path):
    #model_path = 'odo_detector.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Obtain the height and width of the corresponding image from the input tensor
    image_height = input_details[0]['shape'][2] # 640
    image_width = input_details[0]['shape'][3] # 640

    # Image Preparation
    # image_name = 'car.jpg'
    image = Image.open(image_path)
    image_resized = image.resize((image_width, image_height)) # Resize the image to the corresponding size of the input tensor and store it in a new variable

    image_np = np.array(image_resized) #
    image_np = np.true_divide(image_np, 255, dtype=np.float32) 
    image_np = np.moveaxis(image_np, -1, 0)
    image_np = image_np[np.newaxis, :]

    # inference
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()

    # Obtaining output results
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output[0]
    output = output.T

    boxes_xywh = output[:, :4] #Get coordinates of bounding box, first 4 columns of output tensor
    scores = output[:, 4]#np.max(output[..., 5:], axis=1) #Get score value, 5th column of output tensor
    classes = np.zeros(len(scores))#np.argmax(output[..., 5:], axis=1) # Get the class value, get the 6th and subsequent columns of the output tensor, and store the largest value in the output tensor.

    # Threshold Setting
    # threshold = 0.7
    final_score = 0
    x_center, y_center, width, height = 0, 0, 0, 0
    class_name = 'card_number'

    # Bounding boxes, scores, and classes are drawn on the image
    # draw = ImageDraw.Draw(image_resized)

    for box, score, cls in zip(boxes_xywh, scores, classes):
        if score >= final_score:
            x_center, y_center, width, height = box
            final_score = score
            class_name = cls
        else:
            pass
            
    # x1 = int((x_center - width / 2) * image_width)
    # y1 = int((y_center - height / 2) * image_height)
    # x2 = int((x_center + width / 2) * image_width)
    # y2 = int((y_center + height / 2) * image_height)

    output_image_width = 640
    output_image_height = 640
    x1 = int((x_center - width / 2) * output_image_width)
    y1 = int((y_center - height / 2) * output_image_height)
    x2 = int((x_center + width / 2) * output_image_width)
    y2 = int((y_center + height / 2) * output_image_height)

    # draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    # text = f"Class: {class_name}, Score: {final_score:.2f}"
    # draw.text((x1, y1), text, fill="red")

     # Saving Images
    # image_resized.save('test_img.jpg')

    return x1, y1, x2, y2, final_score

def get_digit(model_path, image_path, threshold=0.5):

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Obtain the height and width of the corresponding image from the input tensor
    image_height = input_details[0]['shape'][1] # 640
    image_width = input_details[0]['shape'][2] # 640

    # Image Preparation
    # image_name = 'car.jpg'
    # image = Image.open(image_path2)
    # image_resized = image.resize((image_width, image_height)) # Resize the image to the corresponding size of the input tensor and store it in a new variable
    image = cv2.imread(image_path)
    # image_resized = np.resize(image, (image_width, image_height, 3))

    image_np = np.array(image) #
    image_np = np.true_divide(image_np, 255, dtype=np.float32) 
    image_np = image_np[np.newaxis, :]

    # inference
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()

    # Obtaining output results
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output[0]
    output = output.T

    boxes_xywh = output[:, :4] #Get coordinates of bounding box, first 4 columns of output tensor
    scores = np.max(output[:, 4:], axis=1) #Get score value, 5th column of output tensor
    classes = np.argmax(output[:, 4:], axis=1) # Get the class value, get the 6th and subsequent columns of the output tensor, and store the largest value in the output tensor.

    pred_list = []

    prob_threshold = threshold

    for box, score, cls in zip(boxes_xywh, scores, classes):

        if score < prob_threshold:
            continue

        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        pred_list.append((x1, x2, cls, score))

    pred_list = sorted(pred_list, key=lambda x: x[0])

    num_list = []

    temp_pred_list =[]

    x_prev = 0

    x_diff = min([elem[1] - elem[0] for elem in pred_list]) - 10

    for idx, pred in enumerate(pred_list):
    
        if idx == 0:
            temp_pred_list.append(pred)
            x_prev = pred[0]
        elif idx == len(pred_list) - 1:
            temp_final_num = sorted(temp_pred_list, key=lambda x: x[-1], reverse=True)[0]
            num_list.append(temp_final_num)
        elif pred[0] - x_prev < x_diff:
            temp_pred_list.append(pred)
            x_prev = pred[0]
        else:
            temp_final_num = sorted(temp_pred_list, key=lambda x: x[-1], reverse=True)[0]
            num_list.append(temp_final_num)
            temp_pred_list = []
            x_prev = pred[0]
            temp_pred_list.append(pred)

    sorted_number_list = sorted(num_list, key=lambda x: x[0])
    # sorted_number_list = sorted(sorted_number_list, reverse=True, key= lambda x: x[-1])
    # output_digit = float(''.join([str(int(i[2])) if i[2]!=10 else '.' for i in sorted_number_list]))
    output_digit = float(''.join([str(int(i[2])) if i[2]!=10 else '.' for i in sorted_number_list]))
    # output_digit = ''.join([str(int(i[2])) if i[2]!=10 else '.' for i in sorted_number_list[:10]])

    return output_digit
   