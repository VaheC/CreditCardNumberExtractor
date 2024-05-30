import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
# import imutils
# import easyocr
# import os
# import pathlib
# import platform
# from xyxy_converter import yolov5_to_image_coordinates
# import shutil
from extractor import get_card_xy, get_digit

# system_platform = platform.system()
# if system_platform == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# CUR_DIR = os.getcwd()
# YOLO_PATH = f"{CUR_DIR}/yolov5"
# MODEL_PATH = "runs/train/exp/weights/best.pt"

def main():
    st.title("Card number extractor")

    # Use st.camera to capture images from the user's camera
    img_file_buffer = st.camera_input(label='Please, take a photo of a card', key="card")

    # Check if an image is captured
    if img_file_buffer is not None:
        # Convert the image to a NumPy array
        image = Image.open(img_file_buffer)
        image_np = np.array(image)
        resized_image = cv2.resize(image_np, (640, 640))
        resized_image = resized_image.astype(np.uint8)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('card_image.jpg', resized_image)

        # original_img = cv2.imread('card_image.jpg')
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        x1, y1, x2, y2, card_confidence = get_card_xy(
            model_path='credit_card_number_detector.tflite',
            image_path='card_image.jpg'
        )

        st.write(card_confidence)

        if card_confidence == 0:
            display_text = "A card is not detected in the image!!!"
            st.image('card_image.jpg', caption=f"{display_text}", use_column_width=True)
        else:
            cropped_image = gray[y1:y2, x1:x2]
            # cropped_image = resized_image[y1:y2, x1:x2]
            cropped_image = cv2.resize(cropped_image, (640, 640))
            cv2.imwrite('card_number_image.jpg', cropped_image)
            
            extracted_digit = get_digit(
                model_path="card_number_extractor.tflite", 
                image_path='card_number_image.jpg', 
                threshold=0.4
            )

            display_text = f'Here is the zoomed card number: {extracted_digit}'
            st.image('card_number_image.jpg', caption=f"{display_text}", use_column_width=True)

            image = Image.open('card_image.jpg')
            image_resized = image.resize((640, 640))
            draw = ImageDraw.Draw(image_resized)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            class_name = 'card'
            text = f"Class: {class_name}, Confidence: {card_confidence:.2f}"
            draw.text((x1, y1), text, fill="red")
            # Saving Images
            image_resized.save('card_highlighted_image.jpg')
            display_text = 'Here is the card on the image.'
            st.image('card_highlighted_image.jpg', caption=f"{display_text}", use_column_width=True)

        st.session_state.pop("card")

if __name__ == "__main__":
    main()