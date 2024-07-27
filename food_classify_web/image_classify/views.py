from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import cv2
from skimage import io

def predict(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image to the database
            uploaded_image = form.save()

            # Run prediction code
            image_path = uploaded_image.image.path
            m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')

            input_shape = (224, 224)
            image = np.asarray(io.imread(image_path), dtype="float")
            image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
            image = image / image.max()
            images = np.expand_dims(image, 0)

            output = m(images)
            predicted_index = output.numpy().argmax()
            print(predicted_index)
            classes = pd.read_csv("aiy_food_V1_labelmap.csv")
            food_info = pd.read_csv("food_info.csv", encoding='utf-8')

            # Extract additional information based on predicted class
            food_name = classes["name"][predicted_index]
            price = form.cleaned_data['price']
            quantity, calories, carbohydrate, fat, protein = get_food_info(food_info, food_name)

            prediction = {
                'food_name': food_name,
                'price': price,
                'quantity': quantity,
                'calories': calories,
                'carbohydrate': carbohydrate,
                'fat': fat,
                'protein': protein,
            }

            # Render the result page
            return render(request, 'prediction.html', {'prediction': prediction})
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {'form': form})

def get_food_info(food_info_df, food_name):
    # Function to get additional information for a given food_name from the food_info_df
    food_row = food_info_df[food_info_df['name'] == food_name].iloc[0]
    return food_row['quantity'], food_row['calories'], food_row['carbohydrate'], food_row['fat'], food_row['protein']