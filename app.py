import os
import json
from flask import Flask, render_template, request
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from googlesearch import search
import pandas as pd

app = Flask(__name__, template_folder='templates')
# Load the disease solutions from Excel file
# Load the disease solutions from Excel file
solutions_df = pd.read_excel('leaf_disease_dataset.xlsx')

# Replace hyphens with underscores in the 'Class' column
solutions_df['Class'] = solutions_df['Class'].str.replace('-', '_')

# Create a dictionary from the DataFrame
solutions = dict(zip(solutions_df['Class'], solutions_df['Solution']))





# Load the pre-trained model
model = tf.keras.models.load_model('improved_leaf_disease_model.h5')


# List of class labels
labels = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy','Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)__Common_rust','Corn_(maize)___healthy',
'Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___healthy',
'Potato___Late_blight','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___healthy',
'Strawberry___Leaf_scorch','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight',
'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
'Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]

# Mapping of predicted classes to solutions


def google_search(query, num_results=5):
    try:
        return list(search(query, num_results=num_results))
    except Exception as e:
        print(f"Google search failed: {str(e)}")
        return []


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the form
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Create the 'static/uploads' folder if it doesn't exist
            if not os.path.exists('static/uploads'):
                os.makedirs('static/uploads')

            # Save the uploaded file to 'static/uploads'
            image_filename = uploaded_file.filename
            image_path = os.path.join('static/uploads', image_filename)
            uploaded_file.save(image_path)

            # Preprocess the image
            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)
            predicted_class_label = labels[predicted_class_index[0]]
            print("Length of Predicted Class Label:", len(predicted_class_label))
            print("Length of Keys in the Solutions Dictionary:", [len(key) for key in solutions.keys()])


            # Get the solution for the predicted class
            solution = solutions.get(predicted_class_label, 'No specific solution available.')
            

            # Perform Google search for solutions
            query = f"{predicted_class_label} disease solutions"
            search_results = google_search(query)
            
            context = {
                    "predicted_class": predicted_class_label,
                    "solution": solution,
                    "image_path":image_path, 
                    "image_filename":image_filename,
                    "search_results_json":(search_results) 
                    }
            print(context)


            # Render the result template with the predicted class, image path, solution, and search results
            return render_template('result.html',**context)

    # Render the main page template
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
