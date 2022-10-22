from flask import Flask
from flask import request
from flask import render_template
import numpy as np
import pickle
import os


model = pickle.load(open('knn.pkl', 'rb'))

UPLOAD_FOLDER = './static/uploaded_images'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
image_home = 'https://techieloops.com/wp-content/uploads/2021/09/farm-automation-systems.jpg'

@app.route("/", methods=["GET", "POST"])
def predict():
    print('request.method: ', request.method)
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = UPLOAD_FOLDER + '/' + image_file.filename
            image_file.save(image_location)
            from tensorflow.keras.utils import load_img, img_to_array
            test_image = load_img(image_location, target_size=([220, 220]))
            test_image = img_to_array(test_image)
            test_image = np.array(test_image)
            test_image = test_image.reshape(1,220, 220, 3)
            test_image = test_image.reshape(1, 220 *220 *3)
            
            prediction = model.predict(test_image)
            prediction_proba = "{}%".format(model.predict_proba(test_image).max() * 100)

            if prediction == [1]:
                
                name = 'Sandy Soil'
                crop1 = 'Mellon'
                crop2 = 'Maize'
                crop3 = 'Corn'
                crop4 = 'Carrot'
                crop5 = 'Millets'
                crop6 = 'Zucchini'
                crop7 = 'Collard greens'
                crop8 = 'Radish'
                crop9 = 'Tomatoes'
                crop10 = 'Lettuce'
            elif prediction == [2]:
                
                name = "Clay soil"
                crop1 = 'Beans'
                crop2 = 'Brocolo'
                crop3 = 'Carrot'
                crop4 = 'Rice'
                crop5 = 'Paddy'
                crop6 = 'Swiss Chard'
                crop7 = 'Cabbage'
                crop8 = 'Sprouts'
                crop9 = 'Brussels'
                crop10 = 'Chard'
            elif prediction == [3]:
                
                name = "Loamy soil"
                crop1 = 'Wheat'
                crop2 = 'sugar cane'
                crop3 = 'Cotton'
                crop4 = 'Tomato'
                crop5 = 'Letuce'
                crop6 = 'Onions'
                crop7 = 'Cucumbers'
                crop8 = 'Peppers'
                crop9 = 'Jute'
                crop10 = 'Oilseeds'

        image = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename )
        return render_template("index.html", image = image, prediction_proba = prediction_proba, prediction = name, crop1 = crop1, crop2 = crop2, crop3 = crop3, crop4 = crop4, crop5 = crop5)
    return render_template("index.html", image = image_home )

if __name__ == '__main__':
    app.run(debug = True)
   
