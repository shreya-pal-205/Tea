
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests

API_KEY = "3216916a7343004f4b6944ba5b9a5cfb"

app = Flask(__name__)






import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

# -------------------- CBAM BLOCK --------------------
def cbam_block(feature_map, ratio=8):
    channel = feature_map.shape[-1]

    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(feature_map)
    avg_pool = Dense(channel // ratio, activation='relu')(avg_pool)
    avg_pool = Dense(channel)(avg_pool)

    max_pool = GlobalMaxPooling2D()(feature_map)
    max_pool = Dense(channel // ratio, activation='relu')(max_pool)
    max_pool = Dense(channel)(max_pool)

    channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
    channel_attention = Reshape((1,1,channel))(channel_attention)
    feature_map = Multiply()([feature_map, channel_attention])

    # Spatial Attention
    avg_spatial = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(feature_map)
    max_spatial = Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(feature_map)
    concat = Concatenate(axis=3)([avg_spatial, max_spatial])
    spatial_attention = Conv2D(filters=1, kernel_size=7, padding='same',
                               activation='sigmoid')(concat)
    refined_feature = Multiply()([feature_map, spatial_attention])

    return refined_feature


# -------------------- INCEPTION MODULE --------------------
def inception_module(x, filters):
    f1, f3_in, f3_out, f5_in, f5_out, pool_proj = filters

    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(x)

    conv3 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(x)
    conv3 = Conv2D(f3_out, (3,3), padding='same', activation='relu')(conv3)

    conv5 = Conv2D(f5_in, (1,1), padding='same', activation='relu')(x)
    conv5 = Conv2D(f5_out, (5,5), padding='same', activation='relu')(conv5)

    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    pool = Conv2D(pool_proj, (1,1), padding='same', activation='relu')(pool)

    output = concatenate([conv1, conv3, conv5, pool], axis=3)
    return output


# -------------------- GOOGLENET + CBAM --------------------
def GoogLeNet_CBAM(input_shape=(224,224,3), num_classes=6):
    inp = Input(shape=input_shape)

    x = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(inp)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = Conv2D(64, (1,1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = inception_module(x, (64, 96, 128, 16, 32, 32))
    x = inception_module(x, (128, 128, 192, 32, 96, 64))
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = cbam_block(x)

    x = inception_module(x, (192, 96, 208, 16, 48, 64))
    x = inception_module(x, (160, 112, 224, 24, 64, 64))
    x = inception_module(x, (128, 128, 256, 24, 64, 64))
    x = inception_module(x, (112, 144, 288, 32, 64, 64))
    x = inception_module(x, (256, 160, 320, 32, 128, 128))
    x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = cbam_block(x)

    x = inception_module(x, (256, 160, 320, 32, 128, 128))
    x = inception_module(x, (384, 192, 384, 48, 128, 128))

    x = cbam_block(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model



# Load model
model = GoogLeNet_CBAM()
model.load_weights("best_attention_model.h5")

# Example class labels (change according to your dataset)
idx_to_class = {
    0: "algal_spot",
    1: "brown_blight",
    2: "gray_blight",
    3: "healthy",
    4: "helopeltis",
    5: "red_spot"
}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



def get_smart_remedy(label, confidence):

    # Severity logic
    if confidence < 70:
        severity = "Mild"
    elif confidence < 90:
        severity = "Moderate"
    else:
        severity = "Severe"

    remedies = {
        "brown_blight": {
            "treatment": "Apply copper fungicide regularly",
            "organic": "Neem oil spray weekly",
            "chemical": "Copper oxychloride",
            "recovery": "2-3 weeks"
        },
        "red_spot": {
            "treatment": "Remove infected leaves",
            "organic": "Baking soda spray",
            "chemical": "Chlorothalonil",
            "recovery": "1-2 weeks"
        },
        "gray_blight": {
            "treatment": "Improve drainage",
            "organic": "Compost tea spray",
            "chemical": "Mancozeb",
            "recovery": "2 weeks"
        },
        "healthy": {
            "treatment": "No disease detected",
            "organic": "Maintain soil health",
            "chemical": "None",
            "recovery": "Healthy"
        }
    }

    info = remedies.get(label, {
        "treatment": "Consult expert",
        "organic": "Neem solution",
        "chemical": "General fungicide",
        "recovery": "Varies"
    })

    return severity, info


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    label = idx_to_class[pred_idx]
    confidence = float(preds[0][pred_idx] * 100)

    return label, confidence


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)
            severity, remedy = get_smart_remedy(label, confidence)

            return render_template("predict.html",
                                   prediction=label,
                                   confidence=round(confidence, 2),
                                   severity=severity,
                                   remedy=remedy,
                                   image_path=filepath)

    return render_template("predict.html")




@app.route("/weather-page")
def weather_page():
    return render_template("weather.html")



@app.route("/weather", methods=["POST"])
def weather():
    try:
        data = request.json
        lat = data.get("lat")
        lon = data.get("lon")

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

        response = requests.get(url)
        res = response.json()

        if response.status_code != 200 or "main" not in res:
            return jsonify({
                "temp": "N/A",
                "humidity": "N/A",
                "condition": "API Error",
                "risk": "⚠️ Unable to detect risk",
                "advice": "Weather data unavailable"
            })

        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        condition = res["weather"][0]["main"]

        # 🌿 Risk Logic
        if humidity > 70 and "Rain" in condition:
            risk = "⚠️ High fungal disease risk"
        elif humidity > 60:
            risk = "⚠️ Moderate disease risk"
        else:
            risk = "✅ Low disease risk"

        # 🌱 Tea Recommendation Logic
        if 18 <= temp <= 30 and humidity > 60:
            advice = "🌱 Ideal climate for tea plantation. Maintain drainage and pruning."
        elif temp > 30:
            advice = "🔥 Too hot! Provide shade and irrigation."
        elif temp < 15:
            advice = "❄️ Too cold! Growth may slow. Protect plants."
        else:
            advice = "🌿 Moderate conditions. Monitor plant health regularly."

        return jsonify({
            "temp": f"{temp}°C",
            "humidity": f"{humidity}%",
            "condition": condition,
            "risk": risk,
            "advice": advice
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "temp": "Error",
            "humidity": "Error",
            "condition": "Server error",
            "risk": "⚠️ Try again",
            "advice": "⚠️ No advice available"
        })

if __name__ == "__main__":
    app.run(debug=True)
