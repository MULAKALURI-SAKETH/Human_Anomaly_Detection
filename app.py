from flask import Flask, render_template, request
# from bokeh.plotting import figure
# from bokeh.io import output_notebook, push_notebook
# import cv2
from tensorflow.keras.models import load_model
# import numpy as np
import warnings
# import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")

CLASSES = ["Normal", "Fire", "Accident", "Robbery"]
model = load_model(os.path.join(os.path.dirname(__file__), "models", "anomaly_detection.h5"))


app = Flask(__name__)


# Prediction from Video Input
# def detect_anomaly(path):
#     # Load video
#     output_notebook()

#     cap = cv2.VideoCapture(path)
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     frame = cv2.flip(frame, 0)
#     width = frame.shape[1]
#     height = frame.shape[0]
#     p = figure(
#         x_range=(0, width),
#         y_range=(0, height),
#         output_backend="webgl",
#         width=700,
#         height=400
#     )
#     myImage = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
#     label = ""
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             img = frame
#             img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
#             img = img.astype("float32") / 255.0
#             preds = model.predict(np.expand_dims(img, axis=0))[0]
#             j = np.argmax(preds)
#             label = CLASSES[j]
#             cv2.putText(frame, label, (35, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#             frame = cv2.flip(frame, 0)
#             myImage.data_source.data['image'] = [frame]
#             push_notebook()
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#         # time.sleep(0.2)
#     cap.release()
#     return label


@app.get("/health")
def health():
    return {"status": "ok!"}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/submit")
def submit():
    print(request.files)
    print("Submitted")
    # Write your detect anamoly here
    return render_template("index.html")


if __name__ == "__main__":
    app.run(port=3000, debug=True)
