from taipy.gui import Gui
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook
import cv2
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CLASSES = ["Normal", "Fire", "Accident", "Robbery"]
# import the necessary packages

# load the trained model from disk
print("[INFO] loading model...")
model = load_model('./models/anomaly_detection.h5')


# ### Prediction from Video Input
def detect_anomaly(path):
    #Load video
    output_notebook()

    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # because Bokeh expects a RGBA image
    frame = cv2.flip(frame, 0) # because Bokeh flips vertically
    width = frame.shape[1]
    height = frame.shape[0]
    p = figure(x_range=(0,width), y_range=(0,height), output_backend="webgl", width=700, height=400)
    myImage = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
    label = ""
    show(p, notebook_handle = True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            img = frame
            img = cv2.resize(img,(128, 128),interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0
            preds = model.predict(np.expand_dims(img, axis=0))[0]
            j = np.argmax(preds)
            label = CLASSES[j]
            #print("Predicted Class : ",label)
            cv2.putText(frame, label, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.flip(frame, 0)
            myImage.data_source.data['image'] = [frame]
            push_notebook()
            if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
                break
        else:
            break
        #time.sleep(0.2)
    print("Detected Activity: " + label)
    cap.release()
    print("Detected activity: " + label)

detect_anomaly(input('path:'))#/content/drive/My Drive/Colab Notebooks/DATA/ROBBERY.mp4
                    #   /content/drive/My Drive/Colab Notebooks/DATA/ACCIDENT.mp4





# img_path = "placeholder_image.png"
# content = ""
# anomaly_class = ""
# index = """
# <|text-center|

# <|{anomaly_class}|>

# <|{content}|file_selector|extensions=.mp4|>
# Upload the surveillance video

# |>
# """

# def on_change(state, var_name, var_val):
#     if var_name == "content":        
#         output = detect_anomaly(var_val)
#         state.anomaly_class = "Detected activity: " + str(output)
#     # print(var_name, var_val)


# app = Gui(page=index)

# if __name__ == '__main__':
#     app.run(use_reloader=True)

