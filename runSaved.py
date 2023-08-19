import numpy as np, os
import cv2 
from regressionFinal import predict
from model import HandDetector




w = []
b = 0
mean = []
std = []

models = os.listdir("savedModels")
print("SAVED MODELS . . . ")
for model in models:
    print(model)


model = models[int(input("Choose a model: ")) - 1]


with open("savedModels/"  + model, "r") as numbers:
    modelData = (numbers.read()).split(",")
    for val in modelData[0:14]:
        w.append(float(val))
    b = float(modelData[14])

    for val in modelData[15:29]:
        mean.append(float(val))
    
    for val in modelData[29:43]:
        std.append(float(val))
             

w = np.asarray(w) 

mean = np.asarray(mean)

std = np.asarray(std)

video = cv2.VideoCapture(0)


def main():
    detector = HandDetector()

    while True:
            check, frame = video.read()

            detector.set_image(frame)
            foundPose = detector.set_pose()
            
            if foundPose:
                no = "not"
                prediction = predict(detector.collectData(), w, b, mean, std, True)
                if prediction > 0.5:
                    no = ""
                cv2.putText(frame, f'sign {no} detected {prediction}%', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)
                #   finalX.append(detector.collectData())
                #  finalY.append(1)

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1)


            if key == ord("x"):
                break

main()
