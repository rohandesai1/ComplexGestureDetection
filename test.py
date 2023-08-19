import numpy as np
import cv2, random
from regressionFinal import predict
from model import HandDetector

def get_params():
    w = []
    b = 0
    mean = []
    std = []


    with open("numbers.txt", "r") as numbers:
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

    return w, b, mean, std

video = cv2.VideoCapture(0)


def main():
    detector = HandDetector()
    w, b, mean, std = get_params()
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

            cv2.putText(frame, "key X to close", (frame.shape[0] - 200, int((frame.shape[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

                #   finalX.append(detector.collectData())
                #  finalY.append(1)

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1)


            if key == ord("x"):
                break
                
    while True:
        frame = np.zeros((720,1280))
        cv2.putText(frame,"Click S if you want to save the model, X if you don't", (5,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.imshow("Save", frame)
        key = cv2.waitKey(1)
        
        if key == ord("s"):
            with open(f"savedModels/model{random.randint(0,1000)}.txt", "w") as toWrite:
                with open("numbers.txt", "r") as data:        
                    toWrite.write(data.read())
            break

        if key == ord("x"):
            break


if __name__ == "__main__":
    main()