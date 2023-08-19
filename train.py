from model import HandDetector
import cv2, numpy as np, random
from regressionFinal import logOrLinRegression


video = cv2.VideoCapture(0)

def main():
    detector = HandDetector()
    finalX = []
    finalY = []

    while True:
        check, frame = video.read()

        detector.set_image(frame)
        foundPose = detector.set_pose()

        if foundPose:
            
                finalX.append(detector.collectData())
                finalY.append(0)

             #   finalX.append(detector.collectData())
              #  finalY.append(1)

        cv2.putText(frame, "PROVIDE NON EXAMPLES, PRESS X WHEN DONE", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)


        if key == ord("x"):
            break
    
    while True:
        check, frame = video.read()

        detector.set_image(frame)
        foundPose = detector.set_pose()

        if foundPose:
            
                finalX.append(detector.collectData())
                finalY.append(1)

             #   finalX.append(detector.collectData())
              #  finalY.append(1)



            

        cv2.putText(frame, "PROVIDE EXAMPLES, PRESS X WHEN DONE", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)


        if key == ord("x"):
            break

    temp = list(zip(finalX, finalY))
    random.shuffle(temp)
    
    finalX, finalY = zip(*temp)
    finalX, finalY = list(finalX), list(finalY)


    finalX = np.asarray(finalX)
    finalY = np.asarray(finalY)

    
    w, b, cost_history, mean, std = logOrLinRegression(finalX, finalY, np.zeros(finalX.shape[1]), 0, 2000, 0.001, True, True)

    with open("/Users/rohandesai/Development/Teach/GestureDetection/ComplexGestureDetection/numbers.txt", "w") as numbers:
        for i in range(len(w)):
            numbers.write(f'{w[i]},')

        
        numbers.write(f'{b},')

        for i in range(len(mean)):
            numbers.write(f'{mean[i]},')


        for i in range(len(std)):
            numbers.write(f'{std[i]},')


                 






main()