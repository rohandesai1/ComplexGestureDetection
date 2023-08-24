import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, numpy as np, math
    

class HandDetector:
    def __init__ (self):

        self.landmarks = []
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def set_image(self, image):
        self.detector_image = mp.Image(image_format= mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.height, self.width, _ = image.shape

    def set_pose(self):
        self.landmarks = (self.detector.detect(self.detector_image).hand_landmarks)

        if self.landmarks != []:     
            self.landmarks = self.landmarks[0]
            return True
        return False

    def get_landmark(self, bodyPart):
        try: 
            self.landmarks[bodyPart]
        except (IndexError): 
            return []
         
        x = self.landmarks[bodyPart].x * self.width
        y = self.landmarks[bodyPart].y  * self.height
        z = self.landmarks[bodyPart].z 

        return[x,y,z]
    
    def get_angle(self, bp1, bp2, bp3): # bp -> body part, calc angle between 3 body parts
        p1 = self.get_landmark(bp1)
        p2 = self.get_landmark(bp2)
        p3 = self.get_landmark(bp3)

        
        if p1 != [] and p2 != [] and p3 != []: # requires all 3 values to calculate angle

            locations = [p1, p2, p3]
            distances = self.get_distances(locations)
            a, b, c = np.asarray(distances)/1000
            
            angleInRadians = math.acos((a ** 2 + b ** 2 - c ** 2)/ (2 * a * b))
            angleInDegrees = angleInRadians * (180.0 / math.pi)

            return angleInDegrees
        else:
            return 0

    
    def get_distances(self, points): # all set of distances between any number of points (used for a,b,c in a triangle)
        distances = []
        numOfPoints = len(points)
        for i in range(numOfPoints):
            point1 = points[i]
            if i == numOfPoints - 1:
              point2 = points[0]
            else:
              point2 = points[i+1]

            x_dis = abs(point1[0] - point2[0])
            y_dis = abs(point1[1] - point2[1])
            

            distance = math.sqrt((abs(y_dis ** 2 + x_dis ** 2)))

            distances.append(distance)
        return distances

    
    def collectData(self):
        finalSet = []
        distanceData = []
        
        for i in range(5):
            p1 = self.get_landmark(1 + i * 4)
            p2 = self.get_landmark(2 + i * 4)
            distanceData.append(self.get_distances([p1, p2])[0])

        for perm in self.permutations(distanceData, 2):
            finalSet.append(perm[0]/perm[1])

        
        
        for i in range(4):
            angle = self.get_angle(4 + (4*i), 0, 8 + (4*i))
            finalSet.append(angle)


        return finalSet

    def permutations(self, list, length):
        all = []
        
        def reverse(list):
            reversed = []
            for item in list:
                reversed = [item] + reversed
            return reversed

        def func(start, end=[]):
            if (len(start) == abs(length - len(list))):
                noAppend = False
                for i in range(len(all)):
                    if all[i] == reverse(end):
                        noAppend = True

                if noAppend != True:
                    all.append(end)
            else:
                for i in range(len(start)):
                    func(start[:i] + start[i + 1:], end + start[i:i + 1])

        func(list)
        return all


    


