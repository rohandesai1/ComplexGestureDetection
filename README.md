# **Custom Gesture Detection**

This project combines MediaPipe's advanced hand tracking, a custom-built regression model, and OpenCV's image processing to help users seamlessly train a model to recognize any gesture. This is done by extracting hundreds of ratios along a person's hand to receive examples/non-examples of a given gesture, and then feeding that data to a logistic regression model.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Full Code Breakdown](#code-breakdown)
- [Explanation](#explanation)

## Features

- Real-time hand gesture detection using webcam feed.
- Seamless training of a custom regression model for any gesture.
- Comprehensive hand feature extraction by analyzing various hand ratios and angles.
- Option to save and load trained models for future use.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:
- OpenCV
- MediaPipe
- NumPy

### Installation

1. Clone this repository:
\`\`\`
`git clone https://github.com/rohandesai1/CustomGestureDetection` 
\`\`\`

2. Navigate to the project directory:
\`\`\`
`cd ~/Path/To/Download`
\`\`\`

3. Install the required libraries:
\`\`\`
`pip install -r requirements.txt`
\`\`\`

## Usage

1. **Training a Model**:
   - Run `train.py` and follow the on-screen instructions.
   - Provide non-examples and examples of the gesture when prompted.
   - The trained model parameters will be saved in `numbers.txt`.

2. **Testing the Model**:
   - Run `test.py`.
   - Test the trained model in real-time using the webcam feed.
   - Save the model if desired.

3. **Running a Saved Model**:
   - Run `runSaved.py`.
   - Choose a saved model from the list and run it on real-time webcam feed.

## Explanation

### 1. Hand Detection and Feature Extraction (`model.py`)

The core of this project lies in detecting hand landmarks and extracting features from them.

- **HandDetector Class**: This class is responsible for hand detection and feature extraction.
    - **Initialization**: It uses MediaPipe's `vision.HandLandmarker` for hand landmark detection. [`source`](./model.py#L9-L14)
    - **set_image**: Prepares the image for detection by converting it to the required format. [`source`](./model.py#L16-L20)
    - **set_pose**: Detects hand landmarks. [`source`](./model.py#L22-L31)
    - **get_angle**: Computes the angle between three specified landmarks. Useful for capturing hand poses. [`source`](./model.py#L35-L51)
    - **collectData**: This function aggregates a set of hand ratios and angles, which act as the input features for the regression model. [`source`](./model.py#L75-L98)

### 2. Regression Model (`regressionFinal.py`)

The project uses a custom regression model (either logistic or linear) for gesture recognition.

- **logOrLinRegression**: This function trains the regression model on provided data. It's capable of both logistic and linear regression based on the input flag. [`source`](./regressionFinal.py#L5-L74)
    - It uses **gradient descent** for optimization, updating weights and biases iteratively.
    - The training data can be optionally normalized using Z-score normalization for better convergence.

- **predict**: Given the features extracted from a hand gesture, this function predicts the output using the trained regression model. For logistic regression, a sigmoid function computes the probability. [`source`](./regressionFinal.py#L97-L102)

### 3. Training (`train.py`)

Users can train the model on their own gestures in real-time using a webcam feed. [`source`](./train.py)
- The script first collects non-examples of a gesture.
- Then, it collects examples of the desired gesture.
- The collected data is then used to train the regression model.

### 4. Testing and Running Models (`test.py` and `runSaved.py`)

- **Testing**: After training, users can test the model in real-time on a webcam feed using `test.py`. They also have the option to save the trained model for future use. [`source`](./test.py)
- **Running Saved Models**: Users can run any previously saved models using `runSaved.py`. This allows for easy switching between different gesture models. [`source`](./runSaved.py)



## Code Breakdown 
   * ## Step 1: Video Processing
     
     ```Python
      # train.py
       
      def main():
          detector = HandDetector()
          finalX = []
          finalY = []
      
          while True:
              check, frame = video.read()
      
              detector.set_image(frame)
              foundPose = detector.set_pose()
     ```
       - The main function feeds the video into the HandDetector() class which will use mediapipe's hand landmarker tasks to locate key points of the hand
     
     ```Python
     # model.py
     
     class HandDetector:
         def __init__ (self):
            self.landmarks = []
            base_options = python.BaseOptions(model_asset_path='/Users/rohandesai/Downloads/hand_landmarker.task')
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
     ```
        - If a hand is detected, the landmark positions (key points of the hand) are saved in `self.landmarks`
          
     ```Python
     # train.py
     
     if foundPose:
       finalX.append(detector.collectData())
     ```
       - Once these points are saved, the data collection process initiates
   * ## Step 2: Data Collection
     ```Python
     # model.py
     
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
     ```
       - The points are processed in two key ways, ration of distances and angles.
         
     ```Python
     # model.py
     
     def get_distances(self, points): # all set of distances between any number of points 
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

     ```
       - Distance is calculated using the distance formula.
         
     ```Python
     # model.py
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
     ```
       - Ratios are calculated by finding all permutations of all distances, then consequently dividing each permutation
         
     ```Python
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
     ```
       - The angle between 3 points is calculated by first finding the distances between the 3 points, and then using the arc cosine trigonometric function.
     ```Python
     while True:
        check, frame = video.read()

        detector.set_image(frame)
        foundPose = detector.set_pose()

        if foundPose:
                finalX.append(detector.collectData())
                finalY.append(0)

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

        cv2.putText(frame, "PROVIDE EXAMPLES, PRESS X WHEN DONE", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)

        if key == ord("x"):
            break

     ```
       - This process is repeated for all examples and non-exampels of whatever gesture the user would like to train for

   * ## Step 3: Format Data
     ```Python
     temp = list(zip(finalX, finalY))
     random.shuffle(temp)
      
     finalX, finalY = zip(*temp)
     finalX, finalY = list(finalX), list(finalY)
  
     finalX = np.asarray(finalX)
     finalY = np.asarray(finalY)

     ```
       - The data is shuffled and converted into a numpy array so it can be fed into the regression model


   * ## Step 4: Train the Model
     - A deep dive into how regression models work can be found [here](https://www.coursera.org/learn/machine-learning) . I have written my own regression model which works for lin/log regression, located at `regressionFinal.py`.

   * ## Step 5: Test/Save the Model
     ```Python
     # test.py
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
     ```
       - Loading the parameters of the model
     ```Python
     
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
     ```
       - And finally run the model on a real-time feed of the webcam
     ```Python
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

     ```
       - Along with the option to save the model if you would like.
         
      
     

    



       
     
     
     
        
