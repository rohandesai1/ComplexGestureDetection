# **Custom Gesture Detection**

## Introduction 

  This project combines MediaPipe's advanced hand tracking, a custom-built regression model, and OpenCV's image processing to help users seamlessly train a model to recognize any gesture. This program extracts hundreds of ratios along a person's hand to extract examples/non-examples of a given gesture, and then feeds that data to a logistic regression model. 

## Usage

  To run this, you must do the following:
  
  1. Enter this command to clone the repository
  <br></br>
  `git clone https://github.com/rohandesai1/CustomGestureDetection` 
  <br></br>
  2. Insall the required dependencies
  <br></br>
  `pip install -r requirements.txt`
  <br></br>
  3. Run the Program.
  <br></br>
  `python runSaved.py` or `python train.py`
  

