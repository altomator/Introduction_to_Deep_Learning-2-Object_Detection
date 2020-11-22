# Introduction to Deep Learning #2
**Face Detection**

Use case: face detection on heritage images 

# Goals 
Annotation of visages on heritage material for information retrieval, quantitative analysis, etc.

![Face detection](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/visage.png)

## Hands-on session 

The [OpenCV/dnn](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) module can be used to try some pretrained neural network models imported from frameworks as Caffe or Tensorflow.

This Python 3 script uses dnn to call a ResNet SSD network (see [this post](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) for details). The model can be easily downloaded from the web.

The images of a Gallica document are first loaded thanks to the IIIF protocol. The detection then occurs and annotated images are generated, as well as the CSV data. 

Users may play with the confidence score value and look for the impact on the detection process. A basic filter on very large detections is implemented.

Display the Jupyter notebook with nbviewer.

Launch the notebook with Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/altomator/Introduction_to_Deep_Learning-2-Face_Detection/HEAD?filepath=https%3A%2F%2Fgithub.com%2Faltomator%2FIntroduction_to_Deep_Learning-2-Face_Detection%2Fblob%2Fmain%2Fbinder%2Ffaces-detection-with-dnn.ipynb)
