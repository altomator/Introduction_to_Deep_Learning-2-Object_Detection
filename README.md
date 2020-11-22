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

Users may play with the confidence score value and look for the impact on the detection process. A basic filter on very large (and improbable) detections is implemented.

Display the Jupyter notebook with [nbviewer](https://nbviewer.jupyter.org/github/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/binder/faces-detection-with-dnn.ipynb).

Launch the notebook with Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/altomator/Introduction_to_Deep_Learning-2-Face_Detection/HEAD?filepath=https%3A%2F%2Fgithub.com%2Faltomator%2FIntroduction_to_Deep_Learning-2-Face_Detection%2Fblob%2Fmain%2Fbinder%2Ffaces-detection-with-dnn.ipynb)

## Use cases
- Information Retrieval: [GallicaPix](https://gallicapix.bnf.fr/rest?run=findIllustrations-app.xq&filter=1&start=1&action=first&module=1&locale=fr&similarity=&rValue=&gValue=&bValue=&corpus=1418&sourceTarget=&keyword=&kwTarget=&kwMode=&title=excelsior&author=&publisher=&fromDate=1915-01-01&toDate=1915-12-31&iptc=00&page=true&illTech=00&illFonction=00&illGenre=00&persType=faceM&classif1=&CBIR=*&classif2=&CS=0.5&operator=and&colName=00&size=31&density=26)
- Data analysis of newspapers front page: faces, gender (Excelsior, 1910-1920)
![Front pages analysis](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/faces-excelsior.jpg)






