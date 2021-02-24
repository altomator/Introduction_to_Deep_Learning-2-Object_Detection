# Introduction to Deep Learning #2
**Object detection**

Use case: object detection on heritage images 

## Goals 
Annotation of object on heritage material for information retrieval, quantitative analysis, etc.

![Object detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/objet.JPG)

<sup>[ark:/12148/bpt6k65413493](https://gallica.bnf.fr/ark:/12148/bpt6k65413493/f964.item)</sup>


## Hands-on session 

### YOLO v3 and v4
YOLO v3 (https://pjreddie.com/darknet/yolo/) performs object detection on a 80 classes model. YOLO is well known to be fast and accurate.

This [Python 3 script](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/binder/yolov4.py) uses dnn to ca The model can be easily downloaded from the web.

The images of a Gallica document are first loaded thanks to the IIIF protocol. The detection then occurs and annotated images are generated, as well as the CSV data. 



Display the Jupyter notebook with [nbviewer](https://nbviewer.jupyter.org/github/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/binder/faces-detection-with-dnn.ipynb).

Launch the notebook with Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/altomator/Introduction_to_Deep_Learning-2-Face_Detection/HEAD?filepath=https%3A%2F%2Fgithub.com%2Faltomator%2FIntroduction_to_Deep_Learning-2-Face_Detection%2Fblob%2Fmain%2Fbinder%2Ffaces-detection-with-dnn.ipynb)



### Google Cloud Vision, IBM Watson 

These APIs may be used to perform objects detection. The Perl script described [here](https://github.com/altomator/Image_Retrieval) calls the APIs.



Then the API endpoint is simply called with a curl command:

```
curl --max-time 10 -v -s -H "Content-Type: application/json" https://vision.googleapis.com/v1/images:annotate?key=your_key --data-binary @/tmp/request.json
```


### Others approaches



## Use cases

## Resources

