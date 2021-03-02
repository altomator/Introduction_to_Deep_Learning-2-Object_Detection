# Introduction to Deep Learning #2
**Object detection**

Use case: object detection on heritage images 

## Goals 
Annotation of object on heritage material for information retrieval, quantitative analysis, etc.

![Object detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/objet.JPG)

<sup>[ark:/12148/bpt6k65413493](https://gallica.bnf.fr/ark:/12148/bpt6k65413493/f964.item)</sup>


## Hands-on session 

### YOLO v3 and v4
[YOLO](https://pjreddie.com/darknet/yolo/) performs object detection on a 80 classes model. YOLO is well known to be fast and accurate.

This [Python 3 script](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/binder/object-detection-with-yolo.py) uses a YOLO v4 model that can be easily downloaded from the web. The images of a Gallica document are first loaded thanks to the IIIF protocol. The detection then occurs and annotated images are generated, as well as the CSV data. 




### Google Cloud Vision, IBM Watson 

These APIs may be used to perform objects detection. The Perl script described [here](https://github.com/altomator/Image_Retrieval) calls the APIs. The API endpoint is simply called with a curl command sending the local image files which have been extracted with IIIF.

```
> perl toolbox.pl -CC datafile -ibm
```


### Other approaches

Google Cloud Vision, IBM Watson and other commercial APIs can be use for training a specific object detector on custom data, as [YOLO](https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208) 

## Use cases


## Resources

