# Introduction to Deep Learning #2
**Object detection**

Use case: object detection on heritage images 

## Goals 
Annotation of object on heritage material for information retrieval, quantitative analysis, etc.

![Object detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/objet.jpg)

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

For Google, first, we have to build a JSON request:

```
{
	"requests": [{
		"image": {
			"content": "binary image content"
		},
		"features": [{
			"type": "FACE_DETECTION",
			"maxResults": "30"
		}],
		"imageContext": {
			"languageHints": ["fr"]
		}
	}]
}
```                 

Then the API endpoint is simply called with a curl command:

```
curl --max-time 10 -v -s -H "Content-Type: application/json" https://vision.googleapis.com/v1/images:annotate?key=your_key --data-binary @/tmp/request.json
```


### Others approaches



## Use cases
- Information Retrieval: [GallicaPix](https://gallicapix.bnf.fr/rest?run=findIllustrations-app.xq&filter=1&start=1&action=first&module=1&locale=fr&similarity=&rValue=&gValue=&bValue=&corpus=1418&sourceTarget=&keyword=&kwTarget=&kwMode=&title=excelsior&author=&publisher=&fromDate=1915-01-01&toDate=1915-12-31&iptc=00&page=true&illTech=00&illFonction=00&illGenre=00&persType=face&classif1=&CBIR=*&classif2=&CS=0.5&operator=and&colName=00&size=31&density=26), faces in the 1915 issues of [_L'Excelsior_](https://gallica.bnf.fr/ark:/12148/cb32771891w/date.item)

- Data analysis of newspapers front page: faces, gender (Excelsior, 1910-1920)

![Front pages analysis: genders](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/faces-excelsior.jpg)

- Averaging of faces: _[A Century of Portraits: A Visual Historical Record of American High School Yearbooks](https://arxiv.org/abs/1511.02575)_ (UC Berkeley). Face detection is generally the first step of an averaging pipeline, which also implies detection of the facial features.

![Averaging of faces](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/averaging.jpg)

## Resources
- Face detection with [Google Cloud Vision](https://cloud.google.com/vision/docs/detecting-faces)
- [Kaggle faces dataset](https://www.kaggle.com/dataturks/face-detection-in-images)
- Face detection with [Detectron2](https://medium.com/@sidakw/face-detection-using-pytorch-b756927f65ee)
- [Introduction to deep learning for face recognition](https://machinelearningmastery.com/introduction-to-deep-learning-for-face-recognition/)
- [“Deep Face Recognition: A Survey"](https://arxiv.org/abs/1804.06655)
- [Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu: “SSD: Single Shot MultiBox Detector”, 2016; arXiv:1512.02325](https://arxiv.org/abs/1506.02640)
- [Dang Ha The Hien. A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
- [Howard Jeremy. Lesson 9: Deep Learning Part 2 2018 - Multi-object detection](https://docs.fast.ai/vision.models.unet.html#Dynamic-U-Net)
- [Evaluation of object detection systems](https://pythonawesome.com/most-popular-metrics-used-to-evaluate-object-detection-algorithms/)
