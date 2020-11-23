# Introduction to Deep Learning #2
**Facial detection**

Use case: face detection on heritage images 

## Goals 
Annotation of visages on heritage material for information retrieval, quantitative analysis, etc.

![Face detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/visage.png)

<sup>[ark:/12148/btv1b10544068q](https://gallica.bnf.fr/ark:/12148/btv1b10544068q/f1)</sup>


## Hands-on session 

### OpenCV/dnn
The [OpenCV/dnn](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) module can be used to try some pretrained neural network models imported from frameworks as Caffe or Tensorflow.

This [Python 3 script](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/binder/faces-detection-with-dnn.py) uses dnn to call a ResNet SSD network (see [this post](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) or this [notebook](https://colab.research.google.com/github/dortmans/ml_notebooks/blob/master/face_detection.ipynb) for details). The model can be easily downloaded from the web.

The images of a Gallica document are first loaded thanks to the IIIF protocol. The detection then occurs and annotated images are generated, as well as the CSV data. 

Users may play with the confidence score value and look for the impact on the detection process. A basic filter on very large (and improbable) detections is implemented.

Display the Jupyter notebook with [nbviewer](https://nbviewer.jupyter.org/github/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/binder/faces-detection-with-dnn.ipynb).

Launch the notebook with Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/altomator/Introduction_to_Deep_Learning-2-Face_Detection/HEAD?filepath=https%3A%2F%2Fgithub.com%2Faltomator%2FIntroduction_to_Deep_Learning-2-Face_Detection%2Fblob%2Fmain%2Fbinder%2Ffaces-detection-with-dnn.ipynb)
## Use cases
- Information Retrieval: [GallicaPix](https://gallicapix.bnf.fr/rest?run=findIllustrations-app.xq&filter=1&start=1&action=first&module=1&locale=fr&similarity=&rValue=&gValue=&bValue=&corpus=1418&sourceTarget=&keyword=&kwTarget=&kwMode=&title=excelsior&author=&publisher=&fromDate=1915-01-01&toDate=1915-12-31&iptc=00&page=true&illTech=00&illFonction=00&illGenre=00&persType=faceM&classif1=&CBIR=*&classif2=&CS=0.5&operator=and&colName=00&size=31&density=26), men faces in 1915 issues of [_L'Excelsior_](https://gallica.bnf.fr/ark:/12148/cb32771891w/date.item)

- Data analysis of newspapers front page: faces, gender (Excelsior, 1910-1920)

![Front pages analysis: genders](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/faces-excelsior.jpg)

### Google Cloud Vision 

The Google Cloud Vision API may be used to perform face and gender detection. The Perl script described [here](https://github.com/altomator/Image_Retrieval) calls the API to perform visual recognition of content or human faces.

First, we have to build a JSON request:

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

Note: [IBM Watson](https://www.ibm.com/blogs/policy/facial-recognition-sunset-racial-justice-reforms/) no longer offers facial detection.


## Use cases
- Information Retrieval: [GallicaPix](https://gallicapix.bnf.fr/rest?run=findIllustrations-app.xq&filter=1&start=1&action=first&module=1&locale=fr&similarity=&rValue=&gValue=&bValue=&corpus=1418&sourceTarget=&keyword=&kwTarget=&kwMode=&title=excelsior&author=&publisher=&fromDate=1915-01-01&toDate=1915-12-31&iptc=00&page=true&illTech=00&illFonction=00&illGenre=00&persType=face&classif1=&CBIR=*&classif2=&CS=0.5&operator=and&colName=00&size=31&density=26), faces in the 1915 issues of [_L'Excelsior_](https://gallica.bnf.fr/ark:/12148/cb32771891w/date.item)

- Data analysis of newspapers front page: faces, gender (Excelsior, 1910-1920)

![Front pages analysis: genders](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/faces-excelsior.jpg)

- Averaging of faces: _[A Century of Portraits: A Visual Historical Record of American High School Yearbooks](https://arxiv.org/abs/1511.02575)_ (UC Berkeley)

![Averaging of faces](https://github.com/altomator/Introduction_to_Deep_Learning-2-Face_Detection/blob/main/images/averaging.jpg). Face detection is generally the first step of an averaging pipeline.

## Resources
- Face detection with [Google Cloud Vision](https://cloud.google.com/vision/docs/detecting-faces)
- [Kaggle faces dataset](https://www.kaggle.com/dataturks/face-detection-in-images)
- Face detection with [Detectron2](https://medium.com/@sidakw/face-detection-using-pytorch-b756927f65ee)

