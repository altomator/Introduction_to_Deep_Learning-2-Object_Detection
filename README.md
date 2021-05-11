# Introduction to Deep Learning #2
**Object detection**

Use case: object detection on heritage images 

## Goals 

Automatic annotation of objects on heritage images has uses in the field of information retrieval and digital humanities. Depending on the scenarios considered, this may involve obtaining a new source of textual metadata ("this image contains a *cat*, a *child* and a *sofa*") or locating every object classes of interest within the image (in this image, there is a *car* at position *x,y,w,h*).  

These goals can be satisfy with "out-of-the-box" services or customized solutions. 

![Object detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/objet.JPG)

<sup>*Vogue magazine*, French edition, 1922</sup>

## Hands-on session 

### "Out-of-the-box" services 

#### YOLO
[YOLO](https://pjreddie.com/darknet/yolo/) performs object detection on a 80 classes model. YOLO is well known to be fast and accurate.

This [Python 3 script](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/binder/object-detection-with-yolo.py) uses a YOLO v4 model that can be easily downloaded from the [web](https://github.com/AlexeyAB/darknet). The images of a Gallica document are first loaded thanks to the IIIF protocol. The detection then occurs and annotated images are generated, as well as the CSV data. 

**Display the Jupyter notebook with [nbviewer](https://nbviewer.jupyter.org/github/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/binder/object-detection-with-dnn.ipynb)**

**Launch the notebook with Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/altomator/Introduction_to_Deep_Learning-2-Object_Detection/c52d2bb24290811a95a9aee9f042a515eec03216)**

![Object detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/excelsior.jpg)

<sup>[ark:/12148/bpt6k46000341](https://gallica.bnf.fr/ark:/12148/bpt6k46000341/f1.item)</sup>


#### Google Cloud Vision, IBM Watson Visual Recognition, Clarifai...

These APIs may be used to perform objects detection. They are trained on huge datasets of thousands of object classes (like ImageNet) and may be useful for XXth century heritage content. These datasets are primarily aimed at photography, but the generalizability of artificial neural networks means that they can produce acceptable results for drawings and prints. 


The Perl script described [here](https://github.com/altomator/Image_Retrieval) calls the Google or IBM APIs. 

```
> perl toolbox.pl -CC datafile -google
```

_Note:_ IBM Watson Visual Recognition is discontinued. Existing instances are supported until 1 December 2021.

The API endpoint is simply called with a curl command sending  the request to the API as a JSON fragment including the image data and the features expected to be returned:

```
> curl --insecure  -v -s -H "Content-Type: application/json" https://vision.googleapis.com/v1/images:annotate?key=yourKey --data-binary @/tmp/request.jso
```

```
  ...
	"features": [
			{
				"type": "LABEL_DETECTION"
			},
			{
				"type": "CROP_HINTS"
			},
			{
				"type": "IMAGE_PROPERTIES"
			}
		], ...
```

See also with [Recipe](https://github.com/CENL-Network-Group-AI/Recipes/wiki/Images-Classification-Recipe) which makes use of IBM Watson API to call a  model previously trained with Watson Studio. 

**Cost, difficulties:**  Analyzing an image with such APIs costs a fraction of a cent per image.
Processing can be done entirely using the web platform or with a minimal coding load.


### Customized solutions

#### Transfert learning
Out-of-the box solutions use pretrained models. *Transfert learning* means to cut-off the last classification layer of these models and transfert the "model's knowledge" to a local problem, i.e. the set of images and objects one needs to work with.

*Transfer learning and domain adaptation refer to the situation where what has been learned in one setting … is exploited to improve generalization in another setting.* (*Deep Learning*, Ian Goodfellow and al., 2016)

Google Cloud Vision, IBM Watson Cloud Vision and other commercial framework can be used for training a specific object detector on custom data. Training can be done on the web platform (e.g. AutoML Vision) or using APIs. The trained models can then be deployed in the cloud or locally.

Same is true for YOLO, using a commercial web app like [Roboflow](https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/) or [local code](https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208). 

**Cost, difficulties:** Training means having annotated images available, which implies some preliminary work, and some computing power to train the model.
Depending on the context and the expected performance, tens or hundreds of annotated images may be required.

For commercial products, pricing is higher.  

#### Training from scratch
There is almost no reason to start from complete scratch, as the pretreained models  tend to generalize  well to other tasks, and will reduce overfitting then starting from  small dataset of images.



## Use cases

- **Information Retrieval:** the labels of the object classes are used as metadata and generally feed the library search engine.
  - [GallicaPix](https://github.com/altomator/Image_Retrieval) web app; 
  - [Digitens](https://www.univ-brest.fr/digitens/) project: indexing [wallpaper and textile design patterns](https://gallica.bnf.fr/blog/14032019/murs-de-papier-la-collection-de-papiers-peints-du-18eme-siecle-dans-gallica-historique-1?mode=desktop) from the [The National Archives](https://www.nationalarchives.gov.uk/) and the BnF
  - [Standford University Library: Clustering and Classification on all public images](https://sites.google.com/stanford.edu/sul-ai-studio/clustering-and-classification-on-all-public-images)
  - [Artificial intelligence @ the National Library of Norway](https://fr.slideshare.net/sconul/artificial-intelligence-the-national-library-of-norway-svein-arne-brygfjeld-national-library-of-norway)
  
[![Object detection on patterns: lines](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/tna.jpg)](https://gallicapix.bnf.fr/rest?run=findIllustrations-app.xq&locale=fr&action=first&start=1&corpus=PP&classif2=ligne&CS=0.5&operator=and&sourceTarget=&keyword=&module=0.5)

- **Digital Humanities:** in this context, labels and bounding boxes are used for retrieval or data mining scenarii.
  - [Helsinki Digital Humanities Hackathon 2019](https://www.helsinki.fi/en/helsinki-centre-for-digital-humanities/dhh-hackathon/helsinki-digital-humanities-hackathon-2019-dhh19): data analysis of [newspapers illustrated adds](https://github.com/altomator/Ads-data_mining) regarding transport means 

  - [Numapress](http://www.numapresse.org/) project: data analysis and information retrieval on the newspapers [movie section](http://www.numapresse.org/exploration/cinema_pages/query_illustration.php) (1900-1945): 

[![Object detection on newspapers illustrations](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/numapress.jpg)](http://www.numapresse.org/exploration/cinema_pages/query_illustration.php)

  - Telecom Paris-Tech, Nicolas Gonthier: [Weakly Supervised Object Detection in Artworks](https://wsoda.telecom-paristech.fr/)


## Resources
- [Object Detection in a Nutshell](https://goraft.tech/2020/05/01/object-detection-in-a-nutshell.html)


