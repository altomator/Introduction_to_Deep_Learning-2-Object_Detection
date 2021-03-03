# Introduction to Deep Learning #2
**Object detection**

Use case: object detection on heritage images 

## Goals 

Automatic annotation of objects on heritage images has uses in the field of information retrieval and digital humanities (quantitative analysis). 

![Object detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/objet.JPG)

<sup>[ark:/12148/bpt6k65413493](https://gallica.bnf.fr/ark:/12148/bpt6k65413493/f964.item)</sup>


## Hands-on session 

### YOLO
[YOLO](https://pjreddie.com/darknet/yolo/) performs object detection on a 80 classes model. YOLO is well known to be fast and accurate.

This [Python 3 script](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/binder/object-detection-with-yolo.py) uses a YOLO v4 model that can be easily downloaded from the [web](https://github.com/AlexeyAB/darknet). The images of a Gallica document are first loaded thanks to the IIIF protocol. The detection then occurs and annotated images are generated, as well as the CSV data. 

![Object detection on engraving material](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/excelsior.jpg)

<sup>[ark:/12148/bpt6k46000341](https://gallica.bnf.fr/ark:/12148/bpt6k46000341/f1.item)</sup>


### Google Cloud Vision, IBM Watson 

These APIs may be used to perform objects detection. The Perl script described [here](https://github.com/altomator/Image_Retrieval) calls the APIs. 

```
> perl toolbox.pl -CC datafile -google
```

The API endpoint is simply called with a curl command sending the local image files:

```

```


### Other approaches

Google Cloud Vision, IBM Watson Cloud Vision and other commercial framework can be used for training a specific object detector on custom data. 

Same is true for YOLO, using a commercial web app like [Roboflow](https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/) or [local code](https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208). 


## Use cases

- Information Retrieval: 
  - [GallicaPix](https://github.com/altomator/Image_Retrieval) web app; 
  - [Digitens](https://www.univ-brest.fr/digitens/) project: indexing [wallpaper and textile design patterns](https://gallica.bnf.fr/blog/14032019/murs-de-papier-la-collection-de-papiers-peints-du-18eme-siecle-dans-gallica-historique-1?mode=desktop) from the [The National Archives](https://www.nationalarchives.gov.uk/) and the BnF

[![Object detection on patterns: lines](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/tna.jpg)](https://gallicapix.bnf.fr/rest?run=findIllustrations-app.xq&locale=fr&action=first&start=1&corpus=PP&classif2=ligne&CS=0.5&operator=and&sourceTarget=&keyword=&module=0.5)

- Digital Humanities
  - [Helsinki Digital Humanities Hackathon 2019](https://www.helsinki.fi/en/helsinki-centre-for-digital-humanities/dhh-hackathon/helsinki-digital-humanities-hackathon-2019-dhh19): data analysis of [newspapers illustrated adds](https://github.com/altomator/Ads-data_mining) regarding transport means 

  - [Numapress](http://www.numapresse.org/) project: data analysis and information retrieval on the newspapers [movie section](http://www.numapresse.org/exploration/cinema_pages/query_illustration.php) (1900-1945): 

[![Object detection on newspapers illustrations](https://github.com/altomator/Introduction_to_Deep_Learning-2-Object_Detection/blob/main/images/numapress.jpg)](http://www.numapresse.org/exploration/cinema_pages/query_illustration.php)

  - Telecom Paris-Tech, Nicolas Gonthier: [Weakly Supervised Object Detection in Artworks](https://wsoda.telecom-paristech.fr/)



## Resources



