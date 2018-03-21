# FaceDesc
To describe a face with gender, age, and emotion.

2018.03.14
Gender, age and emotion classification have been implemented by using the people's face image in reality. The three problems were treated as the same classification problem, whose input are people's face image and the output is the gender, age and emotion as class label classified by classifier. Emotion classification is not available because of lacking in training data. Only fer2013 dataset was available now. You can do this by yourself if you need this kind of function.

## Model
I have tried several network, such as Xeception, VGG16, Inception_resnet_v2 and MobileNet. For this task is not much important, I choose MobileNet as the network for tradeoff between accuracy and efficiency.

2018.03.21 Pretrained model has been uploaded, and new model with more training data is coming in few days.

| Model Scope | Model File Link | Update Date | Val. Acc. | Note |
| - | - | - | - | - |
| Gender | [gender.mobilenet.augment.18-0.00.model](https://pan.baidu.com/s/1svMqEQtSfpT2Nl3jVU0XlA) | 20180312 | 99.1% | password:doda | 
| Birth | [birth.mobilenet.07-0.81.model](https://pan.baidu.com/s/1vi92LYbC8toSrVsQCa9hwQ) | 20180312 | 70.3% | password:6w09 |

The validation dataset is 5% verse the rest.

Gender model:
| Softmax Index | Description | 
| ------------- | ----------- |
| 0 | Male |
| 1 | Female |

Birth Model:
| Softmax Index | Description | 
| ------------- | ----------- |
| 0 | 1940 |
| 1 | 1950 |
| 2 | 1960 |
| 3 | 1970 |
| 4 | 1980 |
| 5 | 1990 |
| 6 | 2000 |
| 7 | 2010 |

Note:
> 1. I choose 1940 as the start birth year label because lots of old folks looks like the same.
> 2. the birth year model does not have an accuracy as high as gender model so I choose 10 years as one interval of each birth year label.


## Dataset
I used about 400,000 images of Chinese people with currect age and gender label to train the deep neruel network. More data is collecting.

## Preprocess for training

You need prepare your image dataset by yourself. I used *align_dataset_mtcnn.py* and *facenet.py* in [davidsandberg](https://github.com/davidsandberg)'s [FaceNet](https://github.com/davidsandberg/facenet) to do face detection and face representation. You can also use other method to achieve the same goal. After face detection, you should arrange your dataset in the situation in which the face image in 'Male' directory and 'Female' directory if you want to train a gender model, or in birth directory if train a birth model.

For example:
>-- your dataset path <br/>
>---- Male<br/>
>------ 0001.PNG<br/>
>------ 0002.PNG<br/>
>    ...<br/>
>------ xxxx.PNG<br/>
>---- Female<br/>
>------ 0001.PNG<br/>
>------ 0002.PNG<br/>
>    ...<br/>
>------ xxxx.PNG<br/>

## Train
To implement a gender classifier by using keras. In gender_train.py, a mobilenet, which was provided by Keras team, was used. If you want to train your own model, just run *gender_train.py(or birth_predict.py)*. You can choose other network in network directory.

## Predict
Just load keras model file to predict. See more detail in *gender_predict.py(or birth_predict.py)* file.

## Hybrid model
One more proper way to describe a face is to build a hybrid model by treating it as a multi-label classification problem. Later I will do the expiriment.

## How to run this model.

First of all, you need preprocess your images of people's faces in preprocess if you want to run the pre-trained model, or the model trained my yourself. Secondly, just open *gender_predict.py* or *birth_predict.py*, change the path of model, and then run the script.
