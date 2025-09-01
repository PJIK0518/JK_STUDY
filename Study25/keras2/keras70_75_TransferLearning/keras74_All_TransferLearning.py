from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG16(),
              VGG19(),
              ResNet50(), ResNet50V2(),
              ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
              DenseNet201(), DenseNet169(), DenseNet121(),
              InceptionV3(), InceptionResNetV2(),
              MobileNet(), MobileNetV2(),
              MobileNetV3Small(), MobileNetV3Large(),
              NASNetMobile(), NASNetLarge(),
              EfficientNetB0(), EfficientNetB1(), EfficientNetB2(),
              Xception()
              ]

for model in model_list:
          model.trainable = True
          
          print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')
          print('model_name :', model.name)
          print('total_wght :', len(model.weights))
          print('talbe_wght :', len(model.trainable_weights))
""" model.trainable = False
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : vgg16
total_wght : 32
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : vgg19
total_wght : 38
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet50
total_wght : 320
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet50v2
total_wght : 272
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet101
total_wght : 626
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet101v2
total_wght : 544
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet152
total_wght : 932
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet152v2
total_wght : 816
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : densenet201
total_wght : 1006
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : densenet169
total_wght : 846
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : densenet121
total_wght : 606
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : inception_v3
total_wght : 378
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : inception_resnet_v2
total_wght : 898
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : mobilenet_1.00_224
total_wght : 137
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : mobilenetv2_1.00_224
total_wght : 262
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : MobilenetV3small
total_wght : 210
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : MobilenetV3large
total_wght : 266
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : NASNet
total_wght : 1126
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : NASNet
total_wght : 1546
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : efficientnetb0
total_wght : 314
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : efficientnetb1
total_wght : 442
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : efficientnetb2
total_wght : 442
talbe_wght : 0
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : xception
total_wght : 236
talbe_wght : 0 """

""" model.trainable = True
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : vgg16
total_wght : 32
talbe_wght : 32
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : vgg19
total_wght : 38
talbe_wght : 38
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet50
total_wght : 320
talbe_wght : 214
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet50v2
total_wght : 272
talbe_wght : 174
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet101
total_wght : 626
talbe_wght : 418
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet101v2
total_wght : 544
talbe_wght : 344
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet152
total_wght : 932
talbe_wght : 622
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : resnet152v2
total_wght : 816
talbe_wght : 514
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : densenet201
total_wght : 1006
talbe_wght : 604
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : densenet169
total_wght : 846
talbe_wght : 508
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : densenet121
total_wght : 606
talbe_wght : 364
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : inception_v3
total_wght : 378
talbe_wght : 190
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : inception_resnet_v2
total_wght : 898
talbe_wght : 490
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : mobilenet_1.00_224
total_wght : 137
talbe_wght : 83
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : mobilenetv2_1.00_224
total_wght : 262
talbe_wght : 158
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : MobilenetV3small
total_wght : 210
talbe_wght : 142
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : MobilenetV3large
total_wght : 266
talbe_wght : 174
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : NASNet
total_wght : 1126
talbe_wght : 742
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : NASNet
total_wght : 1546
talbe_wght : 1018
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : efficientnetb0
total_wght : 314
talbe_wght : 213
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : efficientnetb1
total_wght : 442
talbe_wght : 301
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : efficientnetb2
total_wght : 442
talbe_wght : 301
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
model_name : xception
total_wght : 236
talbe_wght : 156
"""