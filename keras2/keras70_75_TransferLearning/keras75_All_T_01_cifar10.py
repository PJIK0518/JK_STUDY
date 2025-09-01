from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications import ResNet101
from keras.applications import DenseNet121
from keras.applications import MobileNetV2
from keras.applications import EfficientNetB0

SHAPE = (32,32,3)
model_list = [VGG16(include_top=False , input_shape=SHAPE),
              ResNet50(include_top=False , input_shape=SHAPE),
              ResNet101(include_top=False , input_shape=SHAPE),
              DenseNet121(include_top=False , input_shape=SHAPE),
              MobileNetV2(include_top=False , input_shape=SHAPE),
              EfficientNetB0(include_top=False , input_shape=SHAPE),
              ]