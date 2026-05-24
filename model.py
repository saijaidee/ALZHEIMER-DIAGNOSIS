import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, Xception, MobileNetV3Large
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.xception import preprocess_input as xcep_pre
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mob_pre
from tensorflow.keras import layers, Model
import numpy as np
from types import SimpleNamespace

def build_efficientnet(input_shape=(224,224,3), n_classes=3):
    base = EfficientNetB0(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    x = base.output
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

def build_xception(input_shape=(224,224,3), n_classes=3):
    base = Xception(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    x = base.output
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

def build_mobilenetv3(input_shape=(224,224,3), n_classes=3):
    base = MobileNetV3Large(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    x = base.output
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

class Ensemble:
    def __init__(self, models):
        # models: dict name->tf.keras.Model
        self._models = models

    def predict(self, x, batch_size=8):
        # returns dict of name->probabilities (numpy arrays)
        preds = {}
        for name, m in self._models.items():
            preds[name] = m.predict(x, batch_size=batch_size)
        return preds

def load_ensemble():
    # NOTE: For speed, these are untrained model architectures.
    # Replace with model.load_weights(path) if you have weights saved.
    models = {
        'efficientnetb0': build_efficientnet(),
        'xception': build_xception(),
        'mobilenetv3': build_mobilenetv3()
    }
    return Ensemble(models)

def preprocess_image(pil_image, target_size=(224,224)):
    import numpy as np
    img = pil_image.resize(target_size)
    arr = np.array(img).astype('float32')
    # Basic normalization to [0,1]
    arr = arr / 255.0
    arr = np.expand_dims(arr, 0)
    return arr
