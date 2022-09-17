import onnxruntime as rt
import cv2
import numpy as np
import time

new_size = (40, 40)

image = cv2.imread('image_5.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_image = np.array([cv2.resize(image, new_size)])
x = resized_image.astype('float32') / 255 #(np.float32)
x = np.expand_dims(x, axis=-1)

providers = ['CPUExecutionProvider']
m = rt.InferenceSession('sequential_2.onnx', providers=providers)
t1 = time.time()
onnx_pred = m.run(['dense_7'], {"input": x})
print(time.time()-t1)
print(onnx_pred)
