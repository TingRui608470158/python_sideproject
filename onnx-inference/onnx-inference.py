
import torch
import onnxruntime as ort
import torch.onnx
import cv2
import numpy as np
import onnx

def test_onnx():
    img_paths = ["./dog.jpg"]
    sess = ort.InferenceSession("./deploy.onnx", None)
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    for i in range(len(sess.get_inputs())):
        print(sess.get_inputs()[i].name, sess.get_inputs()[i].shape)

    for i in range(len(sess.get_outputs())):
        print(sess.get_outputs()[i].name, sess.get_outputs()[i].shape)
    
    for img_path in img_paths:
        X_test = cv2.imread(img_path)
        X_test = cv2.resize(X_test, (640, 640))
        X_test = cv2.cvtColor(X_test, cv2.COLOR_BGR2RGB)
        print(X_test.shape)
        X_test = np.transpose(X_test,(2,0,1))
        print(X_test.shape)
        x = []
        x.append(X_test)

        pred_onx = sess.run([out_name], {input_name:x}) 
        # print(pred_onx)

if __name__ == '__main__':
    test_onnx()