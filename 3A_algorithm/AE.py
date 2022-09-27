#3A控制是指自動曝光控制(AE)、自動聚焦控制(AF)、自動白平衡控制(AWB)
# https://blog.csdn.net/yongshengsilingsa/article/details/37744467
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


class AE:
    def __init__(self,src_path):        
        self.src = cv2.imread(src_path)

class AE:
    def __init__(self,src_path):        
        self.src = cv2.imread(src_path)







if __name__ == '__main__':
    print("This is openimage main")

    # 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/dog.jpg", help="image path")
    args = parser.parse_args()

