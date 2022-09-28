#  python .\mask_detection\mask_data.py
#  只要呼叫 parse_dataset("$data路徑")
import os
import numpy as np
import cv2
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

class annotation:
    def __init__(self,file_path):     
        self.tree  =  ET.parse(file_path)
        self.ETobject = self.tree.getroot()

        self.filename = self.ETobject[1].text
        self.width =   self.ETobject[2][0].text
        self.height =   self.ETobject[2][1].text
        self.single_objects = []
        for n in  self.ETobject.iter('object'):
            self.class_name = n[0].text 
            self.xmin =   n[5][0].text 
            self.ymin =   n[5][1].text 
            self.xmax =   n[5][2].text 
            self.ymax =   n[5][3].text 
            self.single_objects.append([self.class_name,self.xmin,self.ymin, self.xmax, self.ymax])

class parse_dataset:
    def __init__(self,src_path):     
        self.dir_path = Path(src_path.image_path)/"annotations"
        self.all_object = []
        for dirname in os.listdir(self.dir_path):    
            file_name = self.dir_path/dirname
            single_annotation = annotation(file_name)
            self.all_object.append(single_annotation.single_objects)

if __name__ == '__main__':
    ## 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/mask", help="image path")
    args = parser.parse_args()

    dataset = parse_dataset(args)
    print(dataset.all_object[0])

                
        
    

