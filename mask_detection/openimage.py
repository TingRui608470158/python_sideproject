#記得在 python_sideproject 下執行$  python .\test1_openimage\openimage.py
#Class img功能:定義一個影像及相關操作(讀取，顯示，縮放)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

#使用時注意，定義物件時就代表已將img讀取至class裡，function裡的功能做都只對這個img做
class single_img:
    def __init__(self,src_path):        # 將影像讀取至這個物件
        self.src = cv2.imread(src_path)

    def show(self,windows_name="test"):        # 顯示
        cv2.imshow(windows_name, self.src)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def resize(self,wh_size):           # 縮放
        self.src = cv2.resize(self.src, wh_size)
    
    def xflip(self):                    # x軸翻轉
        self.src =cv2.flip(self.src, 1)

    def yflip(self):                    # y軸翻轉
        self.src =cv2.flip(self.src, 0)
    
    def HistEqual(self,show):            # 直方圖均衡(增強對比度)
        # calculate histogram
        hist=[]
        for c in range(3):
            hist.append(cv2.calcHist([self.src],[c],None,[256],[0, 256]))
        color = ['b','g','r']
        if show :   #是否要顯示直方圖           
            for i in range(len(hist)):
                plt.plot(hist[i], color=color[i])
            plt.show()

        #計算pdf
        merge_img = []
        for c, channel_img in enumerate(cv2.split(self.src)):
            # print(channel_img.size)
            pdf = hist[c]/channel_img.size
            cdf = pdf.cumsum()
            # if show :   #是否要顯示直方圖           
            #         plt.plot(cdf[c], color=color[i])
            # plt.show()
            equ_value = np.around(cdf * 255).astype('uint8')
            result = equ_value[channel_img]
            merge_img.append(result)
            # print(pdf)
        cv2.imshow("HistEqual",cv2.merge(merge_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return(cv2.merge(merge_img))






if __name__ == '__main__':
    print("This is openimage main")

    # 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/dog.jpg", help="image path")
    args = parser.parse_args()

    # 讀取img及操作
    my_img = single_img(args.image_path)
    my_img.resize((240,240))
    # my_img.xflip()
    my_img.show("mywindows")
    # my_img.yflip()
    # my_img.show("mywindows")
    histeq = my_img.HistEqual(False)

    
    
    ############################################################
    #待做事項
    #1. 3A演算法
    #2. 攝影機內外參數相關
    #
    #
    #
    #
    #           
    ############################################################