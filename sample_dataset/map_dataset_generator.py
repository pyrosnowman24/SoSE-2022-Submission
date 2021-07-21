import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("image1.png")
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_flat = gray_img1.flatten()
img1_goal = np.array((462,53))
img1_data = np.hstack((img1_flat,img1_goal))
height1,width1 = gray_img1.shape[0:2]

img2 = cv2.imread("image2.png")
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2_flat = gray_img2.flatten()
img2_goal = np.array((206,152))
img2_data = np.hstack((img2_flat,img2_goal))
height2,width2 = gray_img2.shape[0:2]

img3 = cv2.imread("image3.png")
gray_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img3_flat = gray_img3.flatten()
img3_goal = np.array((419,334))
img3_data = np.hstack((img3_flat,img3_goal))
height3,width3 = gray_img3.shape[0:2]

img4 = cv2.imread("image4.png")
gray_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img4_flat = gray_img4.flatten()
img4_goal = np.array((316,89))
img4_data = np.hstack((img4_flat,img4_goal))
height4,width4 = gray_img4.shape[0:2]

img5 = cv2.imread("image5.png")
gray_img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
img5_flat = gray_img5.flatten()
img5_goal = np.array((414,330))
img5_data = np.hstack((img5_flat,img5_goal))
height5,width5 = gray_img5.shape[0:2]

img6 = cv2.imread("image6.png")
gray_img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
img6_flat = gray_img6.flatten()
img6_goal = np.array((26,372))
img6_data = np.hstack((img6_flat,img6_goal))
height6,width6 = gray_img6.shape[0:2]

img7 = cv2.imread("image7.png")
gray_img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
img7_flat = gray_img7.flatten()
img7_goal = np.array((486,94))
img7_data = np.hstack((img7_flat,img7_goal))
height7,width7 = gray_img7.shape[0:2]

img8 = cv2.imread("image8.png")
gray_img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
img8_flat = gray_img8.flatten()
img8_goal = np.array((131,221))
img8_data = np.hstack((img8_flat,img8_goal))
height8,width8 = gray_img8.shape[0:2]

img9 = cv2.imread("image9.png")
gray_img9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
img9_flat = gray_img9.flatten()
img9_goal = np.array((40,68))
img9_data = np.hstack((img9_flat,img9_goal))
height9,width9 = gray_img9.shape[0:2]

img10 = cv2.imread("image10.png")
gray_img10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
img10_flat = gray_img10.flatten()
img10_goal = np.array((470,289))
img10_data = np.hstack((img10_flat,img10_goal))
height10,width10 = gray_img10.shape[0:2]

labels = np.arange(len(img1_flat)).astype('str')
labels = ['data' + s for s in labels]
labels.append("xpos")
labels.append("ypos")

data = np.vstack((img1_data,img2_data,img3_data,img4_data,img5_data,img6_data,img7_data,img8_data,img9_data,img10_data))
df = pd.DataFrame(data,columns = labels)

df.to_csv("~/Scripts/Map_dataset_script/test_dataset.csv")