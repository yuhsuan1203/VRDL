#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt


# In[ ]:


train_data = os.listdir('./train')
print(len(train_data))


# In[ ]:


img = plt.imread('train/1.png')
plt.imshow(img)
plt.show()


# In[ ]:


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs

def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file, 'r')
    print("All data length: " + str(f['/digitStruct/bbox'].shape[0]))
    all_rows = []
    print('image bounding box data construction starting...')
    bbox_df = pd.DataFrame([],columns=['height','img_name','label','left','top','width'])
#     print("Original bbox_df:")
#     print(bbox_df)
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df,pd.DataFrame.from_dict(row_dict, orient='columns')])
    bbox_df['bottom'] = bbox_df['top'] + bbox_df['height']
    bbox_df['right'] = bbox_df['left'] + bbox_df['width']
    print('finished image bounding box data construction...')
    return bbox_df

img_bbox_data = img_boundingbox_data_constructor('train/digitStruct.mat')


# In[ ]:


img_bbox_data


# In[ ]:


img_bbox_data_grouped = img_bbox_data.groupby('img_name')
print(len(img_bbox_data_grouped))
for i in range(len(img_bbox_data_grouped)):
    file_name = "label/" + str(i+1) + ".txt"
    fp = open(file_name, "w")
    img_name = str(i+1) + ".png"
    img = plt.imread('train/' + img_name)
    img_height = img.shape[0]
    img_width = img.shape[1]
    bbox_and_label_df = img_bbox_data_grouped.get_group(img_name)
    #print(bbox_and_label_df)
    #print(len(bbox_and_label_df))
    #print(type(bbox_and_label_df))
#     plt.imshow(img)
#     plt.show()
    #print(img_height)
    #print(img_width)
    for label_count in range(len(bbox_and_label_df)):
#         start_point = (int(bbox_and_label_df['left'][label_count]), int(bbox_and_label_df['top'][label_count]))
#         end_point = (int(bbox_and_label_df['right'][label_count]), int(bbox_and_label_df['bottom'][label_count])) 
#         color = (255, 255, 255)
#         thickness = 2
        label = int(bbox_and_label_df['label'][label_count])
        x_center = (int(bbox_and_label_df['left'][label_count]) + int(bbox_and_label_df['right'][label_count])) // 2
        y_center = (int(bbox_and_label_df['top'][label_count]) + int(bbox_and_label_df['bottom'][label_count])) // 2
        width = int(bbox_and_label_df['width'][label_count])
        height = int(bbox_and_label_df['height'][label_count])
        x_center_normalize = x_center / img_width
        y_center_normalize = y_center / img_height
        width_normalize = width / img_width
        height_normalize = height / img_height
#         img = cv2.rectangle(img, start_point, end_point, color, thickness)
#         print(img_name + ", " +
#               "label: {:d} ".format(label) + 
# #               "top: " + str(int(bbox_and_label_df['top'][label_count])) + " "
# #               "left: " + str(int(bbox_and_label_df['left'][label_count])) + " "
# #               "bottom: " + str(int(bbox_and_label_df['bottom'][label_count])) + " "
# #               "right: " + str(int(bbox_and_label_df['right'][label_count])) + " "
#               "x_center: " + str(x_center) + " " +
#               "y_center: " + str(y_center) + " " +
#               "width: " + str(width) + " " +
#               "height: " + str(height)
#              )
        fp.write(str(label - 1) + " " + str(x_center_normalize)+ " " + str(y_center_normalize)+ " " + 
                 str(width_normalize)+ " " + str(height_normalize))
        if label_count < len(bbox_and_label_df) - 1:
            fp.write("\n")
    fp.close()
print()


# In[ ]:


label_file_count = os.listdir('label/')
print(len(label_file_count))


# In[ ]:


img = plt.imread('train/1.png')
plt.imshow(img)
plt.show()

