from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    #print(dataloader)
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        if (batch_i + 1) % 50 == 0:
            print("batch " + str(batch_i + 1))
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        #current_time = time.time()
        #inference_time = datetime.timedelta(seconds=current_time - prev_time)
        #prev_time = current_time
        #print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    #cmap = plt.get_cmap("tab20b")
    #colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    #print(imgs)
    #print(img_detections)
    #print("\nSaving images:")
    # Iterate through images and save plot of detections
    li = [None] * 13068
    #print(li)
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        if (img_i + 1) % 50 == 0:
            print("img " + str(img_i + 1))
        #print("(%d) Image: '%s'" % (img_i, path))
        print(path)
        name = path.split('/')[3]
        num = name.split('.')[0]
        #print(name)
        #print(num)
        #li[int(num) - 1] = name
        #print(path)
        # Create plot
        img = np.array(Image.open(path))
        #plt.figure()
        #fig, ax = plt.subplots(1)
        #ax.imshow(img)
#         ans_dict = {}
#         ans_dict["bbox"] = [1, 2, 3, 4]
#         print(ans_dict)
        ans_str = ""
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            #print(type(detections.shape[0]))
            #unique_labels = detections[:, -1].cpu().unique()
            #n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds)
            
            #{"bbox": [       [7, 54, 23, 65], [7, 41, 23, 51], [7, 30, 23, 38]          ], 
            count = 0
            ans_str += "{\"bbox\": ["
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                y1_float = float(y1)
                x1_float = float(x1)
                y2_float = float(y2)
                x2_float = float(x2)
                #print(type(y1_float))
                ans_str += "[" + str(round(y1_float)) + ", " + str(round(x1_float)) + ", " + str(round(y2_float)) + ", " + str(round(x2_float)) + "]"
                #print("{\"bbox\": [%f, %f, %f, %f], " % (y1, x1, y2, x2))
                
                count += 1
                if count < detections.shape[0]:
                    ans_str += ", "
                box_w = x2 - x1
                box_h = y2 - y1

                #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                #bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                #ax.add_patch(bbox)
                # Add label
                #plt.text(
                #    x1,
                #    y1,
                #    s=classes[int(cls_pred)],
                #    color="white",
                #    verticalalignment="top",
                #    bbox={"color": color, "pad": 0},
                #)
            ans_str += "], "
            
            #"score": [0.9861333966255188, 0.9711938500404358, 0.9545950293540955], 
            count = 0
            ans_str += "\"score\": ["
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                ans_str += str(cls_conf.item())
                #print("\"score\": [%.5f]" % (cls_conf.item()))
                
                count += 1
                if count < detections.shape[0]:
                    ans_str += ", "
            ans_str += "], "
            
            #"label": [4, 4, 1]}
            count = 0
            ans_str += "\"label\": ["
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                ans_str += classes[int(cls_pred)]
                #print("\"label\": [%s]" % (classes[int(cls_pred)]))
                count += 1
                if count < detections.shape[0]:
                    ans_str += ", "
            ans_str += "]}"
            
        else:
            ans_str += "{\"bbox\": [], \"score\": [], \"label\": []}"
        li[int(num) - 1] = ans_str
        #print("===")
        # Save generated image with detections
        #plt.axis("off")
        #plt.gca().xaxis.set_major_locator(NullLocator())
        #plt.gca().yaxis.set_major_locator(NullLocator())
        #filename = path.split("/")[-1].split(".")[0]
        #plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        #plt.close()
    print(len(li))
    f = open("output/all_epoch45.json", "w")
    f.write("[")
    for i in range(len(li)):
        f.write(li[i])
        if i < len(li) - 1:
            f.write(", ")
    f.write("]")
    f.close()
    
    #### Write li to json file
