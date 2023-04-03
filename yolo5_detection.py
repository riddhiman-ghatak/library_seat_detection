import math
import torch
import numpy as np
import cv2
# import imutils
import sys
import time
import yaml
import os
import json
import warnings
warnings.filterwarnings("ignore")

# https://github.com/ultralytics/yolov5/issues/6460#issuecomment-1023914166
# https://github.com/ultralytics/yolov5/issues/36


# Loading Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.hub.load("yolov5", 'custom', path="yolov5/runs/train/exp/weights/yolo_weights.pt", source='local', force_reload=True)  # local repo


# Configuring Model
model.cpu()  # .cpu() ,or .cuda()
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
# (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.classes = [0, 56]
model.max_det = 20  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference


# Function to draw Centroids on the deteted objects and returns updated image
def draw_centroids_on_image(image, json_person, json_chair):
    data_person = json.loads(json_person)
    data_chair = json.loads(json_chair)  # Converting JSON array to Python List
    # Accessing each individual object and then getting its xmin, ymin, xmax and ymax to calculate its centroid
    for objects in data_chair:
        xmin_chair = objects["xmin"]
        ymin_chair = objects["ymin"]
        xmax_chair = objects["xmax"]
        ymax_chair = objects["ymax"]
        cx_chair = int((xmin_chair+xmax_chair)/2.0)
        cy_chair = int((ymin_chair+ymax_chair)/2.0)
        for object_person in data_person:
            xmin_person = object_person["xmin"]
            ymin_person = object_person["ymin"]
            xmax_person = object_person["xmax"]
            ymax_person = object_person["ymax"]
            cx_person = int((xmin_person+xmax_person)/2.0)
            cy_person = int((ymin_person+ymax_person)/2.0)
            chair_list = [cx_chair, cy_chair]
            person_list = [cx_person, cy_person]
            c1 = (xmin_chair, ymin_chair)
            c2 = (xmax_chair, ymax_chair)
            centroid_dist = math.dist(chair_list, person_list)
            print(centroid_dist)
            
            print(c1 , c2)
            print("......")
            
            if (centroid_dist >= 190):
                cv2.rectangle(image, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 150, 0), 2)
                cv2.putText(image, 'Empty', (int(c1[0]), int(c1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
                break
        #print("Object: ", data.index(objects))
        #print ("xmin", xmin)
        #print ("ymin", ymin)
        #print ("xmax", xmax)
        #print ("ymax", ymax)

        # Centroid Coordinates of detected object
        # cx = int((xmin+xmax)/2.0)
        # cy = int((ymin+ymax)/2.0)
        # print(cx,cy)

        # cv2.circle(output_image, (cx, cy), 2, (0, 0, 255), 2,
        #            cv2.FILLED)  # draw center dot on detected object
        # cv2.putText(output_image, str(str(cx)+" , "+str(cy)), (int(cx)-40, int(cy)+30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return (image)


if __name__ == "__main__":
    while (1):
        try:
            # Start reading camera feed (https://answers.opencv.org/question/227535/solvedassertion-error-in-video-capturing/))
            cap =  cv2.VideoCapture('library_video.mp4')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Now Place the base_plate_tool on the surface below the camera.
            while (1):
                _, frame = cap.read()
                #frame = undistortImage(frame)
                #cv2.imshow("Live" , frame)
                k = cv2.waitKey(5)
                if k == 27:  # exit by pressing Esc key
                    cv2.destroyAllWindows()
                    sys.exit()
                # if k == 13: #execute detection by pressing Enter key
                # OpenCV image (BGR to RGB)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Inference
                results = model(image, size=720)  # includes NMS

                # Results
                # results.print()  # .print() , .show(), .save(), .crop(), .pandas(), etc.
                # results.show()

                results.xyxy[0]  # im predictions (tensor)
                df = results.pandas().xyxy[0]  # im predictions (pandas)
                #      xmin    ymin    xmax   ymax  confidence  class    name
                # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
                # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
                # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
                df_chair = df[df['class'] == 56]
                df_person = df[df['class'] == 0]
                #Results in JSON
                json_results = results.pandas().xyxy[0].to_json(
                    orient="records")  # im predictions (JSON)
                # print(json_results)
                json_person = df_person.to_json(orient="records")
                json_chair = df_chair.to_json(orient="records")
                # results.render()  # updates results.imgs with boxes and labels
                # output_image = results.ims[0]  # output image after rendering
                output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw Centroids on the deteted objects and returns updated image
                output_im = draw_centroids_on_image(
                    output_image, json_person, json_chair)

                # Show the output image after rendering
                cv2.imshow("Output", output_im)
                # cv2.waitKey(1)

        except Exception as e:
            print("Error in Main Loop\n", e)
            cv2.destroyAllWindows()
            sys.exit()

    cv2.destroyAllWindows()

