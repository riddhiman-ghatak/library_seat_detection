# Library Seat Detection

This project is a system which can detect unoccupied seats in library from live CCTV footage.Used YOLOv5 object detection model and Opencv to make bounding boxes around the unocuupied seats.

## Dependencies
Python: 3.8.12
PyTorch: 1.10.2
OpenCV-Python:  4.5.5.62
YOLOv5

## Results


## Documentation

clone this project to your local machine using the following command

```shell
git clone https://github.com/riddhiman-ghatak/library_seat_detection.git
```
then move to project root directory

```shell
cd library_seat_detection
```

Then run the following command to see result

```shell
python yolo5_detection.py
```

## Algorithm

Using YOLOv5 object detection model firstly It detects the **person** and **chair** class from the CCTV footage. 
Then it stores the centroids of person and chair in separate variables.
After that it calculates the distances in between all persons and chairs, if it is more than a thresold limit, then it marks that chair as empty.
Then using Opencv it creates a bounding box around that chair. 






