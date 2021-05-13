# COVID-19-Social-Distancing-Monitor
**Project Team:** Erich Choudhury, Connor Bowler, Alex Wirtz
**Supervisor:** Dr. Hamed Tabkhi
**Date:** May 13, 2021

![Showcase](./demo/Demo.gif)

This is the final project for Real Time AI at UNCC, Spring 2021. This implementation is based off of [Mikel Brostrom's YOLOv5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) project.

This project is a near real time COVID-19 social distance monitoring system the uses pre-existing libraries to help with estimating distances between objects and verifying social distancing guideline in order to create a safer, healthier public. It will also assist in visualizing if two people are too close together. 

To run, first run install the required libraries from requirements.txt with:

```
pip install -r requirements.txt
```

You will need to install python dependencies for YOLOv5, Deep Sort, TensorFlow, and Pytorch. This is described in the README for each project. Additionally, you will need to download the NYU Depth Dataset as described in the README for FCRN. Place it in the root directory and then run the following command from a static location.:

```
python predictdepth.py NYU_FCRN.ckpt img.jpg
```

Finally, run continuously with:

```
python covidtrackerdepth.py --source 0
```
