
Project file structure:

```
Project
│   README.md
│   *.py    
│
└───DIC-C2DH-HeLa
│   │
│   └───Sequence 1
│       │   t000.tif
│       │   t001.tif
│       │   ...
│  
│ 
└───Fluo-N2DL-HeLa
│   │
│   └───Sequence 1
│       │   t000.tif
│       │   t001.tif
|       │   ...
|
├── models
│   ├── marker
│   ├── marker.pkl
│   ├── mask
│   └── mask.pkl
│  
└───PhC-C2DL-PSC
│   │
│   └───Sequence 1
│       │   t000.tif
│       │   t001.tif
│       │   ...
│ 
└───Pickles
│   │   DIC-C2DH-HeLa-1.pickle
│   │   Fluo-N2DL-HeLa-1.pickle
│   │   ...
│
│
└───Tracking_division_output
│   │
│   └───DIC-C2DH-HeLa
│   │	│
│   │	└───Sequence 1
│	│			│	0.png
│	│			│	1.png
│	│			│	...
│	│
│   └───Fluo-N2DL-HeLa
│   │	│
│   │	└───Sequence 1
│	│			│	0.png
│	│			│	1.png
│	│			│	...
│	│
│   └───PhC-C2DL-PSC
│   	│
│   	└───Sequence 1
│				│	0.png
│				│	1.png
│				│	...
│
├── processed_data
│   ├── DIC-C2DH-HeLa
│   │   ├── Sequence 1
│   │   │   ...
│   │
│   ├── Fluo-N2DL-HeLa
│   │   ├── Sequence 1
│   │   │   ...
│   │
│   └── PhC-C2DL-PSC
│       ├── Sequence 1
│       │   ...
│   
└── sequences_ori
    ├── DIC-C2DH-HeLa
    │   ├── 01
    │   ├── 01_GT
    │   │   └── TRA
    │   │   ...
    │
    ├── Fluo-N2DL-HeLa
    │   ├── 01
    │   ├── 01_GT
    │   │   └── TRA
    │   │   ...
    │
    └── PhC-C2DL-PSC
        ├── 01
        ├── 01_GT
        │   └── TRA
        │   ...
```

1.Segmentation and Cell Detection
run:

python3 preprocessing.py
to prepare training and prediction data.

run:
python3 CNN.py --net [MASK_OR_MARKER] --lr [LEARNING_RATE] --batch [BATCH_SIZE] --epo [EPOCH]
to train the mask or marker U-Net model

run:
pyhton3 detect.py --pred [SEQ_COOR_SINGLE] --seq [SEQUENCE] --img [IMAGE] --mask [MASK_MODEL] --marker [MARKER_MODEL] --batch [BATCH_SIZE] --step [TIME_STEP]
to create pickle or mp4 files in the Pickles or mp4 directory or to show single segmentation image.


2.Tracking & Division Detect

python3 TrackByPickle.py [DATASET_NAME] [SEQ_NUMBER]

[DATASET_NAME] = d/p/f
[SEQ_NUMBER] = 1/2/3/4

for example:

python3 TrackByPickle.py d 1

will track DIC-C2DH-HeLa - Sequence 1, and the output file will be Tracking_division_output/Sequence 1/*.*

3.Cell Motion Analysis 

python3 Analysis.py [DATASET_NAME] [SEQ_NUMBER] [WAIT_TIME]

[DATASET_NAME] = d/p/f
[SEQ_NUMBER] = 1/2/3/4

operations:
Press N to display the next frame immediately
Press S to select a cell
Press D to cancel analysis of selected cells
Press Q to exit the program
Press SPACE or ENTER to continue

for example:

python3 Analysis.py d 1 500

it will display DIC-C2DH-HeLa - Sequence 1. Press S to select a cell and then press SPACE or ENTER to continue, 
then realted information will show in the terminal.