# Models
These were the models generated using my scripts. Mainly the [test_model_threaded.py](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/Test-Scripts/test_model_threaded.py)

| model | Accuracy  |
| --- | --- |
| blah | blah |
**UPDATE THIS TABLE**

## [single_model0.1.1.pt](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/models/single_model0.1.1.pt)
Trained [yolo11n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) using 1500 for each target species, and ~220 for none. Took ~13 minutes to train on the Lambda AI Workstation with 10 epochs. Accuracy results on all images (at the time it was trained). Please note that the training images are included in this count, so accuracy is skewed and not really a great statistic here.

| 0.1.1 | narrowleaf cattail | none | phragmites | purple loosestrife |
| --- | --- | ---| --- | --- |
| % Accuracy | 98.94 | 100 | 97.59 | 99.87 |
| Total Count | 5947 | 223 | 9063 | 1504 |
| Correct Count | 5884 | 223 | 8845 | 1502 |
| Wrong Count | 63 | 0 | 218 | 2 |

## [single_model0.2.1.pt](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/models/single_model0.2.1.pt)
Trained [yolo11n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) using 1500 for each target species, and ~1300 for none. Took ~17 minutes to train on the Lambda AI Workstation with 10 epochs. Accuracy results on all images (at the time it was trained). Please note that the training images are included in this count, so accuracy is skewed and not really a great statistic here.

| 0.2.1 | narrowleaf cattail | none | phragmites | purple loosestrife |
| --- | --- | ---| --- | --- |
| % Accuracy | 98.44 | 100 | 97.76 | 99.93 |
| Total Count | 6104 | 1334 | 9063 | 1504 |
| Correct Count | 6009 | 1334 | 8860 | 1503 |
| Wrong Count | 95 | 0 | 203 | 1 |

## [single_model0.3.1.pt](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/models/single_model0.3.1.pt)
Trained using 1500 for each target species, and ~1300 for none. Took ~19 minutes to train on the Lambda AI Workstation (other tasks were being done in parallel during training). Accuracy results are on all images for each class that were **NOT** used in training. That means that these results are much more relevant and not skewed by the training and validation images.

| 0.3.1 | narrowleaf cattail | none | phragmites | purple loosestrife |
| --- | --- | ---| --- | --- |
| % Accuracy | 99.40 | 100 | 97.46 | 100 |
| Total Count | 4829 | 200 | 7788 | 229 |
| Correct Count | 4800 | 200 | 7590 | 229 |
| Wrong Count | 29 | 0 | 198 | 0 |

Citation for YOLO11 model
```
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```