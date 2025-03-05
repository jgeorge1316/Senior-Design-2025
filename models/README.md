# Models
These were the models generated using my scripts.

| model | Accuracy  |
| --- | --- |
| blah | blah |
**UPDATE THIS TABLE**

## [single_model0.1.1.pt](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/models/single_model0.1.1.pt)
Trained using 1500 for each target species, and ~220 for none. Took ~13 minutes to train on the Lambda AI Workstation. Accuracy results on all images (at the time it was trained). Please note that the training images are included in this count, so accuracy is skewed and not really a great statistic here.

| | narrowleaf cattail | none | phragmites | purple loosestrife |
| --- | --- | ---| --- | --- |
| % Accuracy | 98.94 | 1 | 97.59 | 99.87 |
| Total Count | 5947 | 223 | 9063 | 1504 |
| Correct Count | 5884 | 223 | 8845 | 1502 |
| Wrong Count | 63 | 0 | 218 | 2 |

## [single_model0.2.1.pt](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/models/single_model0.2.1.pt)
Trained using 1500 for each target species, and ~1300 for none. Took ~17 minutes to train on the Lambda AI Workstation. Accuracy results on all images (at the time it was trained). Please note that the training images are included in this count, so accuracy is skewed and not really a great statistic here.

| | narrowleaf cattail | none | phragmites | purple loosestrife |
| --- | --- | ---| --- | --- |
| % Accuracy | 98.44 | 1 | 97.76 | 99.93 |
| Total Count | 6104 | 1334 | 9063 | 1504 |
| Correct Count | 6009 | 1334 | 8860 | 1503 |
| Wrong Count | 95 | 0 | 203 | 1 |
