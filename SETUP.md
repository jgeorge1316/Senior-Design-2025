# How to setup and use the library:

## clone library and enter
```bash
git clone https://github.com/jgeorge1316/Senior-Design-2025
cd Senior-Design-2025
```

## install CUDA drivers
CUDA needs to be installed [here](https://developer.nvidia.com/cuda-downloads). You can check the installation of CUDA by running the following command:
```bash
nvidia-smi
```
If this command does not function in the CLI, then you may need to troubleshoot your CUDA drivers.

## create a venv (linux)
```bash
python3 -m venv myenv
```
Enter myenv:
```bash
source myenv/bin/activate
```

## install Ultralytics and necessary libraries
Option 1, use requirements.txt:
```bash
pip install -r requirements.txt
```

Option 2, install Ultralytics and allow pip to choose library versions.:
```bash
pip install Ultralytics
```
Please note that PyTorch may need to be installed manually to ensure that a version that matches the CUDA driver is installed. [Pytorch Install](https://pytorch.org/get-started/locally/)

## check installation of PyTorch and CUDA (optional)
You can check the installation of PyTorch and CUDA by running the [test_cuda_install.py](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/test_cuda_install.py) script.
```
python3 test_cuda_install.py
```

## test model on sample image
The [inference_test.py](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/inference_test.py) script will run inference on the image below.

<img src="narrowleaf_cattail-6-19-24-4746.JPG" alt="Narrowleaf cattail image" width="500">


Run in CLI:
```bash
python3 inference_test.py
```
Example console Output:
```bash
$ python3 inference_test.py 

image 1/1 /home/landon/Senior-Design/Joey_Training/narrowleaf_cattail-6-19-24-4746.JPG: 640x640 narrowleaf_cattail 0.92, phragmites 0.06, purple_loosestrife 0.02, none 0.00, 3.2ms
Speed: 82.9ms preprocess, 3.2ms inference, 0.0ms postprocess per image at shape (1, 3, 640, 640)
```
Output Image should look like this:
<img src="narrowleaf_cattail-6-19-24-4746_results.jpg" alt="Narrowleaf cattail image with inference results" width="500">
