# Installation

>The Installation step is referenced from [CenterNet](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md).
>
><br/>
>
>**Our experimental environment:** 
>
> Ubuntu 16.04.1,  Python 3.5.2, PyTorch 0.4.1, torchvision 0.2.1.
>
> During training we use 8 NVIDIA TITAN XP with cuda 9.0.

<br/>

1. Create a new conda environment and activate the environment.

   ~~~powershell
   conda create --name MOC python=3.5.2
   conda activate MOC
   ~~~
   
2. Install pytorch0.4.1:

   ~~~powershell
   conda install pytorch=0.4.1 torchvision -c pytorch
   ~~~

   Disable cudnn batch normalization(follow [CenterNet](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).

    ~~~powershell
   # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/MOC/lib/python3.5.2/site-packages/
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    ~~~
   
    For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. 

3. Clone this repo (${MOC_ROOT} is the path to clone):

   ~~~powershell
   git clone https://github.com/MCG-NJU/MOC-Detector.git ${MOC_ROOT}
   ~~~


4. Install the requirements

   ~~~powershell
   pip install -r pip-list.txt
   ~~~

5. Compile deformable convolutional in DLA backbone follow [CenterNet](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md).

   ~~~powershell
   cd ${MOC_ROOT}/src/network/DCNv2
   bash make.sh
   ~~~
