# Dataset

>For our experiment, we use UCF101-24 and  JHMDB dataset.
>
>Specifically,  we use UCF101v2 annotation from [here](https://github.com/gurkirt/corrected-UCF101-Annots).

<br/>

You can download the rgb frames , optical flow and ground truth annotations from our [Google drive](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct?usp=sharing).

(We get UCF101-24 v2 from [here](https://github.com/gurkirt/realtime-action-detection) and  JHMDB from [here](https://github.com/vkalogeiton/caffe/tree/act-detector))

```powershell
tar -zxvf UCF101_v2.tar.gz
tar -zxvf JHMDB.tar.gz
```

<br/>

Create soft link:

```powershell
ln -s $PATH_TO_DOWNLOAD/UCF101_v2   ${MOC_ROOT}/data/ucf24
ln -s $PATH_TO_DOWNLOAD/JHMDB   ${MOC_ROOT}/data/JHMDB
```

<br/>

Please make the data folder like this:

```shell
${MOC_ROOT}
|-- data
`-- |-- JHMDB
    `-- |-- Frames
    `-- |-- FlowBrox04
    `-- |-- JHMDB-GT.pkl
`-- |-- ucf24
    `-- |-- rgb-images
    `-- |-- brox-images
    `-- |-- UCF101v2-GT.pkl

```

<br/>

For more details about the format of the `pkl` files, please see [dataset_annotation.py](../src/datasets/dataset/dataset_annotation.py). 

