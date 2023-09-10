# This project is base on [AAML_MCU_TinyML_Lab](https://github.com/liuyy3364/AAML_MCU_TinyML_Lab/tree/master)

## Train the tf model with cifir10 
1. import ```train_with_cifar10.ipynb``` on colab
2. run the code and it will start training
2. after training , download ```resnet_v1_eembc.tflite``` & ```resnet_v1_eembc_quant.tflite```

## basic / cmsis
1. import ```Lab1_2_3/basic_compile.ipynb```(basic) or ```Lab3_4/cmsis_compile.ipynb``` (cmsis)
2. upload the model (cmsis can only run with quant model)
3. modify the ```model_path = ""``` to your model path
3. run the code and it will compile the project
4. after compile , download the .bin file

## inference on F767ZI 

1. connect the F767ZI with your computer
2. download [termite](https://www.compuphase.com/software_termite.htm)
3. open termite & set the baod rate ```115200```
4. copy the .bin file into F767ZI
5. see the output in termite


## inference time & arena size

| - | TFLM | TFLM + int8 Quant | TFLM + int8 Quant + cmsis | 
| - |------|-------------------|---------------------------|
|Time(us)|4,634,854|4,672,161|437,600|
|Arena Size(bytes) | 201696 |  54452 | 55108 |


