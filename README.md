# This project is base on [AAML_MCU_TinyML_Lab](https://github.com/liuyy3364/AAML_MCU_TinyML_Lab/tree/master)

# Lab 1~4 Images Classification

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


## inference time with Lab 1~4 Images Classification & arena size

| NUCLEO_F767ZI | TFLM | TFLM + int8 Quant | TFLM + int8 Quant + cmsis | 
| - |------|-------------------|---------------------------|
|Time(us)|4,634,854|4,672,161|437,600|
|Arena Size(bytes) | 201,696 |  54,452 | 55,108 |

| DISCO_F746NG | TFLM | TFLM + int8 Quant | TFLM + int8 Quant + cmsis | 
| - |------|-------------------|---------------------------|
|Time(us)|4,625,590 |5,747,635 | 428,668 |
|Arena Size(bytes) |201,696|54,452| 55,108  |

## Hw1 : Images Classification 
1. change the model architecture (add / remove some layer) in `train_with_cifar10.ipynb`
2. compare the new model's(add/remove) `Inference Time` , `Arena size` , `Model size` with origin one in non-quant / quant / quant + CMSIS condition

### you need to complete the following table
| Method | Time(us) | Model size | Arena Size |
|--------------------|----------|------------|------------|
| remove layer float | - | - | - |
| remove layer quant | - | - | - |
| remove layer quant + CMSIS | - | - | - |
| origin float | - | - | - |
| origin quant | - | - | - |
| origin quant + CMSIS | - | - | - |
| add layer float | - | - | - |
| add layer quant | - | - | - |
| add layer quant + CMSIS | - | - | - |



# hello world lab
## Train the tf model with Sin function & Compile to bin file
1. import ```hello_world_lab/compile.ipynb``` on colab
2. run the code and it will start training & compile
2. after alll , download ```hello_world.bin ``` 
## inference on F767ZI
1. connect the F767ZI with your computer
2. download termite
3. open termite & set the baod rate 9600
4. copy the .bin file into F767ZI
5. see the output in termite