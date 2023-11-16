# The TFLM [hello world example](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/README.md) on [NUCLEO_F767ZI](https://www.st.com/en/evaluation-tools/nucleo-f767zi.html)

## run `train_and_compile.ipynb` on colab
1. modify the  `hyperparameter` block
```py
hidden_layer = 0
hidden_dim = 0
epoch = 0
```
2. (optional) to use CMSIS need modify the `run script` part with `!bash setup.sh -c`
3. dowload the bin file in `BUILD/NUCLEO_F767ZI/GCC_ARM/hello_world.bin`
## inference on NUCLEO_F767ZI
1. connect the F767ZI with your computer
2. download [termite](https://www.compuphase.com/software_termite.htm)
3. open termite & set the baod rate ```9600```
4. copy the .bin file into F767ZI
5. see the output in termite