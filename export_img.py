from PIL import Image
import argparse
import numpy as np
import subprocess

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str, help='image path' , default="data/images/dogs.jpg")

    opt = parser.parse_args()
    return opt

def main(opt):
    
    img = Image.open(opt['img']).resize((32,32))
    img = np. array(img)
    bin_img = ''.join(opt['img'].split('.')[:-1]) + ".bin"
    img.tofile(bin_img)
    cc_img = ''.join(opt['img'].split('.')[:-1]) + ".cc"

    subprocess.run(f"xxd -i {bin_img} > {cc_img}" , shell=True , timeout=5)


if __name__ == '__main__':
    opt = parse_opt()
    main(vars(opt))


