from PIL import Image
import argparse
import numpy as np
import subprocess

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str, help='image path' , default="data/images/dog.jpg")

    opt = parser.parse_args()
    return opt


def main(opt):
    
    img = Image.open(opt['img']).resize((32,32))
    img = np.array(img).astype(np.float32)
    
    with open("img.cc" , 'w')  as f:
        f.write("float img_bin[] = {\n")
        idx = 0
        for w in img:
            for h in w:
                for c in h:
                    idx+=1
                    if idx != 32*32*3:
                        f.write(f"{c}, ")
                    else:
                        f.write(f"{c}")
            f.write("\n")

        f.write("};\n")
        f.write("unsigned int img_bin_len = 3072;")



    # bin_img = ''.join(opt['img'].split('.')[:-1]) + ".bin"
    # img.tofile(bin_img)
    # cc_img = ''.join(opt['img'].split('.')[:-1]) + ".cc"

    # subprocess.run(f"xxd -i {bin_img} > {cc_img}" , shell=True , timeout=5)


if __name__ == '__main__':
    opt = parse_opt()
    main(vars(opt))


