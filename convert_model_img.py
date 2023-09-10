from PIL import Image
import argparse
import numpy as np
import subprocess
from pathlib import Path

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str, help='image path' , default="data/images/dog.jpg")
    parser.add_argument('--quant', action="store_true")
    parser.add_argument('--model', type=str, help='model.tflite path')
    parser.add_argument('--project', type=str, help='project path')

    opt = parser.parse_args()
    return opt

def cover_img(opt):
    project_path = Path(opt['project'])
    img = Image.open(opt['img']).resize((32,32))
    if opt['quant']:
        in_type = "uint8_t"
        out_type = "uint8_t"
        quant = "true"
        op_num = 8
        img = np.array(img)
    else:
        in_type = "float"
        out_type = "float"
        quant = "false"
        op_num = 7
        img = np.array(img).astype(np.float32)
    
    with open( project_path / "img.cc" , 'w')  as f:
        f.write("#include <stdint.h>\n")
        f.write(in_type + " img_bin[] = {\n")
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
        f.write("unsigned int img_bin_len = 3072;\n")

    with open(project_path/ "img.h" , 'w')  as f:
        f.write("#ifndef IMG_DATA_H_\n")
        f.write("#define IMG_DATA_H_\n")
        f.write(f"#define IN_IO_TYPE  {in_type}\n")
        f.write(f"#define OUT_IO_TYPE {out_type}\n")
        f.write(f"#define QUANT_MODEL {quant}\n")
        f.write(f"#define OP_NUM {op_num}\n")
        f.write(f"   extern const {in_type} img_bin[];\n")
        f.write("   extern const unsigned int img_bin_len;\n")
        f.write("#endif  // IMG_DATA_H_\n")

def cover_model(opt):
    project_path = Path(opt['project'])
    subprocess.run(["/bin/bash", "cover_model.sh" , opt['model'] , str(project_path)] , timeout=5 )
    


def main(opt):
    cover_img(opt)
    cover_model(opt)



if __name__ == '__main__':
    opt = parse_opt()
    main(vars(opt))


