#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import numpy as np
import paddle.fluid as fluid

#加载自定义文件
import models
from attack_code.attack.attack_pp import FGSM, PGD
from attack_code.utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments

#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim',        int,   120,                  "Class number.")
add_arg('shape',            str,   "3,224,224",          "output image shape")
add_arg('input',            str,   "../input_image/",     "Input directory with images")
add_arg('output',           str,   "./output_image/",    "Output directory with images")

args = parser.parse_args()
print_arguments(args)

######Init args
image_shape = [int(m) for m in args.shape.split(",")]
class_dim=args.class_dim
input_dir = args.input
output_dir = args.output
model_name="MobileNetV2"
# model_name="ResNeXt50_32x4d"
pretrained_model="./models_parameters/MobileNetV2"
# pretrained_model="./models_parameters/ResNeXt50_32x4d"
val_list = 'val_list.txt'
use_gpu=True

######Attack graph
adv_program=fluid.Program()
#完成初始化s
with fluid.program_guard(adv_program):
    input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    #设置为可以计算梯度
    input_layer.stop_gradient=False

    # model definition
    model = models.__dict__[model_name]()
    out_logits = model.net(input=input_layer, class_dim=class_dim)
    out = fluid.layers.softmax(out_logits)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #记载模型参数
    fluid.io.load_persistables(exe, pretrained_model)

#设置adv_program的BN层状态
init_prog(adv_program)

#创建测试用评估模式
eval_program = adv_program.clone(for_test=True)

#定义梯度
with fluid.program_guard(adv_program):
    label = fluid.layers.data(name="label", shape=[1] ,dtype='int64')
    loss = fluid.layers.cross_entropy(input=out, label=label)
    gradients = fluid.backward.gradients(targets=loss, inputs=[input_layer])[0]

######Inference
def inference(img):
    fetch_list = [out.name]

    result = exe.run(eval_program,
                     fetch_list=fetch_list,
                     feed={ 'image':img })
    result = result[0][0]
    pred_label = np.argmax(result)
    pred_score = result[pred_label].copy()
    return pred_label, pred_score

######FGSM attack
#untarget attack
def attack_nontarget_by_FGSM(img, src_label):
    pred_label = src_label

    step = 8.0/256.0
    eps = 32.0/256.0
    while pred_label == src_label:
        #生成对抗样本
        adv=FGSM(adv_program=adv_program,eval_program=eval_program,gradients=gradients,o=img,
                 input_layer=input_layer,output_layer=out,step_size=step,epsilon=eps,
                 isTarget=False,target_label=0,use_gpu=use_gpu)

        pred_label, pred_score = inference(adv)
        step *= 2
        if step > eps:
            break

    print("Test-score: {0}, class {1}".format(pred_score, pred_label))

    adv_img=tensor2img(adv)
    return adv_img

####### Main #######
def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files

def gen_adv():
    mse = 0
    original_files = get_original_file(input_dir + val_list)

    for filename, label in original_files:
        img_path = input_dir + filename
        print("Image: {0} ".format(img_path))
        img=process_img(img_path)
        adv_img = attack_nontarget_by_FGSM(img, label)
        image_name, image_ext = filename.split('.')
        ##Save adversarial image(.png)
        save_adv_image(adv_img, output_dir+image_name+'.png')

        org_img = tensor2img(img)
        score = calc_mse(org_img, adv_img)
        mse += score
    print("ADV {} files, AVG MSE: {} ".format(len(original_files), mse/len(original_files)))

def main():
    gen_adv()


if __name__ == '__main__':
    main()
