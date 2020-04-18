import pickle
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import dnnlib
import os

#projector libs
import dataset_tool
import run_projector
import projector
import training.dataset as dataset
import training.misc as misc

import runway

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


def get_projected_real_images(dataset_name, data_dir, num_images, num_snapshots,num_steps, _Gs):
    proj = projector.Projector()
    proj.set_network(_Gs)
    proj.num_steps = num_steps

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == _Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        out = run_projector.get_projected_images(proj, targets=images, num_snapshots=num_snapshots)

    return out



@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    tflib.init_tf()
    with open(opts['checkpoint'], 'rb') as file:
        _, _, Gs = pickle.load(file, encoding='latin1')
    return Gs




#@runway.command('project', inputs=project_inputs, outputs={'images': runway.array(item_type=runway.image, max_length=10)})
@runway.command('project', inputs={'projectionImage': runway.image( min_width=1024, min_height=1024, max_width=1024, max_height=1024)}, outputs={'image': runway.image})
def project(model, inputs):
    im = inputs['projectionImage']
    if not os.path.exists('./projection'):
        os.mkdir('./projection')
    if not os.path.exists('./projection/imgs'):
        os.mkdir ('./projection/imgs')
    if not os.path.exists('./projection/records'):
        os.mkdir('./projection/records')
    if not os.path.exists('./projection/out'):
        os.mkdir('./projection/out')

    if os.path.isfile('./projection/imgs/project.png'):
        os.remove('./projection/imgs/project.png')

    for f in os.listdir('./projection/records/'):
        if os.path.isfile(os.path.join('./projection/records/', f)):
            os.remove (os.path.join('./projection/records/', f))

    im.save('./projection/imgs/project.png')

    dataset_tool.create_from_images("./projection/records/", "./projection/imgs/", True)

    output = get_projected_real_images("records","./projection/",1,10, 100, model)

    return output[9]


if __name__ == '__main__':
    runway.run(model_options={ 'checkpoint': 'stylegan2-ffhq-config-f.pkl' })
        #runway.run(host='localhost', port=8888, debug=True, model_options={'checkpoint': './vox-cpk.pth.tar'})
