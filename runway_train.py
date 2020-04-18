from preprocessing import preprocess_images
import tempfile
import time
import os
import sys

import runway
from runway.exceptions import * 
from copy import deepcopy

from flask import Flask, jsonify, send_file
from multiprocessing import Process, Queue
import json
import pickle

import dnnlib
from dnnlib.util import EasyDict
from dataset_tool import create_from_images
from metrics import metric_base
from metrics.metric_defaults import metric_defaults
from training.training_loop import training_loop

import training_sdk
from preprocessing import *

def create_training_config(
    tfrecord_dir,
    checkpoint_path,
    run_dir,
    G_beta1=0.0,
    G_beta2=0.99,
    D_beta1=0.0,
    D_beta2=0.99,
    mirror_augment=False,
    style_mixing_probability=0.9,
    generator_learning_rate=0.002,
    discriminator_learning_rate=0.004,
    weight_averaging_half_life=5,
    **kwargs
):


    train     = EasyDict() # Options for training loop.
    G         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    G_opt     = EasyDict(beta1=G_beta1, beta2=G_beta2, epsilon=1e-8)           # Options for generator optimizer.
    D_opt     = EasyDict(beta1=D_beta1, beta2=D_beta2, epsilon=1e-8)           # Options for discriminator optimizer.
    G_loss    = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')      # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='8k', layout='random')                           # Options for setup_snapshot_image_grid().
    submit_config        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().

    train.mirror_augment = mirror_augment

    sched.minibatch_gpu_base = 4
    # Uncomment to enable gradient accumulation
    sched.minibatch_size_base = 12

    train.G_smoothing_kimg = weight_averaging_half_life

    D_loss.gamma = 10
    metrics = [metric_defaults['fid5k']]
    desc = 'stylegan2'

    dataset_args = EasyDict(tfrecord_dir=tfrecord_dir)

    desc += '-1gpu'
    desc += '-config-f'

    sched.G_lrate_dict = {128: generator_learning_rate, 256: generator_learning_rate, 512: generator_learning_rate, 1024: generator_learning_rate}
    sched.D_lrate_dict = {128: discriminator_learning_rate, 256: discriminator_learning_rate, 512: discriminator_learning_rate, 1024: discriminator_learning_rate}

    kwargs = EasyDict(train)

    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config, resume_pkl=checkpoint_path)
    kwargs.submit_config = deepcopy(submit_config)
    kwargs.submit_config.run_dir = run_dir
    kwargs.submit_config.run_desc = desc
    return kwargs


def train(datasets, options, ctx):
    images_path = datasets['images']

    print('Reading expected image size from starting checkpoint')
    dnnlib.tflib.init_tf()
    _, _, Gs = pickle.load(open(options['checkpoint'], 'rb'))
    width, height = Gs.output_shape[::-1][:2]

    print('Resizing images')
    tmp_resized = tempfile.TemporaryDirectory()
    preprocess_images(images_path, tmp_resized.name, width, height, options['crop_method'])

    print('Converting dataset to TFRecord')
    tmp_dataset = tempfile.TemporaryDirectory()
    create_from_images(tmp_dataset.name, tmp_resized.name, 1)

    print('Creating training config')
    result_dir = runway.utils.generate_uuid()
    os.makedirs(result_dir)
    kwargs = create_training_config(tmp_dataset.name, options['checkpoint'], result_dir, **options)
    kwargs.update(max_steps=options['max_steps'])
    gen = training_loop(**kwargs)

    for (step, metrics, samples, checkpoints) in gen:
        ctx.step = step
        for k, v in metrics.items():
            ctx.metrics[k] = v
        for k, v in samples.items():
            ctx.samples.add(k, v)
        for k, v in checkpoints.items():
            ctx.checkpoints.add(k, v)
    

if __name__ == '__main__':
    training_sdk.run(train, {
        'datasets': {
            'images': runway.file(is_directory=True)
        },
        'options': {
            'checkpoint': runway.file(extension='.pkl', description='Model checkpoint to start training from'),
            'max_steps': runway.number(default=3000, min=1, max=25000, description='Total number of training iterations'),
            'generator_learning_rate': runway.number(default=0.002, min=0.001, max=0.05, description='Generator learning rate'),
            'discriminator_learning_rate': runway.number(default=0.002, min=0.001, max=0.05, description='Discriminator learning rate'),
            'weight_averaging_half_life': runway.number(default=3, min=0.1, max=10, step=0.1, description='Weight averaging half life'),
            'crop_method': runway.category(choices=['Center Crop', 'Random Crop', 'No Crop'], description='Crop method to use during preprocessing'),
        },
        'samples': {
            'generated_images_grid': runway.image,
            'interpolation_video': runway.file(extension='.mp4'),
            'interpolation_gif': runway.file(extension='.gif')
        },
        'metrics': {
            'FID': runway.number
        }
    })