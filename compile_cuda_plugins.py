import dnnlib.tflib as tflib
from dnnlib.tflib.custom_ops import get_plugin

tflib.init_tf()

get_plugin('dnnlib/tflib/ops/fused_bias_act.cu')
get_plugin('dnnlib/tflib/ops/upfirdn_2d.cu')
