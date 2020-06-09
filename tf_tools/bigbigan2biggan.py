import os
import tensorflow as tf


def bikey_to_key(key):
    split = key.split('/')
    if len(split) > 1 and split[1] == 'Generator':
        key = os.path.join(*split[1:])

        if key.startswith('Generator/linear'):
            key = key.replace('Generator/', '')

        key = key.replace('Generator/BatchNorm/acc', 'Generator/ScaledCrossReplicaBNbn/acc') \
            .replace('Generator/BatchNorm/', 'Generator/ScaledCrossReplicaBN/') \
            .replace('BatchNorm_1/acc', 'CrossReplicaBN_1/acc') \
            .replace('BatchNorm/acc', 'CrossReplicaBN/acc') \
            .replace('BatchNorm_1', 'HyperBN_1') \
            .replace('BatchNorm', 'HyperBN') \
            .replace('GenZ', 'G_Z') \
            .replace('ema_0.9999', 'ema_b999900') \
            .replace('offset', 'beta') \
            .replace('scale', 'gamma')
        return key
    return None


def bigbigan2biggan(bigbigan, biggan):
    conversions = {}
    assigners = []
    biggan_keys = biggan.variable_map.keys()
    for bi_key, var in bigbigan.variable_map.items():
        key = bikey_to_key(bi_key)
        if key is not None:
            if key in biggan_keys:
                try:
                    assigners.append(biggan.variable_map[key].assign(var))
                    conversions[key] = bi_key
                except Exception:
                    continue
            else:
                print('skip: {} -> {}'.format(os.path.join(*bi_key.split('/')[1:]), key))

    bigbiGAN_linear = bigbigan.variable_map['GenEncWrapper/Generator/linear/w']
    bigbiGAN_linear = tf.repeat(bigbiGAN_linear, 1000, axis=0)
    bigbiGAN_linear_ema = bigbigan.variable_map['GenEncWrapper/Generator/linear/w/ema_0.9999']
    bigbiGAN_linear_ema = tf.repeat(bigbiGAN_linear_ema, 1000, axis=0)

    assigners.append([
        biggan.variable_map['linear/w'].assign(bigbiGAN_linear),
        biggan.variable_map['linear/w/ema_b999900'].assign(bigbiGAN_linear_ema)
    ])

    return assigners, conversions
