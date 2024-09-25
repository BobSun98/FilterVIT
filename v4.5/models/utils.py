def transfer_featuremap_to_att_shape(x):
    num_channel, h, w = x.shape[1], x.shape[2], x.shape[3]
    x = x.reshape(-1, num_channel, h * w).transpose(1, 2)
    return x


def transfer_att_shape_to_featuremap(x, h=None):
    if h is None:
        h = int((x.shape[1]) ** 0.5)
    w = h
    num_channel = x.shape[2]
    x = x.transpose(1, 2).reshape(-1, num_channel, h, w)
    return x
