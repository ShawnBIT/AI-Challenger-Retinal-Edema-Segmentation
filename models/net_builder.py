from nets.UNet import UNet,UNet_Nested,UNet_Nested_dilated


def net_builder(name,pretrained_model=None,pretrained=False):
    if('resunet50' in name.lower()):
        net = resnet50_UNet(pretrained=pretrained)
    elif('resunet34' in name.lower()):
        net = resnet34_UNet(pretrained=pretrained)
    elif('deeplabv3' in name.lower()):
        net = DeepLabv3_plus(2, small=True,pretrained=pretrained)
    elif('drn' in name.lower()):
        net = DRNSeg('drn_d_105', 2, pretrained_model=None,pretrained=pretrained)
    elif name == 'unet':
        net = UNet(n_classes=4,feature_scale=4)
    elif name == 'unet_3':
        net = UNet_3(n_classes=2,feature_scale=4)
    elif name == 'unet_2':
        net = UNet_2(feature_scale=4)
    elif name == 'unet_3d':
        net = UNet_3D(n_classes=4,feature_scale=8)
    elif name == 'unet_old':
        net = UNet_old(n_classes=4,feature_scale=4)
    elif name == 'rnn_gru_unet':
        net = RNN_GRU_UNet2d()
    elif name == 'unet_aspp':
        net = UNet_aspp(feature_scale=4)
    elif name == 'unet_nonlocal':
        net = UNet_nonlocal(feature_scale=4)
    elif name == 'unet_nopooling':
        net = UNet_nopooling(feature_scale=4)
    elif name == 'unet_dilation':
        net = UNet_dilation(feature_scale=4)
    elif name == 'unet_k':
        net = UNet_k(feature_scale=4,k=2)
    elif name == 'unet_selu':
        net = UNet_SELU(feature_scale=4)
    elif name == 'unet_multi':
        net = UNet_M(feature_scale=4)
    elif name == 'unet_nested':
        net = UNet_Nested(feature_scale=4)
    elif name == 'unet_nested_1c':
        net = UNet_Nested_1c(feature_scale=4)
    elif name == 'unet_nested_superds':
        net = UNet_Nested_superds(feature_scale=4)
    elif name == 'unet_nested_res':
        net = UNet_Nested_Res(feature_scale=4)
    elif name == 'unet_nested_se':
        net = UNet_Nested_SE(feature_scale=4)
    elif name == 'unet_nested_dilated':
        net = UNet_Nested_dilated(feature_scale=4)
    elif name == 'unet_nested_dilated2':
        net = UNet_Nested_dilated2(feature_scale=4)
    elif name == 'unet_nested_dual_super':
        net = UNet_Nested_Dual_Super(feature_scale=4)
    elif name == 'unet_nested_botong':
        net = UNet_Nested_botong(feature_scale=4)
    elif name == 'unet_nested_botong_plus':
        net = UNet_Nested_botong_plus(feature_scale=4)
    elif name == 'unet_nested_dialted_botong':
        net = UNet_Nested_dialted_botong(feature_scale=4)
    
    
    

    elif name == 'denseaspp121':
        net = DenseASPP121()
    elif name == 'densenet':
        net = DenseNet121()
    elif name == 'resnet':
        net = ResNet18()
    elif name == 'vgg':
        net = VGG19()
    elif name == 'squeeze':
        net = squeezenet1_0()
    else:
        raise NameError("Unknow Model Name!")
    return net
