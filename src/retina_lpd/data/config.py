# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_mnet_reduced = {
    'name': 'mobilenet0.25',
    'min_sizes': [[64, 128], [256, 512]],
    'steps': [16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_mnetv3 = {
    'name': 'mobilenetv3',  # 关键：指定使用MobileNetV3
    'min_sizes': [[16, 32], [64, 128], [256, 512]],  # 保持不变（适配车牌anchor尺寸）
    'steps': [8, 16, 32],  # 保持不变（与特征层步长对应）
    'variance': [0.1, 0.2],  # 保持不变（边框回归方差）
    'clip': False,  # 保持不变（是否裁剪预测框）
    'loc_weight': 2.0,  # 保持不变（定位损失权重）
    'gpu_train': True,  # 保持不变（是否使用GPU）
    'batch_size': 32,  # 可根据显存调整（MobileNetV3-small与V1 0.25体量接近，32通常可行）
    'ngpu': 4,  # 保持不变（根据实际GPU数量调整）
    'epoch': 100,  # 有预训练权重可保持100，无预训练建议改为120-150
    'decay1': 190,  # 学习率衰减点（若epoch=120，可改为100；原190适配100epoch足够）
    'decay2': 220,  # 学习率衰减点（若epoch=120，可改为110）
    'image_size': 640,  # 保持不变（输入尺寸为32的倍数，适配所有特征层）
    'pretrain': True,  # 已下载预训练权重，设为True
    'mode': 'small',  # 新增：明确使用MobileNetV3-small模式
    'width_mult': 1.0,  # 新增：网络宽度缩放因子（与打印的特征通道数匹配）
    'return_layers': {  # 关键：根据实际特征层选择（步长8/16/32）
        'features.2': '0',  # 步长8，输出通道24（FPN第一个输入）
        'features.4': '1',  # 步长16，输出通道40（FPN第二个输入）
        'features.9': '2'   # 步长32，输出通道96（FPN第三个输入）
    },
    'in_channel': 16,  # 关键：MobileNetV3初始输出通道（features.0的输出通道16，与结构一致）
    'out_channel': 64  # 保持不变（FPN输出通道，与原V1配置兼容）
}