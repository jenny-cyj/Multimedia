import torch
import torchvision.models as models

# 下载并保存MobileNetV3预训练模型
def download_mobilenetv3():
    # 加载Large版本（ImageNet预训练）
    #mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    #torch.save(mobilenet_v3_large.state_dict(), "mobilenetv3_large_pretrained.pth")
    
    # 加载Small版本（ImageNet预训练）
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    torch.save(mobilenet_v3_small.state_dict(), "d:/cyj/multimedia/License-Plate-Recognition/src/retina_lpd/weights/mobilenetv3_small_pretrained.pth")
    
    print("预训练模型下载完成！")

if __name__ == "__main__":
    #download_mobilenetv3()
    model = models.mobilenet_v3_small(pretrained=True)
    # 打印所有特征层名称和输出尺寸
    x = torch.randn(1, 3, 640, 640)
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"features.{i}: 输出尺寸={x.shape}（步长={640//x.shape[2]}）")