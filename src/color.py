import cv2
import numpy as np


def detect_plate_color(image):
    # 转换为HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围（HSV范围）
    # 黄色: H(20-30), S(100-255), V(100-255)
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # 蓝色: H(100-130), S(100-255), V(100-255)
    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([130, 255, 255])

    # 绿色: H(40-80), S(100-255), V(100-255)
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])

    # 创建掩码
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # 统计每种颜色的像素数量
    yellow_count = cv2.countNonZero(yellow_mask)
    blue_count = cv2.countNonZero(blue_mask)
    green_count = cv2.countNonZero(green_mask)

    # 比较颜色数量，找出主色
    color_counts = {'Yellow': yellow_count, 'Blue': blue_count, 'Green': green_count}

    # 找出像素最多的颜色
    dominant_color = max(color_counts, key=color_counts.get)
    print(dominant_color)
    return dominant_color

def detect_text_color_by_rules(plate_image, plate_color, recognized_text):
    """
    基于车牌颜色规则推断文字颜色
    :param plate_image: 车牌图像（用于备用检测）
    :param plate_color: 检测到的车牌颜色
    :param recognized_text: 识别出的车牌文字
    :return: 推断的文字颜色
    """
    
    # 中国车牌颜色与文字颜色对应规则
    color_rules = {
        "Blue": "White",      # 蓝牌：白字
        "Yellow": "Black",    # 黄牌：黑字
        "Green": "Black",     # 绿牌：黑字（新能源）
        "White": "Black",     # 白牌：黑字（警车、军车等）
        "Black": "White"      # 黑牌：白字（外资企业）
    }
    
    # 基于车牌颜色的默认规则
    if plate_color in color_rules:
        text_color = color_rules[plate_color]
        print(f"基于规则推断：车牌{plate_color}色 → 文字{text_color}色")
        return text_color
    
    # 如果颜色规则中没有，使用备用检测方法
    print("未匹配到颜色规则，使用图像检测...")
    return detect_text_color_direct(plate_image)

def detect_text_color_direct(plate_image):
    """
    直接检测文字颜色（备用方法）
    """
    # 转换为HSV色彩空间
    hsv_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    
    # 定义颜色范围
    # 白色: 低饱和度，高亮度
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    
    # 黑色: 低亮度
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])
    
    # 创建掩码
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    black_mask = cv2.inRange(hsv_image, black_lower, black_upper)
    
    # 统计像素数量
    white_count = cv2.countNonZero(white_mask)
    black_count = cv2.countNonZero(black_mask)
    
    # 返回数量较多的颜色
    if white_count > black_count:
        return "White"
    else:
        return "Black"