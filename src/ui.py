from tkinter.filedialog import *
from tkinter import *
# from tkinter.ttk import *
from PIL import Image, ImageTk
from tkinter.font import Font
from extract import extract_plate_yolo
from retina import extract_retina
from split_char import split_char
from color import detect_plate_color, detect_text_color_by_rules
import os
import cv2

COLOR_MAP = {
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "Green": "#008000",
}

COLOR_MAP_TEXT = {
    "Black": "#000000",      # 黑色文字
    "White": "#FFFFFF",      # 白色文字  
    "Yellow": "#FFFF00",     # 黄色文字
    "Red": "#FF0000",        # 红色文字
    "Blue": "#0000FF",       # 蓝色文字
    "Green": "#008000",      # 绿色文字
}

class WinGUI(Tk):
    # 初始化窗口 设置各个部件的位置 大小 颜色 字体 对应事件
    # label用于显示图片&文字
    # button用于响应用户的点击 运行函数
    # frame用于放置部件 是label和button的载体
    def __init__(self):
        super().__init__()
        self.__win()
        # 定义颜色方案
        self.colors = {
            "primary": "#2c3e50",      # 主色调 - 深蓝灰
            "secondary": "#3498db",    # 次要色 - 亮蓝
            "accent": "#e74c3c",       # 强调色 - 红色
            "background": "#ecf0f1",   # 背景色 - 浅灰
            "text_dark": "#2c3e50",    # 深色文字
            "text_light": "#ffffff",   # 浅色文字
            "success": "#27ae60",      # 成功色 - 绿色
            "warning": "#f39c12"       # 警告色 - 橙色
        }
        
        # 配置主窗口背景
        self.configure(bg=self.colors["background"])
        self.pyt = None # 原始图片 类型为ImageTk.PhotoImage
        self.locate = None # 车牌定位图 类型为ImageTk.PhotoImage
        self.color = ""  # 车牌颜色 类型为str
        self.result = "" # 识别结果 类型为str
        self.pic_path = "" # 原图路径 类型为str
        self.tk_frame_frame_origin = self.__tk_frame_frame_origin(self)
        self.tk_label_label_origin = self.__tk_label_label_origin(self.tk_frame_frame_origin)
        self.tk_label_label_name = self.__tk_label_label_name(self)
        #self.tk_frame_frame_command = self.__tk_frame_frame_command(self)
        #self.tk_label_label_name_command = self.__tk_label_label_name_command( self.tk_frame_frame_command) 
        self.tk_button_button_choose = self.__tk_button_button_choose(self) 
        self.tk_button_button_apply = self.__tk_button_button_apply(self) 
        self.tk_frame_frame_locate = self.__tk_frame_frame_locate(self)
        self.tk_label_label_locate = self.__tk_label_label_locate( self.tk_frame_frame_locate) 
        self.tk_frame_frame_result = self.__tk_frame_frame_result(self)
        self.tk_label_label_result = self.__tk_label_label_result( self.tk_frame_frame_result) 
        self.tk_label_label_name_locate = self.__tk_label_label_name_locate(self)
        self.tk_label_label_name_result = self.__tk_label_label_name_result(self)

        # 设置字体和字号
        font = Font(family="Microsoft YaHei", size=24, weight="bold")
        self.tk_label_label_name.configure(font=font)
        #font = Font(family="STSong", size=15, weight="bold")
        #self.tk_label_label_name_command.configure(font=font)
        #self.tk_label_label_name_locate.configure(font = font)
        #self.tk_label_label_name_result.configure(font = font)
        font = Font(family="SimHei", size=30, weight="bold")
        self.tk_label_label_result.configure(font = font)

        # 添加状态标签
        self.tk_label_status = self.__tk_label_status(self)

    def __tk_label_status(self, parent):
        label = Label(parent,
                    text="请选择图片开始识别",
                    anchor="center",
                    bg=self.colors["background"],
                    fg=self.colors["text_dark"],
                    font=Font(family="Microsoft YaHei", size=10))
        label.place(x=740, y=630, width=242, height=25)
        return label

    # 设置窗口大小 标题
    def __win(self):
        self.title("智能车牌识别系统")
        width = 1100  # 稍微加宽窗口
        height = 750
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        
        # 设置窗口图标（如果有的话）
        # self.iconbitmap("license_plate_icon.ico")
        
        self.resizable(width=False, height=False)

    
    # 连接识别函数
    def connection(self, image_path):
        """
        调用主程序完成车牌定位、字符分割与识别
        返回定位图路径、识别结果、车牌颜色、文字颜色
        """
        try:
            origin_image = cv2.imread(image_path)
            plate_image = extract_plate_yolo(origin_image)

            # 1. 提取原始图像文件名，拼接临时文件路径
            filename = os.path.basename(image_path)
            tmp_dir = ".tmp"
            path = os.path.join(tmp_dir, filename)

            # 2. 创建临时文件夹
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                print(f"临时文件夹 '{tmp_dir}' 不存在，已自动创建")

            cv2.imwrite(path, plate_image if plate_image is not None else origin_image)

            plate_retina = extract_retina(image_path if plate_image is None else path, path)
            chars = split_char(image_path if not plate_retina else path)

            if not chars:
                return path, "未检测到", "#000000", "#ffffff"
            
            # 检测车牌颜色
            plate_color = detect_plate_color(cv2.imread(path))
            
            # 基于规则推断文字颜色
            text_color = detect_text_color_by_rules(cv2.imread(path), plate_color, chars)
            
            # 处理车牌格式
            if chars[2] != '·':
                new_characters = chars[0:2]
                new_characters += '·'
                new_characters += chars[2:]
                chars = new_characters
            
            # 新能源车牌特殊处理
            if len(chars) == 9:
                plate_color = "Green"
                text_color = "Black"  # 新能源车牌一定是黑字

            # 将颜色字符串转换为颜色编码
            color_code = COLOR_MAP.get(plate_color, "#FFFFFF")
            text_color_code = COLOR_MAP_TEXT.get(text_color, "#000000")

            #print(f"最终结果: 车牌{plate_color}色, 文字{text_color}色, 号码:{chars}")
            
            return path, chars, color_code, text_color_code
            
        except Exception as e:
            print("识别错误:", e)
            return path, chars, color_code, text_color_code
    # “识别图片”按键对应的函数 作用是处理图片得到车牌并显示定位和结果
    def work(self):
        """
        点击识别图片按钮后的处理逻辑
        """
        if not self.pic_path:
            self.tk_label_status.configure(text="请先选择图片", fg=self.colors["accent"])
            return
        
        # 更新状态
        self.tk_label_status.configure(text="正在识别中...", fg=self.colors["warning"])
        self.update()  # 强制更新界面

        try:
            # 调用识别逻辑，现在返回四个值
            lp_img, self.result, self.color, self.text_color = self.connection(self.pic_path)
            
            # 更新车牌定位图
            image = Image.open(lp_img)
            label_width = self.tk_label_label_locate.winfo_width()
            label_height = self.tk_label_label_locate.winfo_height()
            image_width, image_height = image.size
            scale_ratio = min(label_width / image_width, label_height / image_height)
            new_width = int(image_width * scale_ratio)
            new_height = int(image_height * scale_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            self.locate = ImageTk.PhotoImage(image)
            self.tk_label_label_locate.configure(image=self.locate)

            # 更新识别结果
            self.tk_label_label_result.configure(
                text=self.result,
                background=self.color,      # 车牌背景颜色
                foreground=self.text_color  # 车牌文字颜色
            )
            
            # 根据识别结果更新状态
            if self.result and self.result != "未检测到":
                self.tk_label_status.configure(text="识别完成！", fg=self.colors["success"])
            else:
                self.tk_label_status.configure(text="未检测到车牌", fg=self.colors["accent"])
            
        except Exception as e:
            self.tk_label_status.configure(text=f"识别出错: {str(e)}", fg=self.colors["accent"])
            print("识别错误:", e)

    # “选择图片”按键对应的函数 作用是选择图片并显示原图
    def choose_pic(self):
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg"),("png图片", "*.png")])
        if self.pic_path:
            image = Image.open(self.pic_path)
            # 获取Label的尺寸
            label_width = self.tk_label_label_origin.winfo_width()
            label_height = self.tk_label_label_origin.winfo_height()
            # 计算图片的缩放比例
            image_width, image_height = image.size
            width_ratio = label_width / image_width
            height_ratio = label_height / image_height
            scale_ratio = min(width_ratio, height_ratio)
            # 缩放图片
            new_width = int(image_width * scale_ratio)
            new_height = int(image_height * scale_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            # 将缩放后的图片转换为PhotoImage对象
            self.pyt = ImageTk.PhotoImage(image)
            # 配置Label的image属性
            self.tk_label_label_origin.configure(image=self.pyt)
    # 初始化装载原图的frame
    def __tk_frame_frame_origin(self,parent):
        frame = Frame(parent, 
                    borderwidth=5, 
                    relief='groove',
                    bg='white',
                    highlightbackground=self.colors["primary"],
                    highlightthickness=3)
        frame.place(x=30, y=160, width=700, height=470)
        return frame

    def __tk_button_button_choose(self,parent):
        btn = Button(parent, 
                    command=self.choose_pic, 
                    text="选择图片", 
                    takefocus=False,
                    bg=self.colors["secondary"],
                    fg=self.colors["text_light"],
                    font=Font(family="Microsoft YaHei", size=12, weight="bold"),
                    relief='raised',
                    bd=3,
                    cursor="hand2")
        btn.place(x=740, y=175, width=242, height=60)
        
        def on_enter(e):
            #print("鼠标进入按钮")  # 调试信息
            btn.configure(bg='#2980b9')
            #print(f"按钮背景色改为: {btn.cget('bg')}")  # 检查颜色是否改变
        
        def on_leave(e):
            #print("鼠标离开按钮")  # 调试信息
            btn.configure(bg=self.colors["secondary"])
            #print(f"按钮背景色恢复: {btn.cget('bg')}")
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def __tk_button_button_apply(self,parent):
        btn = Button(parent, 
                    command=self.work, 
                    text="识别图片", 
                    takefocus=False,
                    bg=self.colors["success"],
                    fg=self.colors["text_light"],
                    font=Font(family="Microsoft YaHei", size=12, weight="bold"),
                    relief='raised',
                    bd=3,
                    cursor="hand2")
        btn.place(x=740, y=255, width=242, height=60)
        
        def on_enter(e):
            btn['bg'] = '#219955'  # 更深的绿色
            
        def on_leave(e):
            btn['bg'] = self.colors["success"]
            
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn
    # 初始化显示原图的label
    def __tk_label_label_origin(self,parent):
        # 创建默认占位图片
        self.default_image = self.create_placeholder_image(680, 445, "请选择图片")
        
        label = Label(parent, 
                    image=self.default_image,
                    anchor="center",
                    bg='white')
        label.place(width=680, height=445)
        return label

    def create_placeholder_image(self, width, height, text):
        """创建占位图片"""
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # 绘制边框
        draw.rectangle([0, 0, width-1, height-1], outline='#bdc3c7', width=2)
        
        # 绘制文字
        try:
            font = ImageFont.truetype("simhei.ttf", 24)  # 黑体
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill='#7f8c8d', font=font)
        
        return ImageTk.PhotoImage(image)
    # 初始化label用于显示标题
    def __tk_label_label_name(self,parent):
        label = Label(parent, 
                    text="智能车牌识别系统", 
                    anchor="center", 
                    bg=self.colors["primary"],
                    fg=self.colors["text_light"],
                    font=Font(family="Microsoft YaHei", size=28, weight="bold"))
        label.place(x=280, y=40, width=540, height=80)
        # 添加副标题
        subtitle = Label(parent,
                        text="基于深度学习的车牌检测与识别",
                        anchor="center",
                        bg=self.colors["background"],
                        fg=self.colors["text_dark"],
                        font=Font(family="Microsoft YaHei", size=12))
        subtitle.place(x=280, y=125, width=540, height=25)
        return label
    # 初始化frame用于装载车牌定位图
    def __tk_frame_frame_locate(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        # frame.place(x=740, y=372, width=242, height=100)
        frame.place(x=740, y=372, width=242, height=84) 
        return frame
    # 初始化label用于显示车牌定位图
    def __tk_label_label_locate(self,parent):
        label = Label(parent,anchor="center", )
        # label.place(x=0, y=0, width=235, height=93)
        label.place(x=0, y=0, width=235, height=77)
        return label
    def __tk_frame_frame_result(self,parent):
        frame = Frame(parent, borderwidth=5, relief='ridge')
        # frame.place(x=740, y=530, width=242, height=100)
        frame.place(x=740, y=530, width=242, height=84) #upd 2025/1/14 adjusted the ratio
        return frame
    
    def __tk_label_label_result(self,parent):
        label = Label(parent,
                    anchor="center",
                    font=Font(family="Arial", size=20, weight="bold"),
                    relief='sunken',
                    bd=2)
        label.place(x=0, y=0, width=235, height=77)
        return label

    def __tk_label_label_name_result(self,parent):
        label = Label(parent,
                    text="识别结果：",
                    anchor="center",
                    bg=self.colors["background"],
                    fg=self.colors["text_dark"],
                    font=Font(family="Microsoft YaHei", size=14, weight="bold"))
        label.place(x=740, y=490, width=100, height=30)
        return label
    # 初始化label用于显示文字 “车牌定位：”
    def __tk_label_label_name_locate(self,parent):
        label = Label(parent,
                    text="车牌定位：",
                    anchor="center",
                    bg=self.colors["background"],
                    fg=self.colors["text_dark"],
                    font=Font(family="Microsoft YaHei", size=14, weight="bold"))
        label.place(x=740, y=332, width=100, height=30)
        return label

if __name__ == "__main__":
    win = WinGUI() # 创建窗口
    win.mainloop() # 开始循环（等待用户操作）