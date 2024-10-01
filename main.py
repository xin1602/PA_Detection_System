from pathlib import Path
from tkinter import PhotoImage
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog, messagebox  
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
import math
from ultralytics import YOLO
import matplotlib.pyplot as plt
import shutil
from matplotlib import rcParams
import matplotlib
from tkinter.font import nametofont
import json
import csv
import onnxruntime as rt
from ttkbootstrap.scrolled import ScrolledFrame
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# 獲取當前檔案的路徑
PATH = Path(__file__).parent

class ToothRecognitionSystem(ttk.Frame):

    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)



        self.selected_image_path = None  # 儲存選取的圖片路徑

        self.predictions_list = []

        self.frames_dict = {}
        self.labels_dict = {}
        self.image_dict = {}

        # 初始化儲存預測結果的字典
        self.predictions_per_tooth = {}

        sf = ScrolledFrame(self, autohide=True)
        sf.pack(fill=BOTH, expand=YES, padx=0, pady=0)

        self.col1 = ttk.Frame(sf, padding=10)
        self.col1.grid(row=0, column=0, sticky=NSEW)

        self.col2 = ttk.Frame(sf, padding=0)
        # self.col2.grid_forget()
        self.col2.grid(row=0, column=1, sticky=NSEW)


        
        # 載入圖片
        self.load_images()

        # 1. 匯入圖片的區塊
        self.create_import_image_section()

        # 2. 影像處理區塊
        self.create_image_processing_section()

        # 3. 影像辨識區塊
        self.create_image_recognition_section()

        # 4. 執行按鈕
        self.create_execute_button()
        
        # 5. 進度條
        self.create_progress()

        # 6. 預測結果圖片顯示區塊
        self.create_predict()

        # 7. 預測結果統計與輸出區塊
        self.create_result_output()



        # self.style = ttk.Style()
        # # 定義自定義樣式，並明確指定父樣式
        # self.style.configure('Bold.TLabelframe', borderwidth=30, parent='TLabelframe')





    def load_images(self):
        # 載入圖片至 PhotoImage 物件中
        self.images = {
            'open_file': PhotoImage(file=PATH / 'photo'/ 'open_file.png'),
            'PA_file': PhotoImage(file=PATH / 'photo'/ 'no_file_2.png'),
            'predict_image' : PhotoImage(file=PATH / 'photo'/ 'not_detect_2.png'),
            'chart_image': PhotoImage(file=PATH / 'photo'/ 'no_file_2.png'),
            'flow_chart': PhotoImage(file=PATH / 'photo'/ 'flow_chart.png'),
            'cnn_img1': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img2': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img3': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img4': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img5': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img6': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img7': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img8': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img9': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png'),
            'cnn_img10': PhotoImage(file=PATH / 'photo'/ 'cnn_img_example.png')
            
        }

    def create_import_image_section(self):
        # 匯入圖片區塊
        self.import_frame = ttk.Labelframe(self.col1, text='匯入圖片', padding=10)
        self.import_frame.pack(side=TOP, fill=BOTH, expand=YES)

        # 選取檔案區塊
        import_frame_header = ttk.Frame(self.import_frame, padding=1)
        import_frame_header.pack(fill=X)

        self.lbl = ttk.Label(import_frame_header, text='選擇檔案')
        self.lbl.pack(side=LEFT, fill=X, padx=15)

                       
        btn1 = ttk.Button(
            master=import_frame_header,
            image=self.images['open_file'],
            bootstyle=LINK,
            command=self.select_image
        )
        btn1.pack(side=RIGHT)

        self.image_label = ttk.Label(self.import_frame, image=self.images['PA_file'])
        # self.image_label.pack_forget()
        self.image_label.pack(fill="y")


    def resize_image(self,file_path,max_width,max_height,save_folder,save_basename):
        # 開啟圖片
        image = Image.open(file_path)
        # 確認長寬比，調整大小並轉換格式
        width, height = image.size
        if width > height:
            new_width = max_width
            new_height = int(new_width * height / width)
        else:
            new_height = max_height
            new_width = int(new_height * width / height) 
        resized_image = image.resize((new_width, new_height))
        # 確認 adjusted_select_image 資料夾存在，若不存在則創建
        save_folder.mkdir(parents=True, exist_ok=True)
        # 儲存調整後的圖片為 PNG 格式
        save_path = save_folder / save_basename
        resized_image.save(save_path, 'PNG')

        return save_path


    def select_image(self):

        # 定義可接受的檔案類型
        file_types = [
            ("Image files", "*.png *.jpg"),
        ]
        # 選取圖片的函式
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            file_name = os.path.basename(file_path)
            
            # 處理圖片
            try:
                # 開啟圖片
                image = Image.open(file_path)
                
                # 確認 select_image 資料夾存在，若不存在則創建
                select_image_dir = PATH / 'select_image'
                select_image_dir.mkdir(parents=True, exist_ok=True)
                
                # 儲存調整後的圖片為 PNG 格式
                select_image = PATH / 'select_image' / '{}.png'.format(os.path.splitext(file_name)[0])
                image.save(select_image, 'png')
                # 儲存選取的圖片路徑
                self.selected_image_path = select_image

                # 確認長寬比，調整大小並轉換格式
                save_path =self.resize_image(select_image,500,355,PATH / 'adjusted_select_image', 'adjusted_select_image.png')
                
                # 更新按鈕文字為 "已選擇"
                self.lbl.config(text='已選擇：'+ file_name)
                
                # 處理圖片並更新圖片顯示
                adjusted_image = PhotoImage(file=save_path)
                self.image_label.configure(image=adjusted_image)  # 更新圖片顯示
                self.images['PA_file'] = adjusted_image  # 更新圖片字典中的 'PA_file'
                # self.image_label.pack(fill="y")
                    
            except Exception as e:
                messagebox.showerror('錯誤', f'處理圖片時發生錯誤：{e}')

    def create_image_processing_section(self):
        # 影像處理區塊
        processing_frame = ttk.Labelframe(
            master=self.col1,
            text='影像處理',
            padding=(15, 10)
        )
        processing_frame.pack(
            side=TOP,
            fill=BOTH,
            expand=YES,
            pady=(10, 0)
        )
    
        # 在這裡建立影像處理的相關元件，例如單選按鈕等

        self.image_processing_method = ttk.StringVar()

        op11 = ttk.Radiobutton(
            master=processing_frame,
            text='無強化',
            variable=self.image_processing_method,
            value='無強化'
        )
        op11.pack(fill=X, pady=5)
        self.image_processing_method.set('無強化')

        op12 = ttk.Radiobutton(
            master=processing_frame,
            text='強化牙周與加強牙齒輪廓 ',
            variable=self.image_processing_method,
            value='強化牙周與加強牙齒輪廓 '
        )
        op12.pack(fill=X, pady=5)

        op13 = ttk.Radiobutton(
            master=processing_frame,
            text='強化牙根與病灶點 ',
            variable=self.image_processing_method,
            value='強化牙根與病灶點 '
        )
        op13.pack(fill=X, pady=5)

        op14 = ttk.Radiobutton(
            master=processing_frame,
            text='強化整體(牙周+牙根)',
            variable=self.image_processing_method,
            value='強化整體(牙周+牙根)'
        )
        op14.pack(fill=X, pady=5)
     
    def create_image_recognition_section(self):
        # 影像辨識區塊
        recognition_frame = ttk.Labelframe(
            master=self.col1,
            text='影像辨識',
            padding=(15, 10)
        )
        recognition_frame.pack(
            side=TOP,
            fill=BOTH,
            expand=YES,
            pady=(10, 0)
        )

        # CNN區塊
        cnn_frame = ttk.Frame(recognition_frame, padding=1)
        cnn_frame.pack(fill=X)

        # 在這裡建立影像辨識的相關元件，例如單選按鈕和下拉選單等
        self.image_recognition_method = ttk.StringVar()

        op11 = ttk.Radiobutton(
            master=cnn_frame,
            text='CNN 分類',
            variable=self.image_recognition_method,
            value='CNN 分類'
        )
        op11.pack(side=LEFT,fill=X, pady=5)
        self.image_recognition_method.set('CNN 分類')

        # self.image_cnn_method = ttk.StringVar()
        # self.image_cnn_method.set('AlexNet')  # 設定預設選項為AlexNet

        # two_finger_cbo = ttk.Combobox(
        #     master=cnn_frame,
        #     textvariable=self.image_cnn_method,
        #     values=['AlexNet','VGG16','ConvNeXt'],
        #     state="readonly"  # 設定為唯讀，使其成為單選框
        # )


        # two_finger_cbo.current(0)
        # two_finger_cbo.pack(side=RIGHT,fill=X, padx=(20, 0), pady=5)

        op12 = ttk.Radiobutton(
            master=recognition_frame,
            text='YOLOv8 物件偵測',
            variable=self.image_recognition_method,
            value='YOLOv8 物件偵測'
        )
        op12.pack(fill=X, pady=5)

    def create_execute_button(self):
        
        # 執行按鈕區塊
        self.execute_frame = ttk.Labelframe(self.col1, padding=0,borderwidth=0)
        self.execute_frame.pack(side=TOP, fill=BOTH, expand=YES)

        btn_execute = ttk.Button(
            master=self.execute_frame,
            text='執行',
            bootstyle="warning",
            command=self.execute_process
        )
        btn_execute.pack(padx=0,pady=10,ipady=6, fill=X)

    def execute_process(self):
    # 執行按鈕的動作，在這裡根據選擇的影像處理和影像辨識方式執行相應的處理和辨識流程
        if self.selected_image_path==None:
            print('尚未匯入檔案')
            Messagebox.show_error(
                title='請先匯入檔案', 
                message="請選擇X光片，我們將為您進行牙齒病徵辨識。",
                parent=self.import_frame  # 將消息框置於此層上方
            )

        else:  
            # 判斷影像辨識選擇CNN分類還是YOLO物件偵測
            if self.image_recognition_method.get() == 'CNN 分類':  # CNN分類
                print('CNN 分類')

                self.update_progress(15,'影像切割中，請稍後')
                # 單顆牙齒的影像切割
                self.segmentation(self.selected_image_path)
                print('影像切割')

                # 確認長寬比，調整大小並轉換格式
                save_path = self.resize_image(PATH / 'single_tooth_predict' / (os.path.splitext(os.path.basename(self.selected_image_path))[0] + '.png'),500,500,PATH / 'adjusted_select_image','adjusted_select_image.png')

                # 更新按鈕文字為 "已選擇"
                self.lbl.config(text='已選擇：'+ os.path.basename(self.selected_image_path))

                # 更新圖片顯示
                single_tooth_predict_image = PhotoImage(file= save_path)
                self.image_label.configure(image=single_tooth_predict_image)  # 更新圖片顯示
                self.images['PA_file'] = single_tooth_predict_image  # 更新圖片字典中的 'PA_file'

                self.update_progress(30,'影像處理中，請稍後')
                # 判斷選擇哪個影像處理並執行
                if self.image_processing_method.get() == '無強化':  # 無強化
                    print('無強化')
                    processed_image_path = self.process_image_grayscale_folder(PATH / 'single_tooth',PATH / 'single_tooth_processed_image')
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth_processed_image',PATH / 'single_tooth_processed_resize_padding_image')
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth',PATH / 'single_tooth_resize_padding_image')
                    processed_image_path = self.process_image_grayscale_folder(PATH / 'single_tooth_resize_padding_image',PATH / 'single_tooth_resize_padding_processed_image')
                elif self.image_processing_method.get() == '強化牙周與加強牙齒輪廓 ':  # 高斯高通濾波器
                    print('強化牙周與加強牙齒輪廓 ')
                    processed_image_path = self.process_image_gaussian_folder(PATH / 'single_tooth',PATH / 'single_tooth_processed_image')
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth_processed_image',PATH / 'single_tooth_processed_resize_padding_image')
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth',PATH / 'single_tooth_resize_padding_image')
                    processed_image_path = self.process_image_gaussian_folder(PATH / 'single_tooth_resize_padding_image',PATH / 'single_tooth_resize_padding_processed_image')
                elif self.image_processing_method.get() == '強化牙根與病灶點 ':  # 自適應直方圖均衡化
                    print('強化牙根與病灶點 ')
                    processed_image_path = self.process_image_adaptive_histogram_equalization_folder(PATH / 'single_tooth',PATH / 'single_tooth_processed_image',4.0,4,4)
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth_processed_image',PATH / 'single_tooth_processed_resize_padding_image')
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth',PATH / 'single_tooth_resize_padding_image')
                    processed_image_path = self.process_image_adaptive_histogram_equalization_folder(PATH / 'single_tooth_resize_padding_image',PATH / 'single_tooth_resize_padding_processed_image',4.0,4,4)
                    
                elif self.image_processing_method.get() == '強化整體(牙周+牙根)':  # 強化整體(牙周+牙根)
                    print('強化整體(牙周+牙根)')
                    processed_image_path = self.process_image_gaussian_folder(PATH / 'single_tooth',PATH / 'single_tooth_processed_image')
                    processed_image_path = self.process_image_adaptive_histogram_equalization_folder(PATH / 'single_tooth_processed_image',PATH / 'single_tooth_processed_image',4.0,4,4)
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth_processed_image',PATH / 'single_tooth_processed_resize_padding_image')
                    processed_image_path = self.process_image_resize_padding_folder(PATH / 'single_tooth',PATH / 'single_tooth_resize_padding_image')
                    processed_image_path = self.process_image_gaussian_folder(PATH / 'single_tooth_resize_padding_image',PATH / 'single_tooth_resize_padding_processed_image')
                    processed_image_path = self.process_image_adaptive_histogram_equalization_folder(PATH / 'single_tooth_resize_padding_processed_image',PATH / 'single_tooth_resize_padding_processed_image',4.0,4,4)
                
                self.update_progress(50,'影像辨識中，請稍後')
                # # 判斷選擇哪個CNN並進行預測
                # if self.image_cnn_method.get() == 'AlexNet':
                #     print('AlexNet')
                #     self.process_image_classification(processed_image_path)
                # elif self.image_cnn_method.get() == 'VGG16': 
                #     print('VGG16')
                #     self.process_image_classification(processed_image_path)
                # elif self.image_cnn_method.get() == 'ConvNeXt':
                #     print('ConvNeXt') 
                #     self.process_image_classification(processed_image_path)
                
                self.process_image_classification(processed_image_path)

                self.update_progress(100,'影像辨識完成')

            elif self.image_recognition_method.get() == 'YOLOv8 物件偵測':  # YOLOv8物件偵測
                print('YOLOv8 物件偵測')

                self.update_progress(30,'影像處理中，請稍後')

                # 預測前 先外擴輸入YOLO圖像 防止最後標籤和概率出界
                self.expand_image(self.selected_image_path, self.selected_image_path, expand_size=100, fill_color=(0, 0, 0))

                # 判斷選擇哪個影像處理並執行
                if self.image_processing_method.get() == '無強化':  # 無強化
                    print('無強化')
                    processed_image_path = self.process_image_grayscale_file(self.selected_image_path)
                elif self.image_processing_method.get() == '強化牙周與加強牙齒輪廓 ':  # 強化牙周與加強牙齒輪廓
                    print('強化牙周與加強牙齒輪廓 ')
                    processed_image_path = self.process_image_gaussian_file(self.selected_image_path)
                elif self.image_processing_method.get() == '強化牙根與病灶點 ':  # 強化牙根與病灶點
                    print('強化牙根與病灶點 ')
                    processed_image_path = self.process_image_adaptive_histogram_equalization_file(self.selected_image_path,2.0,4,4)
                elif self.image_processing_method.get() == '強化整體(牙周+牙根)':  # 強化整體(牙周+牙根)
                    print('強化整體(牙周+牙根)')
                    self.process_image_grayscale_file(self.selected_image_path)
                    self.process_image_adaptive_histogram_equalization_file(PATH / 'PA_processed_image' /os.path.basename(self.selected_image_path),2.0,4,4)

                self.update_progress(50,'影像辨識中，請稍後')
                self.process_image_detection(PATH / 'PA_processed_image' /os.path.basename(self.selected_image_path),PATH / 'YOLO_predict_image')
                self.update_progress(100,'影像辨識完成')

    
#  預測前先外擴輸入YOLO圖像
    def expand_image(self,image_path, output_path, expand_size=200, fill_color=(0, 0, 0)):
        # 打开图像
        img = Image.open(image_path)

        # 获取原始图像大小
        width, height = img.size

        # 计算扩展后的大小
        new_width = width + 2 * expand_size
        new_height = height + 2 * expand_size

        # 创建新图像
        new_img = Image.new(img.mode, (new_width, new_height), fill_color)

        # 将原始图像粘贴到新图像中心
        new_img.paste(img, (expand_size, expand_size))

        # 保存新图像
        new_img.save(output_path)

    def create_progress(self):
        # 新增進度條
        # 監控執行進度

        # gauge = ttk.Floodgauge(
        #     self.execute_frame,
        #     bootstyle="warning",
        #     # font=(None, 14, 'bold'),
        #     text='0%'
        # )
        # gauge.pack(fill=BOTH, expand=YES, padx=0, pady=20)
        # # autoincrement the gauge
        # gauge.start()
        # # stop the autoincrement
        # # gauge.stop()

        self.pb = ttk.Progressbar(self.execute_frame, value=0,style='warning.Striped.Horizontal.TProgressbar')
        self.pb.pack_forget()
        self.progressbar = ttk.Label(self.pb, text='0%',background="#eeeeee")
        self.progressbar.pack_forget()

        self.pb_lb_text = ttk.StringVar()
        self.pb_lb_text.set('準備中')

        # progress message
        self.pb_lb = ttk.Label(
            master=self.execute_frame,
            text=self.pb_lb_text,
            anchor=CENTER
        )
        self.pb_lb.pack_forget()
         
    def update_progress(self, value,text):
        self.pb.pack(fill=X,pady=20, padx=0,ipady=5)
        self.progressbar.pack(side=RIGHT,ipady=0)
        self.pb_lb.pack()
        # 更新進度條的值
        self.pb["value"] = value
        self.progressbar.config(text=f'{value}%')
        # self.pb_lb_text.set=text
        self.pb_lb['text'] = text
        self.update()  # 立即更新視窗畫面

    def create_predict(self):
        # 預測結果圖片顯示區塊
        self.predict_frame = ttk.Labelframe(self.col2, padding=0,borderwidth=0)
        self.predict_frame.pack(side=TOP, fill=BOTH, expand=YES)
        # self.predict_frame.pack_forget()



        for i in range(10): 
            self.frames_dict[i+1] = ttk.Frame(self.predict_frame, padding=0)
            self.frames_dict[i+1].pack_forget()

            self.labels_dict[f'f{i+1}_lb1'] = ttk.Label(self.frames_dict[i+1], text='',style="inverse_primary.TLabel") #
            self.labels_dict[f'f{i+1}_lb1'].pack_forget()

            self.labels_dict[f'f{i+1}_lb2'] = ttk.Label(self.frames_dict[i+1], text='')
            self.labels_dict[f'f{i+1}_lb2'].pack_forget()

            self.labels_dict[f'f{i+1}_lb3'] = ttk.Label(self.frames_dict[i+1], text='')
            self.labels_dict[f'f{i+1}_lb3'].pack_forget()

            self.image_dict[f'f{i+1}'] = ttk.Label(self.frames_dict[i+1], image=self.images[f'cnn_img{i+1}'])
            self.image_dict[f'f{i+1}'].pack_forget()



        self.predict_image_label = ttk.Label(self.predict_frame, image=self.images['predict_image'])
        self.predict_image_label.pack(fill="y")
                 

    def create_result_output(self):
        # 預測結果統計與輸出區塊
        self.result_output_frame = ttk.Labelframe(self.col2, padding=0,borderwidth=0)
        self.result_output_frame.pack(side=TOP, fill=BOTH, expand=YES)

        self.flow_chart_label = ttk.Label(self.result_output_frame, image=self.images['flow_chart'])
        self.flow_chart_label.pack(fill="both")
        # self.flow_chart_label.pack_forget
        

        # 第一欄區塊
        self.result_frame = ttk.Frame(self.result_output_frame, padding=1)
        # self.result_frame.pack(side=LEFT)
        self.result_frame.pack_forget()

        self.result_label = ttk.Label(self.result_frame, image=self.images['chart_image'])
        # self.result_label.pack(fill=X)
        self.result_label.pack_forget()

        # 第二欄區塊
        self.output_frame = ttk.Frame(self.result_output_frame, padding=1)
        # self.output_frame.pack(side=RIGHT,expand=1)
        self.output_frame.pack_forget()

        self.output_btn1 = ttk.Button(
            master=self.output_frame,
            text='輸出圖片',
            bootstyle=INFO,
            command=self.output_imgOrPDF  #output_img
        )
        # self.output_btn1.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn1.pack_forget()

        self.output_btn2 = ttk.Button(
            master=self.output_frame,
            text='輸出txt',
            bootstyle=INFO,
            command=self.output_txt
        )
        # self.output_btn2.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn2.pack_forget()

        self.output_btn3 = ttk.Button(
            master=self.output_frame,
            text='輸出json',
            bootstyle=INFO,
            command=self.output_json
        )
        # self.output_btn3.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn3.pack_forget()

        self.output_btn4 = ttk.Button(
            master=self.output_frame,
            text='輸出csv',
            bootstyle=INFO,
            command=self.output_csv
        )
        # self.output_btn4.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn4.pack_forget()

        self.output_btn5 = ttk.Button(
            master=self.output_frame,
            text='清除資料',
            bootstyle=SECONDARY,
            command=self.clean_data
        )
        # self.output_btn5.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)  
        self.output_btn5.pack_forget()
           
           

        

# CNN影像切割   
    def segmentation(self,image_path):
        # 實現影像切割成多個單顆牙齒的圖像
        # 存放於single_tooth資料夾內

        # Rotation matrix function
        def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
            """
            Rotates a point in the xy-plane counterclockwise through an angle about the origin
            https://en.wikipedia.org/wiki/Rotation_matrix
            :param x: x coordinate
            :param y: y coordinate
            :param x_shift: x-axis shift from origin (0, 0)
            :param y_shift: y-axis shift from origin (0, 0)
            :param angle: The rotation angle in degrees
            :param units: DEGREES (default) or RADIANS
            :return: Tuple of rotated x and y
            """

            # Shift to origin (0,0)
            x = x - x_shift
            y = y - y_shift

            # Convert degrees to radians
            if units == "DEGREES":
                angle = math.radians(angle)

            # Rotation matrix multiplication to get rotated x & y
            xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
            yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

            return xr, yr


        # 初始化 YOLOv8 模型
        model = YOLO(PATH / 'CNN_segment_model' / "best.pt")
        

        # 讀取圖片
        image = cv2.imread(image_path)
        # 執行預測
        results = model(image)
        # 獲取第一個結果
        result = results[0]
        # 獲取 OBB 信息
        obb_info = result.obb
        # 獲取圖像的寬度和高度
        height, width, channels = image.shape
        center_x = width//2
        center_y = height//2
        
        bbox_info = []


        original_filename = os.path.splitext(os.path.basename(image_path))[0]

        # 對每個邊界框進行處理
        sorted_indices = sorted(range(len(obb_info.xyxyxyxy)), key=lambda i: obb_info.xyxyxyxy[i][0][0].item())  # 根據 orbit_center_x 排序的索引列表
        for i, idx in enumerate(sorted_indices):

            # 獲取邊界框的四個角點座標
            pts = obb_info.xyxyxyxy[idx].numpy().reshape(-1, 2).astype(np.int32)

            # 計算旋轉角度
            angle = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]) * 180 / math.pi

            rotation_center = (center_x, center_y)

            # 將圖像旋轉至水平位置，並指定旋轉中心
            rotated_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).rotate(angle, expand=True, center=rotation_center)

            # 保存旋转后的图像
            output_folder_rotated = PATH / 'single_tooth_rotated_images'
            if not os.path.exists(output_folder_rotated):
                os.makedirs(output_folder_rotated)
            rotated_image_path = os.path.join(output_folder_rotated, f'{os.path.splitext(os.path.basename(image_path))[0]}_{i+1}_rotated.png')
            rotated_image.save(rotated_image_path)

            # 圖像旋轉 擴展後 圖像寬度、長度、中心點位置
            width_expand ,height_expand= rotated_image.size
            center_x_expand = width_expand//2
            center_y_expand = height_expand//2


            ax, ay = rotate_matrix(pts[0][0]+(center_x_expand-center_x), pts[0][1]+(center_y_expand-center_y), -angle, center_x_expand, center_y_expand)
            bx, by = rotate_matrix(pts[1][0]+(center_x_expand-center_x), pts[1][1]+(center_y_expand-center_y), -angle, center_x_expand, center_y_expand)
            cx, cy = rotate_matrix(pts[2][0]+(center_x_expand-center_x), pts[2][1]+(center_y_expand-center_y), -angle, center_x_expand, center_y_expand)
            dx, dy = rotate_matrix(pts[3][0]+(center_x_expand-center_x), pts[3][1]+(center_y_expand-center_y), -angle, center_x_expand, center_y_expand)



            # 將旋轉後的圖像根據範圍裁剪
            cropped_image = rotated_image.crop((int(ax), int(cy), int(cx), int(ay)))

            # 将边界框信息保存到列表中
            bbox_info.append((ax, cy, cx, ay))


            # 保存边界框信息
            output_folder_bbox = PATH / 'single_tooth_bbox_info'
            if not os.path.exists(output_folder_bbox):
                os.makedirs(output_folder_bbox)
            bbox_info_path = os.path.join(output_folder_bbox, f'{os.path.splitext(os.path.basename(image_path))[0]}_{i+1}_bbox_info.npy')
            np.save(bbox_info_path, bbox_info)
            

            # 保存提取的图像部分
            output_folder = PATH / 'single_tooth' 
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cropped_image.save(os.path.join(output_folder, f'{original_filename}_{i+1}.png'))

        # 保存预测结果
        output_folder = PATH / 'single_tooth_predict'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(os.path.join(output_folder, f'{original_filename}.png'))  # save image
        return output_folder

# CNN影像處理
    def process_image_grayscale_folder(self,input_folder,output_folder):
        # 實現將影像轉為灰階的方法
        # 返回處理後的影像路徑

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):

            image_path = os.path.join(input_folder, filename)

            # 讀取原始影像
            img = cv2.imread(image_path)

            # 將影像轉換為灰度
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            output_path = output_folder / os.path.basename(image_path)
            cv2.imwrite(output_path, gray_img)

        return output_folder

    def process_image_gaussian_folder(self,input_folder,output_folder):
        # 實現高斯高通濾波器的影像處理方法
        # 返回處理後的影像路徑

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            
            image_path = os.path.join(input_folder, filename)
            # Read the image
            f = cv2.imread(image_path, 0)

            # Transform the image into frequency domain, f --> F
            F = np.fft.fft2(f)
            Fshift = np.fft.fftshift(F)

            # Create Gaussian Filter: High Pass Filter
            M, N = f.shape
            H = np.zeros((M, N), dtype=np.float32)
            D0 = 2.5
            for u in range(M):
                for v in range(N):
                    D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
                    H[u, v] = 1 - np.exp(-D**2 / (2 * D0**2))

            # Apply High Pass Filter
            Gshift = Fshift * H
            G = np.fft.ifftshift(Gshift)
            g = np.abs(np.fft.ifft2(G))

            # Convert g to uint8
            g = g.astype(np.uint8)

            output_path = output_folder/ os.path.basename(image_path)

            cv2.imwrite(output_path, cv2.subtract(f, g))

        return output_folder

    def process_image_adaptive_histogram_equalization_folder(self,input_folder,output_folder,cli,til1,til2):
        # 實現自適應直方圖均衡化的影像處理方法
        # 返回處理後的影像路徑

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            
            image_path = os.path.join(input_folder, filename)

            # 讀取原始影像
            img = cv2.imread(image_path)

            # 將影像轉換為灰度
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 使用自適應直方圖均衡化處理影像
            clahe = cv2.createCLAHE(clipLimit=cli, tileGridSize=(til1, til2))
            adaptive_equalized_img = clahe.apply(gray_img)

            output_path = output_folder / os.path.basename(image_path)
            cv2.imwrite(output_path, adaptive_equalized_img)

        return output_folder

# 調整大小並黑色填充(CNN)
    def process_image_resize_padding_folder(self,input_folder,output_folder):
        # 調整大小並黑色填充(CNN)
        # 返回處理後的影像路徑

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):

            image_path = os.path.join(input_folder, filename)

            # 讀取圖片
            img = cv2.imread(image_path)

            # 轉換成灰階圖像
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 轉換成彩色圖像 (3通道)
            color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # 計算原始圖片的寬高比例
            aspect_ratio = img.shape[1] / img.shape[0]

            # 設定目標高度為224，計算目標寬度並調整圖片大小
            target_height = 224
            target_width = int(aspect_ratio * target_height)
            resized_image = cv2.resize(color_image, (target_width, target_height))

            # 計算填充黑色的寬度和高度
            padding_width = (224 - resized_image.shape[1]) // 2
            padding_height = (224 - resized_image.shape[0]) // 2

            # 進行黑色填充
            resized_image_padded = cv2.copyMakeBorder(resized_image, padding_height, padding_height,
                                                      padding_width, padding_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # 保存處理後的圖片
            output_path = output_folder / os.path.basename(image_path)
            cv2.imwrite(output_path, resized_image_padded)

        return output_folder

    def process_image_grayscale_file(self,image_path):
        # 實現將影像轉為灰階的方法
        # 返回處理後的影像路徑

        # 讀取原始影像
        img = cv2.imread(image_path)

        # 將影像轉換為灰度
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        output_folder = PATH / 'PA_processed_image'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = PATH / 'PA_processed_image' / os.path.basename(image_path)
        cv2.imwrite(output_path, gray_img)

    def process_image_gaussian_file(self,image_path):
        # 實現高斯高通濾波器的影像處理方法
        # 返回處理後的影像路徑
        
        # Read the image
        f = cv2.imread(image_path, 0)

        # Transform the image into frequency domain, f --> F
        F = np.fft.fft2(f)
        Fshift = np.fft.fftshift(F)

        # Create Gaussian Filter: High Pass Filter
        M, N = f.shape
        H = np.zeros((M, N), dtype=np.float32)
        D0 = 2.5
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
                H[u, v] = 1 - np.exp(-D**2 / (2 * D0**2))

        # Apply High Pass Filter
        Gshift = Fshift * H
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))

        # Convert g to uint8
        g = g.astype(np.uint8)

        output_folder = PATH / 'PA_processed_image'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = PATH / 'PA_processed_image' / os.path.basename(image_path)

        cv2.imwrite(output_path, cv2.subtract(f, g))

    def process_image_adaptive_histogram_equalization_file(self,image_path,cli,til1,til2):
        # 實現自適應直方圖均衡化的影像處理方法
        # 返回處理後的影像路徑

        # 讀取原始影像
        img = cv2.imread(image_path)

        # 將影像轉換為灰度
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用自適應直方圖均衡化處理影像
        clahe = cv2.createCLAHE(clipLimit=cli, tileGridSize=(til1, til2))
        adaptive_equalized_img = clahe.apply(gray_img)

        output_folder = PATH / 'PA_processed_image'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = PATH / 'PA_processed_image' / os.path.basename(image_path)
        cv2.imwrite(output_path, adaptive_equalized_img)

# CNN影像辨識
    def process_image_classification(self, image_path):
        # 實現CNN模型的影像分類方法
        # 這裡需要將影像切割成單顆牙齒的圖像，存放於single_tooth資料夾內
        # 再導入訓練好的CNN模型(.onnx)並進行每張圖片的預測結果，需要有預測標籤、概率
        # 最後顯示到畫面上給用戶看，並統計不同標籤的預測數量，再顯示在畫面上

        self.predict_image_label.pack_forget()

        # # 判斷是哪個影像處理方法
        # if self.image_processing_method.get() == '無強化':  # 無強化
        #     if self.image_cnn_method.get() == 'AlexNet':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'original' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'VGG16':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'original' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'ConvNeXt':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'original' / 'model.onnx'

        # elif self.image_processing_method.get() == '強化牙周與加強牙齒輪廓 ':  # 高斯高通濾波器
        #     if self.image_cnn_method.get() == 'AlexNet':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'GH' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'VGG16':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'GH' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'ConvNeXt':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'GH' / 'model.onnx'
        
        # elif self.image_processing_method.get() == '強化牙根與病灶點 ':  # 自適應直方圖均衡化
        #     if self.image_cnn_method.get() == 'AlexNet':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'AHE' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'VGG16':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'AHE' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'ConvNeXt':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'AHE' / 'model.onnx'
        
        # elif self.image_processing_method.get() == '強化整體(牙周+牙根)':  # 強化整體(牙周+牙根)
        #     if self.image_cnn_method.get() == 'AlexNet':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'GH+AHE' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'VGG16':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'GH+AHE' / 'model.onnx'
        #     elif self.image_cnn_method.get() == 'ConvNeXt':
        #         self.model = PATH / 'CNN_model' / 'alexnet' / 'GH+AHE' / 'model.onnx'

        # 判斷是哪個影像處理方法
        if self.image_processing_method.get() == '無強化':  # 無強化
            self.model = PATH / 'CNN_model' / 'alexnet' / 'original' / 'model.onnx'
            self.save_CNN_class_prob(227)
            self.model = PATH / 'CNN_model' / 'googlenet' / 'original' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'places365googlenet' / 'original' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'resnet50' / 'original' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'vgg16' / 'original' / 'model.onnx'
            self.save_CNN_class_prob(224)

            self.show_CNN_class_prob()

        elif self.image_processing_method.get() == '強化牙周與加強牙齒輪廓 ':  # 高斯高通濾波器
            self.model = PATH / 'CNN_model' / 'alexnet' / 'GH' / 'model.onnx'
            self.save_CNN_class_prob(227)
            self.model = PATH / 'CNN_model' / 'googlenet' / 'GH' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'places365googlenet' / 'GH' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'resnet50' / 'GH' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'vgg16' / 'GH' / 'model.onnx'
            self.save_CNN_class_prob(224)

            self.show_CNN_class_prob()            
        
        elif self.image_processing_method.get() == '強化牙根與病灶點 ':  # 自適應直方圖均衡化
            self.model = PATH / 'CNN_model' / 'alexnet' / 'AHE' / 'model.onnx'
            self.save_CNN_class_prob(227)
            self.model = PATH / 'CNN_model' / 'googlenet' / 'AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'places365googlenet' / 'AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'resnet50' / 'AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'vgg16' / 'AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)

            self.show_CNN_class_prob()               
        
        elif self.image_processing_method.get() == '強化整體(牙周+牙根)':  # 強化整體(牙周+牙根)
            self.model = PATH / 'CNN_model' / 'alexnet' / 'GH+AHE' / 'model.onnx'
            self.save_CNN_class_prob(227)
            self.model = PATH / 'CNN_model' / 'googlenet' / 'GH+AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'places365googlenet' / 'GH+AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'resnet50' / 'GH+AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)
            self.model = PATH / 'CNN_model' / 'vgg16' / 'GH+AHE' / 'model.onnx'
            self.save_CNN_class_prob(224)

            self.show_CNN_class_prob()     

        # self.predict_frame = ttk.Labelframe(self.col2, padding=0,borderwidth=0)
        # self.predict_frame.pack(side=TOP, fill=BOTH, expand=YES)

         
        # for i in range(10): 
        #     if f'f{i+1}_lb1' in self.labels_dict:
        #         self.labels_dict[f'f{i+1}_lb1'].destroy()
        #     if f'f{i+1}_lb2' in self.labels_dict:
        #         self.labels_dict[f'f{i+1}_lb2'].destroy()
        #     if f'f{i+1}_lb3' in self.labels_dict:
        #         self.labels_dict[f'f{i+1}_lb3'].destroy()
        #     if f'f{i+1}' in self.image_dict:
        #         self.image_dict[f'f{i+1}'].destroy()




    def save_CNN_class_prob(self,size):
        # 文件夹路径
        folder_path = PATH / 'single_tooth_resize_padding_processed_image'

        # 获取文件夹中所有图像文件的文件名列表，并按名称排序
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")])
        
        for filename in image_files:
            image_path = folder_path / filename
            predictions = self.predict_CNN_image(image_path,size)

            # 获取最高概率的类别和概率值
            high_prob_index = np.argmax(predictions)
            high_prob_value = round(predictions[0][high_prob_index], 4)
            # 將小數轉換為百分比形式（保留兩位小數）
            percentage_prob = round(high_prob_value * 100, 2)
            class_labels = ['根尖病灶', '無病徵', '根尖病灶合併牙周流失']  
            predicted_class = class_labels[high_prob_index]
            print(f"filename：{filename}, Predicted Class: {predicted_class}, Probability: {percentage_prob}%")

            # 將預測結果添加到 predictions_per_tooth 字典中
            if filename not in self.predictions_per_tooth:
                self.predictions_per_tooth[filename] = []
            self.predictions_per_tooth[filename].append({
                "class": predicted_class,
                "prob": percentage_prob
            })

    def calculate_final_probability(self, filename):
        # 儲存每個類別的平均概率
        class_probabilities = {}

        # 計算每個類別的平均概率
        for pred in self.predictions_per_tooth[filename]:
            class_name = pred['class']
            prob = pred['prob']
            if class_name not in class_probabilities:
                class_probabilities[class_name] = []
            class_probabilities[class_name].append(prob)

        # 計算每個類別的平均概率
        for class_name, probs in class_probabilities.items():
            avg_prob = sum(probs) / len(probs)
            # 如果等于100.0或100，则替换为99.99
            if avg_prob >= 100:
                avg_prob = 99.99
            class_probabilities[class_name] = round(avg_prob, 2)


        # 確定最大平均概率的類別及其最終概率
        max_class = max(class_probabilities, key=class_probabilities.get)
        final_prob = class_probabilities[max_class]

        return max_class, final_prob

    def show_CNN_class_prob(self):
        w = 800 / len(self.predictions_per_tooth)
        h = 600

        # 文件夾路径
        folder_path = PATH / 'single_tooth_resize_padding_processed_image'

        # 获取文件夹中所有图像文件的文件名列表，并按名称排序
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")])     
        
        for i, filename in enumerate(image_files):
            self.frames_dict[i+1].config(width=w, height=h)
            self.frames_dict[i+1].pack(side=LEFT, fill=X)

            # 使用預測結果字典計算最終類別和概率
            final_class, final_prob = self.calculate_final_probability(filename)

            # 將預測結果添加到 self.predictions_list 中
            self.predictions_list.append({
                "filename": filename,
                "predicted_class": final_class,
                "probability": f'{final_prob}%'
            })
            
            # 使用 config 方法設置新文本
            self.labels_dict[f'f{i+1}_lb1'].config(text=f'第{i+1}顆牙齒')
            self.labels_dict[f'f{i+1}_lb1'].pack(side=TOP, fill="y", padx=0, ipadx=0, ipady=10)

            self.labels_dict[f'f{i+1}_lb2'].config(text=final_class)
            self.labels_dict[f'f{i+1}_lb2'].pack(side=TOP, fill="y", padx=0, ipadx=0, ipady=10)

            self.labels_dict[f'f{i+1}_lb3'].config(text=f'{final_prob}%')
            self.labels_dict[f'f{i+1}_lb3'].pack(side=TOP, fill="y", padx=0, ipadx=0, ipady=10)
            

            # 確認長寬比，調整大小並轉換格式
            save_path =self.resize_image( PATH / 'single_tooth_processed_image' /filename ,w,400, PATH / 'CNN_resize_image', filename)
            
            # 更新圖片顯示
            img = PhotoImage(file=save_path)
            self.images[f'cnn_img{i+1}'] = img  # 更新圖片字典中的 f'cnn_img{i+1}'

            self.image_dict[f'f{i+1}'] = ttk.Label(self.frames_dict[i+1], image=self.images[f'cnn_img{i+1}'])
            self.image_dict[f'f{i+1}'].pack(fill="y", expand=1)


        self.flow_chart_label.pack_forget()
        self.result_frame.pack(side=LEFT)
        self.result_label.pack(fill=X)
        self.output_frame.pack(side=RIGHT,expand=1)
        self.output_btn1.config(text='輸出pdf')
        self.output_btn1.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn2.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn3.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn4.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn5.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)

        self.update_chart()
        
    
    # CNN預測函數
    def predict_CNN_image(self,image_path,size):
        # 載入圖像並預處理
        img = Image.open(image_path)
        img = img.resize((size, size))
        # 将灰度图像扩展为3通道，再转换为期望的形状 [1, 3, 227, 227]
        img_array = np.array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=-1)  # 扩展为3通道
        img_array = np.repeat(img_array, 3, axis=-1)  # 复制通道
        img_array = img_array.transpose((2, 0, 1))  # 将通道置于第二维
        img_array = np.expand_dims(img_array, axis=0)  # 添加 batch 维度

        session = rt.InferenceSession(self.model)

        # 執行預測
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img_array})

        return result[0]

# YOLO物件偵測
# 初始化YOLO模型
    def process_image_detection(self, image_path, predict_folder):
        # 實現YOLO模型的物件偵測方法
        # 導入訓練好的YOLO模型(.onnx)並進行預測，需要有預測標籤、概率
        # 最後顯示到畫面上給用戶看，並統計不同標籤的預測數量，再顯示在畫面上

        # 判斷是哪個模型並執行
        if self.image_processing_method.get() == '無強化':  # 無強化
            self.model = PATH / 'object_detection_model' / 'original' / 'best.pt'
        elif self.image_processing_method.get() == '強化牙周與加強牙齒輪廓 ':  # 高斯高通濾波器
            self.model = PATH / 'object_detection_model' / 'GH' / 'best.pt'
        elif self.image_processing_method.get() == '強化牙根與病灶點 ':  # 自適應直方圖均衡化
            self.model = PATH / 'object_detection_model' / 'AHE' / 'best.pt'
        elif self.image_processing_method.get() == '強化整體(牙周+牙根)':  # 強化整體(牙周+牙根)
            self.model = PATH / 'object_detection_model' / 'GH+AHE' / 'best.pt'
        
        model=YOLO(self.model)



        # 讀取圖片
        image = cv2.imread(image_path)
        # 執行預測
        results = model(image)
        # 獲取第一個結果
        result = results[0]

        # 繪製預測結果並儲存圖片
        if not os.path.exists(predict_folder):
                os.makedirs(predict_folder)
        #Process results list
        for result in results:
            # print(result.obb.xyxy[1][0].item())
            # result.show()  # display to screen
            result.save(predict_folder / os.path.basename(self.selected_image_path))  # save image

            # 檢查是否有偵測到物件
            if len(result.obb.cls) > 0:
                if len(result.obb.xyxy) > 0:

                    # 從左到右處理偵測物件並排序
                    sorted_indices = sorted(range(len(result.obb.cls)), key=lambda i: result.obb.xyxy[i][0].item())

                # 輸出排序後的結果
                for idx in sorted_indices:
                    obb_info = result.obb  # 直接使用 result 中的 obb_info
                    cls = obb_info.cls[idx].tolist()
                    conf = obb_info.conf[idx].tolist()
                    xyxy = obb_info.xyxy[idx].tolist()
                    xyxyxyxy = obb_info.xyxyxyxy[idx].tolist()

                    # 將資訊存入列表
                    self.predictions_list.append((cls, conf, xyxy, xyxyxyxy))

                # 輸出列表中的物件資訊
                for obj_info in self.predictions_list:
                    cls, conf, xyxy, xyxyxyxy = obj_info
                    # print(f"Class: {cls}, Confidence: {conf}, Bounding Box (x1, y1, x2, y2): {xyxy}, 4 Points: {xyxyxyxy}")
                    # print(f"Class: {cls}, Confidence: {conf}, Bounding Box (x1, y1, x2, y2): {xyxy}, 4 Points: {xyxyxyxy}")
            else:
                print("No objects detected.")

        
        # 確認長寬比，調整大小並轉換格式
        save_path =self.resize_image( PATH / 'YOLO_predict_image' / os.path.basename(self.selected_image_path),800,600, PATH / 'YOLO_predict_image', os.path.basename(self.selected_image_path))

        # 更新圖片顯示
        predict_image = PhotoImage(file= save_path)
        self.images['predict_image'] = predict_image  # 更新圖片字典中的 'predict_image'
        self.predict_image_label.configure(image=self.images['predict_image'])  # 更新圖片顯示


        self.flow_chart_label.pack_forget()
        self.result_frame.pack(side=LEFT)
        self.result_label.pack(fill=X)
        self.output_frame.pack(side=RIGHT,expand=1)
        self.output_btn1.config(text='輸出圖片')
        self.output_btn1.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn2.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn3.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn4.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)
        self.output_btn5.pack(ipady=5,pady=10,side=TOP,fill=BOTH,expand=1)

        self.update_chart()



    def chart(self,count):

        # 設置全局字體
        rcParams['font.family'] = 'Microsoft JhengHei'
        # 修復負號顯示問題
        rcParams['axes.unicode_minus']=False    

        
        rcParams['font.size'] = 22  # 全局字體大小設置

        cls = [0,1,2]
        cnt = count
        color = ['#E1BBBB', '#D55454','#FF8738']
        label = ['無病徵','根尖病灶','根尖病灶合併牙周流失']  # 'normal','apical lesion','peri-endo lesion'
        plt.barh(cls, cnt, color=color, tick_label=label, height=0.5)  # 改成 barh

        # # 調整 label 的字體大小
        # plt.tick_params(axis='y', labelsize=16)  # 設置 y 軸刻度的字體大小為 16
        # plt.tick_params(axis='x', labelsize=16)  # 設置 x 軸刻度的字體大小為 16

        # 在長條圖上顯示 count 數值
        for i, count in enumerate(count):
            plt.text(count + 0.01, i, str(count), ha='left', va='center')  # 設置文字大小為 10

         # 添加標題
        plt.title('病徵預測統計數量長條圖', fontsize=22)  # 設置標題和字體大小 #病徵預測統計數量長條圖 Symptom Prediction Count Bar Chart  #, fontsize=20

        folder = PATH / 'chart'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(PATH / 'chart' / 'chart.png',
            transparent=True,
            bbox_inches='tight',
            pad_inches=0.5)
        
        plt.clf()  # 清除當前圖形
        # plt.show() 

    def update_chart(self):

        apical_lesion_count = 0
        normal_count = 0
        peri_endo_count = 0

        if self.image_recognition_method.get()=='CNN 分類':
            # # 遍歷預測結果列表
            for prediction in self.predictions_list:
                predicted_class = prediction["predicted_class"]

                # 根據預測類別進行計數
                if predicted_class == '根尖病灶':
                    apical_lesion_count += 1
                elif predicted_class == '無病徵':
                    normal_count += 1
                elif predicted_class == '根尖病灶合併牙周流失':
                    peri_endo_count += 1

            # 輸出計數結果
            print(normal_count, apical_lesion_count, peri_endo_count)
            # 進行其他操作，比如繪製圖表等
            self.chart([normal_count, apical_lesion_count, peri_endo_count])

        elif self.image_recognition_method.get()=='YOLOv8 物件偵測':

            for i in self.predictions_list:
                for j in i:
                    if int(j)==0:
                        apical_lesion_count = apical_lesion_count + 1
                    elif int(j)==1:
                        normal_count = normal_count + 1
                    elif int(j)==2: 
                        peri_endo_count = peri_endo_count + 1
                    break

            self.chart([normal_count,apical_lesion_count,peri_endo_count])
            print(normal_count,apical_lesion_count,peri_endo_count)

        # 確認長寬比，調整大小並轉換格式
        save_path =self.resize_image(PATH / 'chart' / 'chart.png',600,300,PATH / 'chart', 'chart.png')


        # 更新圖片顯示
        single_tooth_result_image = PhotoImage(file= save_path)
        self.result_label.configure(image=single_tooth_result_image)  # 更新圖片顯示
        self.images['chart_image'] = single_tooth_result_image  # 更新圖片字典中的 'chart_image'


    # 用戶可以選擇存取文件的位置
    def choose_directory(self):
        self.choose_directory_path = filedialog.askdirectory()

    def copy_file(self,source_file, target_folder,mb_title,mb_message):
        try:
            # 確認目標資料夾存在，如果不存在則創建
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # 獲取文件名
            file_name = os.path.basename(source_file)
            
            # 目標文件路徑
            target_file = os.path.join(target_folder, file_name)

            # 複製文件
            shutil.copyfile(source_file, target_file)
            print(f"文件已成功複製到目標資料夾：{target_file}")

            Messagebox.ok(
                title=mb_title, 
                message=mb_message,
                parent=self.result_output_frame  # 將消息框置於此層上方
            )
        except Exception as e:
            print(f"複製文件時發生錯誤：{e}")

    def output_pdf(self):
        if(self.choose_directory_path!=None):
            file_path_pdf =PATH / self.choose_directory_path / f"{self.image_recognition_method.get()}_{self.image_processing_method.get()}_預測結果_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.pdf"
            self.save_to_pdf(file_path_pdf)
            Messagebox.ok(
                title='已匯出csv', 
                message='已匯出預測結果pdf，儲存於' + self.choose_directory_path + '中，可前往查看',
                parent=self.result_output_frame  # 將消息框置於此層上方
            )
        self.choose_directory_path = None


    def output_imgOrPDF(self):
        self.choose_directory()
        if(self.choose_directory_path!=None):
            if self.image_recognition_method.get() == 'CNN 分類':
                 self.output_pdf()
            else:
                self.copy_file(PATH / 'YOLO_predict_image' /os.path.basename(self.selected_image_path), self.choose_directory_path,'已匯出圖片',"已匯出預測結果圖片，儲存於" + self.choose_directory_path + '中，可前往查看')
                old_path = os.path.join(PATH, self.choose_directory_path, os.path.basename(self.selected_image_path))
                new_path = os.path.join(PATH, self.choose_directory_path, f"{self.image_recognition_method.get()}_{self.image_processing_method.get()}_預測結果_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.png")
                # 修改檔名
                os.rename(old_path, new_path)
        self.choose_directory_path = None

    def output_txt(self):
        self.choose_directory()
        if(self.choose_directory_path!=None):
            file_path_txt = PATH / self.choose_directory_path / f"{self.image_recognition_method.get()}_{self.image_processing_method.get()}_預測結果_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.txt"
            self.save_to_txt(file_path_txt)
            Messagebox.ok(
                title='已匯出txt', 
                message='已匯出預測結果txt，儲存於' + self.choose_directory_path + '中，可前往查看',
                parent=self.result_output_frame  # 將消息框置於此層上方
            )
        self.choose_directory_path = None


    def output_json(self):
        self.choose_directory()
        if(self.choose_directory_path!=None):
            file_path_json = PATH / self.choose_directory_path / f"{self.image_recognition_method.get()}_{self.image_processing_method.get()}_預測結果_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.json"
            self.save_to_json(file_path_json)
            Messagebox.ok(
                title='已匯出json', 
                message='已匯出預測結果json，儲存於' + self.choose_directory_path + '中，可前往查看',
                parent=self.result_output_frame  # 將消息框置於此層上方
            )
        self.choose_directory_path = None


    def output_csv(self):
        self.choose_directory()
        if(self.choose_directory_path!=None):
            file_path_csv =PATH / self.choose_directory_path / f"{self.image_recognition_method.get()}_{self.image_processing_method.get()}_預測結果_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.csv"
            self.save_to_csv(file_path_csv)
            Messagebox.ok(
                title='已匯出csv', 
                message='已匯出預測結果csv，儲存於' + self.choose_directory_path + '中，可前往查看',
                parent=self.result_output_frame  # 將消息框置於此層上方
            )
        self.choose_directory_path = None

    def save_to_pdf(self, output_pdf):
        c = canvas.Canvas(str(output_pdf), pagesize=letter)
        width, height = letter

        # 設定字型和大小
        c.setFont("Times-Roman", 12)

        # 寫入標題
        c.drawString(100, height - 50, "Prediction Result Report")

        # 設定初始 Y 座標位置
        y_position = height - 100

        # 逐個處理預測結果
        for idx, prediction in enumerate(self.predictions_list, start=1):
            filename = prediction["filename"]
            predicted_class = prediction["predicted_class"]
            probability = prediction["probability"]

            # 匯入圖片
            img_reader = ImageReader(PATH / 'single_tooth_processed_image' / filename)
            c.drawImage(img_reader, 120, 160,preserveAspectRatio=True,width=180)

            # 添加文字內容
            c.drawString(120, 100, f"File Name: {filename}")
            if predicted_class== '無病徵':
                c.drawString(120, 120, f"Predicted Class: normal")
            elif predicted_class== '根尖病灶合併牙周流失':
                c.drawString(120, 120, f"Predicted Class: peri-endo lesion")
            elif predicted_class== '根尖病灶':
                c.drawString(120, 120, f"Predicted Class: apical lesion")
            c.drawString(120, 140, f"Probability: {probability}")

            # 調整 Y 座標位置
            y_position -= 320

            # 每頁最多顯示 1 筆預測結果，超過時新增頁面
            if idx % 1 == 0 and idx != len(self.predictions_list):
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = height - 100

        c.save()
        print(f"PDF 檔案 '{output_pdf}' 已創建完成。")


    def save_to_txt(self, file_path):
        open(file_path, 'a').close()  # 創建空文件
        with open(file_path, 'w') as f:
            for item in self.predictions_list:
                f.write(f"{item}\n")

    def save_to_json(self, file_path):

        if self.image_recognition_method.get() == 'CNN 分類':
            open(file_path, 'a').close()  # 創建空文件
            with open(file_path, 'w') as f:
                json.dump(self.predictions_list, f)
        else:
            open(file_path, 'a').close()  # 創建空文件
            with open(file_path, 'w') as f:
                json.dump(self.predictions_list, f)

    def save_to_csv(self, file_path):
        open(file_path, 'a').close()  # 創建空文件
        if self.image_recognition_method.get() == 'CNN 分類':
            # 開啟 CSV 檔案，使用 'w' 寫入模式
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                # 定義 CSV 欄位名稱
                fieldnames = ['filename', 'predicted_class', 'probability']
                # 建立 CSV 寫入物件
                writer = csv.DictWriter(file, fieldnames=fieldnames) 
                # 寫入欄位名稱
                writer.writeheader()
                # 寫入每一筆資料
                for prediction in self.predictions_list:
                    writer.writerow(prediction)
        else:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Confidence', 'XYXY', 'XYXYXYXY'])
                writer.writerows(self.predictions_list)


    def clear_folder(self,folder_path):
        # 檢查目標路徑是否存在
        if not os.path.exists(folder_path):
            print(f"目錄 '{folder_path}' 不存在.")
            return

        # 確認目標路徑是資料夾
        if not os.path.isdir(folder_path):
            print(f"'{folder_path}' 不是資料夾.")
            return

        # 使用 shutil.rmtree 刪除資料夾及其內容
        try:
            shutil.rmtree(folder_path)
            print(f"已刪除資料夾及其內容: {folder_path}")
        except Exception as e:
            print(f"刪除失敗: {folder_path}, 錯誤: {e}")

    def clean_data(self):
        # self.model
        for i in range(10): 
            self.frames_dict[i+1].pack_forget()
            self.labels_dict[f'f{i+1}_lb1'].pack_forget()
            self.labels_dict[f'f{i+1}_lb2'].pack_forget()
            self.labels_dict[f'f{i+1}_lb3'].pack_forget()
            self.image_dict[f'f{i+1}'].pack_forget()

        #
        # 更新按鈕文字為 "選擇檔案"
        self.lbl.config(text='選擇檔案')
            

        # 更新圖片顯示
        predict_image = PhotoImage(file= PATH / 'photo' / 'not_detect_2.png')
        self.images['predict_image'] = predict_image  # 更新圖片字典中的 'predict_image'
        self.predict_image_label.configure(image=self.images['predict_image'])  # 更新圖片顯示
        self.predict_image_label.pack(fill="y")

        # 更新圖片顯示
        single_tooth_predict_image = PhotoImage(file= PATH / 'photo' / 'no_file_2.png')
        self.images['PA_file'] = single_tooth_predict_image  # 更新圖片字典中的 'PA_file'
        self.image_label.configure(image=self.images['PA_file'])  # 更新圖片顯示

        #進度條隱藏
        self.pb.pack_forget()
        self.progressbar.pack_forget()
        self.pb_lb.pack_forget()
        

    
        self.result_label.pack_forget()
        self.result_frame.pack_forget()
        self.result_label.pack_forget()
        self.output_frame.pack_forget()
        self.output_btn1.pack_forget()
        self.output_btn2.pack_forget()
        self.output_btn3.pack_forget()
        self.output_btn4.pack_forget()
        self.output_btn5.pack_forget()

        
        # 更新圖片顯示
        self.flow_chart_label.pack(fill="both")

        
        self.clear_folder(PATH / 'adjusted_select_image')

        self.clear_folder(PATH / 'single_tooth')
        self.clear_folder(PATH / 'single_tooth_bbox_info')
        self.clear_folder(PATH / 'single_tooth_predict')
        self.clear_folder(PATH / 'single_tooth_processed_resize_padding_image')
        self.clear_folder(PATH / 'single_tooth_resize_padding_image')
        self.clear_folder(PATH / 'single_tooth_resize_padding_processed_image')
        self.clear_folder(PATH / 'single_tooth_rotated_images')
        self.clear_folder(PATH / 'single_tooth_processed_image')
        self.clear_folder(PATH / 'select_image')
        self.clear_folder(PATH / 'CNN_resize_image')
        self.clear_folder(PATH / 'YOLO_predict_image')
        self.clear_folder(PATH / 'PA_processed_image')

        self.clear_folder(PATH / 'chart')


        self.selected_image_path=None
        self.predictions_list = []
        # 初始化儲存預測結果的字典
        self.predictions_per_tooth = {}

        self.update()  # 立即更新視窗畫面

        Messagebox.ok(
            title='刪除成功', 
            message='刪除成功！',
            parent=self.result_output_frame  # 將消息框置於此層上方
            )







if __name__ == '__main__':

    app = ttk.Window(title="牙齒病徵辨識系統", themename="flatly",iconphoto='', size=[1400,1100], position=None, minsize=None, maxsize=None, resizable=None, hdpi=True, scaling=None, transient=None, overrideredirect=False, alpha=1.0)
    app.place_window_center()



    default_font = nametofont("TkDefaultFont")
    default_font.configure(size=12,weight='bold') #

    # button_style = ttk.Style()
    # button_style.configure('my.TButton',font=('Helvetica',16))
    # labelframe_style = ttk.Style()
    # labelframe_style.configure('my.TLabelframe',font=('Helvetica',20))

    ToothRecognitionSystem(app)

    app.mainloop()
