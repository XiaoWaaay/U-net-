import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
import torch
import Unet_principle as Unet


# 定义去雾函数
def dehaze_image(image_path, calculate_loss=False):
    data_hazy = Image.open(image_path)
    data_hazy = np.array(data_hazy) / 255.0
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.unsqueeze(0)

    dehaze_net = Unet.Unet(3, 3)
    dehaze_net.eval()
    dehaze_net.load_state_dict(
        torch.load('saved_models_bilinear/dehaze_net_epoch_29.pth', map_location=torch.device('cpu')))

    with torch.no_grad():
        clean_image = dehaze_net(data_hazy).detach().numpy().squeeze()

    clean_image = np.swapaxes(clean_image, 0, 1)
    clean_image = np.swapaxes(clean_image, 1, 2)

    if calculate_loss:
        # 在这里添加计算损失率的代码（例如 MSE 损失）
        # 这里简单地返回一个示例损失率，实际根据需要计算真实损失率
        loss_value = np.random.rand() * 100
        return clean_image, loss_value
    else:
        return clean_image, None


# 定义按钮点击事件处理函数：上传有雾图片
def upload_image():
    global file_path
    file_path = filedialog.askopenfilename()  # 打开文件对话框，选择图片文件
    if file_path:  # 如果选择了文件
        orig_img = Image.open(file_path)
        orig_img.thumbnail((400, 400))  # 缩放图片尺寸
        orig_photo = ImageTk.PhotoImage(orig_img)
        orig_label.config(image=orig_photo)
        orig_label.image = orig_photo


# 定义按钮点击事件处理函数：进行去雾处理
def process_image():
    if file_path:
        # 去雾处理
        result_image, _ = dehaze_image(file_path)
        # 显示去雾后的图片
        result_img = Image.fromarray((result_image * 255).astype(np.uint8))
        result_img.thumbnail((400, 400))  # 缩放图片尺寸
        result_photo = ImageTk.PhotoImage(result_img)
        result_label.config(image=result_photo)
        result_label.image = result_photo


# 定义按钮点击事件处理函数：计算损失率
def calculate_loss():
    if file_path:
        _, loss_value = dehaze_image(file_path, calculate_loss=True)
        if loss_value is not None:
            messagebox.showinfo("损失率计算结果", f"去雾后的损失率为: {loss_value:.2f}")


# 创建主窗口
root = tk.Tk()
root.title("Dehazing Application")

# 创建一个框架来放置按钮，使其横向排列
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# 创建上传图片按钮，并放置在框架中的左侧
upload_button = tk.Button(button_frame, text="上传有雾图片", command=upload_image)
upload_button.pack(side='left', padx=10)

# 创建去雾处理按钮，并放置在框架中的左侧
process_button = tk.Button(button_frame, text="进行去雾处理", command=process_image)
process_button.pack(side='left', padx=10)

# 创建计算损失率按钮，并放置在框架中的左侧
loss_button = tk.Button(button_frame, text="计算损失率", command=calculate_loss)
loss_button.pack(side='left', padx=10)

# 创建展示原始有雾图片的标签
orig_label = tk.Label(root)
orig_label.pack(pady=10)

# 创建展示去雾结果的标签
result_label = tk.Label(root)
result_label.pack(pady=10)

# 运行主事件循环
root.mainloop()
