import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from Unet_principle import Unet
import data
from torch.utils.tensorboard import SummaryWriter

# 模型训练
def train(orig_images_path, haze_images_path, batch_size, epochs):
    # 数据加载
    train_dataset = data.dehazing_loader(orig_images_path, haze_images_path, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 定义模型
    dehaze_net = Unet(3,3).cuda()
    #dehaze_net.apply(weights_init)
    # 定义损失函数 criterion
    criterion = nn.MSELoss().cuda()    #Mean Squared Error 均方误差函数
    # 定义优化器 optimizer
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=1e-3, weight_decay=1e-4)   # dehaze_net返回模型的全部参数，将它们传入Adam函数构造出一个Adam优化器对象optimizer
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # 混合精度训练工具
    scaler = GradScaler()

    # TensorBoard日志记录
    writer = SummaryWriter(log_dir='./logs')

    # 指定为train模式
    dehaze_net.train()
    for epoch in range(epochs): # 迭代训练
        epoch_loss = 0  # 每轮迭代将loss置0
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            # 目标 + 计算网络输出 -> 计算损失
            img_orig = img_orig.cuda()                     # 目标图像：GT
            img_haze = img_haze.cuda()
            with autocast():
                clean_image = dehaze_net(img_haze)      # 计算网络输出：去雾后的图像
                loss = criterion(clean_image, img_orig) # 计算损失值loss

            # 计算梯度 -> 反向传播 -> 更新参数
            optimizer.zero_grad()   # 将上轮已更新的参数梯度清0，防止影响该轮的参数更新
            scaler.scale(loss).backward()         # 反向计算出个参数的梯度
            scaler.unscale_(optimizer)            # 取消缩放以便梯度裁剪
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), 0.1) # 梯度裁剪
            scaler.step(optimizer)        # 更新网络中的参数
            scaler.update()               # 更新缩放器

            epoch_loss += loss.item()   # 记录损失函数值
        avg_loss = epoch_loss / len(train_loader)   # 每个epoch结束后计算平均损失
        print('EPOCH : %04d  LOSS : %2.3f' % (epoch, avg_loss)) # 终端输出损失信息
        writer.add_scalar('Loss/train', avg_loss, epoch)   # 记录损失到TensorBoard

        # 更新学习率
        scheduler.step()

        # 模型保存
        torch.save(dehaze_net.state_dict(), './saved_models_bilinear/dehaze_net_epoch_%d.pth' % epoch)

    writer.close()

if __name__ == '__main__':
    orig_images_path = 'train_image/gt/'
    haze_images_path = 'train_image/haze/'
    batch_size = 1  # 单次传递给程序用以训练的数据个数
    epochs = 30     # 30轮迭代训练
    train(orig_images_path, haze_images_path, batch_size, epochs)
