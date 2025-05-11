from torch.utils import data
from PIL import Image
import torch
import numpy as np
import cv2         
import random      

# --- 数据增强函数 ---
def random_erasing(img_np, p=0.5, sl=0.02, sh=0.2, r1=0.3, r2=1/0.3, attempts=20):
    """
    img_np: HWC BGR NumPy 数组
    """
    if random.random() > p:
        return img_np
    img_h, img_w, img_c = img_np.shape
    area = img_h * img_w
    for _ in range(attempts):
        erase_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, r2)
        h = int(round(np.sqrt(erase_area * aspect_ratio)))
        w = int(round(np.sqrt(erase_area / aspect_ratio)))
        if w < img_w and h < img_h:
            x1 = random.randint(0, img_w - w)
            y1 = random.randint(0, img_h - h)
            img_erased = img_np.copy()
            img_erased[y1:y1+h, x1:x1+w, :] = 0 # 涂黑
            return img_erased
    return img_np

def gaussian_blur(img_np, p=0.5, max_kernel_size=5):
    if random.random() > p:
        return img_np
    kernel_size = random.choice(list(range(1, max_kernel_size + 1, 2)))
    if kernel_size == 1: return img_np
    img_blurred = cv2.GaussianBlur(img_np.copy(), (kernel_size, kernel_size), 0)
    return img_blurred

def add_gaussian_noise(img_np, p=0.5, mean=0, std_dev_max=25):
    if random.random() > p:
        return img_np
    std_dev = random.uniform(0, std_dev_max)
    noise = np.random.normal(mean, std_dev, img_np.shape)
    img_noisy = img_np.astype(np.float32) + noise
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    return img_noisy

def pixel_jitter(img_np, p=0.5, max_jitter_strength=10):
    if random.random() > p:
        return img_np
    jitter_strength = random.uniform(-max_jitter_strength, max_jitter_strength)
    img_jittered = img_np.astype(np.float32) + jitter_strength
    img_jittered = np.clip(img_jittered, 0, 255).astype(np.uint8)
    return img_jittered

def apply_custom_augmentations(image_pil):
    """
    对 PIL Image 应用自定义增强。
    image_pil: 输入的 PIL Image 对象。
    返回: 增强后的 PIL Image 对象。

    MODIFICATION: 这是我们添加的核心增强应用函数。
    """
    # 1. 将 PIL Image 转换为 OpenCV NumPy 数组 (BGR)
    # PIL Image 通常是 RGB，OpenCV 通常是 BGR
    img_np = np.array(image_pil)
    if len(img_np.shape) == 2: # 如果是灰度图
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR) # 转换为 BGR 以便处理
    elif img_np.shape[2] == 3: # RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif img_np.shape[2] == 4: # RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR) # 转换为 BGR，丢弃 alpha 通道
                                                          # 如果增强函数需要 alpha，则需相应修改

    augmented_image_np = img_np.copy() # 创建副本以进行增强

    # --- 可自定义部分：选择和配置数据增强 ---
    # 你可以在这里选择应用哪些增强、它们的顺序以及各自的参数。
    # p 参数控制每种增强应用的概率。
    # 其他参数控制增强的强度或行为。

    # 示例：应用随机擦除 (Cutout)
    # 说明：模拟图像中的遮挡或裁块。
    # 参数：
    #   p: 应用此增强的概率。
    #   sl, sh: 擦除区域相对于图像总面积的最小和最大比例。
    #   r1, r2: 擦除区域的最小和最大纵横比。
    augmented_image_np = random_erasing(augmented_image_np, p=0.3, sh=0.15) # 示例参数

    # 示例：应用高斯模糊
    # 说明：平滑图像，可能有助于模型对轻微模糊的鲁棒性。
    # 参数：
    #   p: 应用此增强的概率。
    #   max_kernel_size: 高斯核的最大尺寸（必须是奇数）。核越大，模糊越强。
    augmented_image_np = gaussian_blur(augmented_image_np, p=0.3, max_kernel_size=3) # 示例参数

    # 示例：应用高斯噪声
    # 说明：向图像添加随机噪声，模拟传感器噪声或图像质量下降。
    # 参数：
    #   p: 应用此增强的概率。
    #   mean: 噪声的均值（通常为0）。
    #   std_dev_max: 噪声标准差的最大值（从0到此值之间随机选择）。
    augmented_image_np = add_gaussian_noise(augmented_image_np, p=0.3, std_dev_max=15) # 示例参数
    
    # 示例：应用像素抖动
    # 说明：对所有像素值添加一个小的随机偏移，轻微改变图像亮度/对比度。
    # 参数：
    #   p: 应用此增强的概率。
    #   max_jitter_strength: 抖动强度的最大值（从 -max 到 +max 之间随机选择）。
    augmented_image_np = pixel_jitter(augmented_image_np, p=0.2, max_jitter_strength=5) # 示例参数
    
    # 你可以添加更多的增强方法，例如：
    # - 随机水平翻转: if random.random() < 0.5: augmented_image_np = cv2.flip(augmented_image_np, 1)
    # - 颜色抖动 (亮度、对比度、饱和度、色调): 这通常通过 torchvision.transforms.ColorJitter 实现，
    #   如果在这里用 OpenCV 实现会更复杂，但也是可能的。
    #   如果使用 torchvision 的 ColorJitter，应在转换为 Tensor 之后，或直接将 PIL Image 传入。
    #   但我们这里的流程是在转换为 Tensor 之前用 OpenCV 处理。
    # --- 结束：可自定义部分 ---

    # 2. 将增强后的 OpenCV NumPy 数组 (BGR) 转换回 PIL Image (RGB)
    augmented_image_pil = Image.fromarray(cv2.cvtColor(augmented_image_np, cv2.COLOR_BGR2RGB))

    return augmented_image_pil


class Dataset_Csv(data.Dataset):
    "Characterizes a dataset for PyTorch"

    # MODIFICATION: 添加了 augment_after_epoch 和 get_current_epoch_fn 参数
    def __init__(self, folders, labels, transform=None,
                 should_augment=False,
                 augment_after_epoch=0,      # 从第几个 epoch 之后开始增强 (0 表示从第一个 epoch 就增强)
                 get_current_epoch_fn=None): # 一个返回当前 epoch 的函数
        "Initialization"
        self.labels = labels
        self.folders = folders
        # self.transform = transform # 旧的 transform 处理方式
        self.should_augment = should_augment
        self.augment_after_epoch = augment_after_epoch
        self.get_current_epoch_fn = get_current_epoch_fn

        # MODIFICATION: 根据 should_augment 和 transform 的类型来设置 transform
        # 假设 transform 是一个字典，包含 'train' 和 'val' 的键
        if isinstance(transform, dict):
            if self.should_augment: # 通常用于训练集
                self.transform_to_apply = transform.get('train')
            else: # 通常用于验证集
                self.transform_to_apply = transform.get('val')
        else: # 如果 transform 不是字典，则直接使用它（旧的行为）
            self.transform_to_apply = transform

        if self.should_augment and self.get_current_epoch_fn is None:
            # 如果启用了按epoch增强，但没有提供获取epoch的函数，则发出警告或抛出错误
            # 这里我们选择总是应用增强，如果get_current_epoch_fn未提供，则augment_after_epoch条件不生效
            print("Warning: `get_current_epoch_fn` is None. Augmentation will be applied from epoch 0 if `should_augment` is True.")


    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path):
        image = Image.open(path).convert('RGB')
        return image

    def __getitem__(self, index):
        "Generates one sample of data"
        folder_path = self.folders[index]
        X_pil = self.read_images(folder_path)

        apply_aug = False
        if self.should_augment:
            if self.get_current_epoch_fn is not None:
                current_epoch = self.get_current_epoch_fn()
                if current_epoch >= self.augment_after_epoch:
                    apply_aug = True
            else:
                # 如果没有 epoch 函数，但 should_augment 为 True，则总是应用增强
                # （或者你可以选择在这种情况下不增强，具体取决于你的逻辑）
                apply_aug = True # 默认行为：如果开启了增强且无epoch函数，则增强

        if apply_aug:
            X_pil = apply_custom_augmentations(X_pil)

        if self.transform_to_apply is not None:
            X_tensor = self.transform_to_apply(X_pil)
        else:
            # 至少需要 ToTensor，如果 transform 为 None
            from torchvision import transforms # 仅在此处需要时导入
            X_tensor = transforms.ToTensor()(X_pil)


        y = torch.FloatTensor([self.labels[index]])
        return X_tensor, y