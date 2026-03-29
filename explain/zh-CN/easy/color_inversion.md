这个 Triton 程序实现了一个**GPU 加速的图像反色（Invert Colors / 负片）滤镜**。

它的核心亮点在于：它专门针对 **RGBA 格式**（包含红、绿、蓝和透明度四个通道）的图像进行处理，并且在反转 RGB 颜色的同时，**巧妙地利用掩码（Mask）保留了 Alpha 透明度通道不变**。图像的修改是**原地（In-place）**进行的，直接覆盖原始张量，节省了内存。

以下是代码的逐行详细解析：

### 1. 启动函数：`solve` (Host 端)
这个函数运行在 CPU 上，负责配置并启动 GPU 上的计算任务。

* **`BLOCK_SIZE = 1024`**: 定义了每个 GPU 线程块（Block）负责处理 **1024 个像素**。由于每个像素有 4 个通道（RGBA），每个 Block 实际上处理 4096 个数据点。
* **`n_pixels = width * height`**: 计算图像的总像素数。
* **`grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)`**: 计算需要启动多少个 Block 才能覆盖整张图片。`triton.cdiv` 是向上取整除法（Ceiling Division）。例如，如果总共有 1500 个像素，它会启动 2 个 Block 来确保所有像素都被处理。
* **`invert_kernel[grid](...)`**: 以一维网格的结构，将图像指针、宽高和块大小传递给 GPU 核函数并启动执行。

### 2. 核心逻辑：`invert_kernel` (GPU 核函数)
带有 `@triton.jit` 装饰器的函数会被编译成高效的 GPU 机器码。

* **`image_ptr = image_ptr.to(tl.pointer_type(tl.uint8))`**
    将传入的张量指针强制转换为无符号 8 位整数（`uint8`）指针。因为标准的图像像素值范围是 0-255，每个通道正好占用 1 个字节。
* **`pid = tl.program_id(axis = 0)`**
    获取当前正在执行的线程块在网格中的 ID（一维索引）。
* **`offsets = pid * BLOCK_SIZE * 4 + tl.arange(0, BLOCK_SIZE * 4)`**
    计算当前 Block 需要处理的内存偏移量。
    * 为什么要 `* 4`？因为图像大概率是 RGBA 格式，**1 个像素 = 4 个通道（字节）**。
    * `tl.arange` 生成一个从 0 到 4095 的向量，代表这个 Block 内的所有通道索引。
* **`mask = (offsets < width * height * 4) & (offsets % 4 != 3)`**
    这是程序中最精妙的一行，定义了两个掩码条件：
    1.  **防止越界 (`offsets < ...`)**: 防止最后一个 Block 处理超出实际图像内存的数据。
    2.  **忽略 Alpha 通道 (`offsets % 4 != 3`)**: RGBA 的索引周期是 0(R), 1(G), 2(B), 3(A)。`% 4 != 3` 意味着**只处理余数为 0, 1, 2 的红绿蓝通道**，完全跳过 Alpha 通道，从而保留了图像的透明度信息。
* **`image = tl.load(image_ptr + offsets, mask=mask)`**
    根据计算出的偏移量和掩码，从 GPU 全局内存中将 RGB 像素数据批量加载到超高速的 SRAM/寄存器中。被 mask 掉的 Alpha 通道不会被加载（或者被赋予默认值）。
* **`image = 255 - image`**
    执行核心的**反色计算**。例如，纯黑 (0) 变成纯白 (255)，红色 (255, 0, 0) 变成青色 (0, 255, 255)。
* **`tl.store(image_ptr + offsets, image, mask=mask)`**
    将计算完的反色数据**写回**原来的内存地址。同样依靠掩码，只覆盖 RGB 通道，原始内存中的 Alpha 通道数据不会被这步操作修改。

---

**总结**：这是一种极其高效的底层图像处理方式。相比于直接用 PyTorch 写 `image[:, :, :3] = 255 - image[:, :, :3]`，使用 Triton 可以更精细地控制 GPU 内存的读写行为（合并访存），在处理超大分辨率图像时能达到极致的带宽利用率。

你想了解如何编写一个测试脚本来实际运行并验证这段 Triton 代码的效果吗？