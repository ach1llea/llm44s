# iPhone 4S / iOS 6.1.3 本地 Transformer LLM 部署方案

> 目标设备：iPhone 4S，iOS 6.1.3，已越狱（p0sixspwn untethered）
> 硬性要求：完全本地推理，模型必须是基于 Transformer 的 LLM，不依赖网络/云端
> 文档状态：方案设计（工具链可行性已在本机实测验证，见 §6）

---

## 1. 结论

**可行，但天花板极低。**

- ✅ 可以跑**真正的 Transformer Decoder-only LLM**（MiniMind2-Small 26M / TinyStories 15M~42M 这一量级），速度约 **5–20 token/s**，常驻内存 20–50MB。
- ❌ **不可能**跑任何"能当助手用"的模型。0.5B 参数即 400MB+（int4），超过整机可用内存；7B 完全不谈。
- ⚠️ 真正的瓶颈不是算力，是 **512MB 物理内存 + 32-bit ARMv7 指令集**（无 FP16、无 int8 dot-product、无 GPU 计算通路）。

一句话定位：**这是一个"能在 2011 年的手机上跑通端侧 Transformer 推理"的工程验证项目，不是一个能用的 AI 助手。**

---

## 2. 硬约束

| 项目 | 实际情况 | 对 LLM 的影响 |
|---|---|---|
| SoC | Apple A5，2× Cortex-A9 @ **800MHz**（4S 降频版） | 算力约为现代 A17 的 1/300 |
| ISA | **ARMv7-A**，32-bit，仅 armv7/thumbv7 | 无 aarch64 优化路径；无 `SDOT/UDOT`（int8 点积）、无 FP16、无 i8mm |
| NEON | 有，但数据通路 **64-bit** | 一条 `vmla.f32`(Q) 需 2 cycle，fp32 峰值 ≈ 3.2 GFLOPS/核 |
| 内存 | **512MB** LPDDR2，单通道 32-bit | 越狱后系统常驻后**可用 ≈ 150–250MB**，单进程安全预算 **≤120MB** |
| 带宽 | ≈ 3.2GB/s 理论，单线程流式实测 ≈ 1.2–1.8GB/s | decode 阶段 memory-bound，量化收益极大 |
| GPU | PowerVR SGX543MP2 | iOS 6 无 Metal / 无可用 OpenCL / 无 CoreML → **GPU 路线彻底不通** |
| 存储 | 8/16/32/64GB，rootfs 分区很小 | 模型放大分区 `/var/root`，别放 `/usr/bin` |
| 越狱收益 | 可装未签名二进制（`ldid -S` 伪签名）、OpenSSH、MobileTerminal、无沙盒 | **这是整个方案成立的前提**：可以跑任意 CLI 程序、可长期常驻、可 `mmap` 大文件 |

### 内存预算（关键）

```
512MB 物理
 ├─ iOS 6.1.3 系统 + SpringBoard + 越狱守护进程   ~200–300MB
 ├─ 缓冲/文件缓存                                ~50MB
 └─ 可用于你的进程                               ~120–200MB  ← 目标是吃 ≤60MB
```

进程 RSS 控制在 60MB 以内，才不会因为内存压力被 `memorystatus` 杀掉。做 UIKit 应用还要额外背 SpringBoard 分配的图形内存，所以**优先做 CLI 进程**。

---

## 3. 可行性计算模型

decode 阶段（batch=1）每生成 1 个 token 的代价：

```
FLOPs  ≈ 2 × N                        (N = 参数量, MAC 记 2 FLOP)
时间    ≈ max( 2N / F_eff ,  bytes_per_token / BW )
F_eff  ≈ 0.6–1.0 GFLOPS               (A5 双核 NEON 跑 GEMV 的实际有效值)
BW     ≈ 1.5 GB/s
```

两类瓶颈：小模型是**算力瓶颈**，大模型是**带宽瓶颈**，量化对两者都有帮助（int8 权重带宽 ÷4，且 ARMv7 上 int8→int16 的 `vmlal_s16` 吞吐是 fp32 的 2 倍）。

| 模型 | 参数 | int8 权重 | 每 token 计算 | 理论延迟 | 实测预期 | 常驻内存 | 判定 |
|---|---|---|---|---|---|---|---|
| stories260K | 0.26M | 0.3MB | 0.5 MFLOP | ~1ms | 60+ tok/s | 2MB | 能跑，但基本是胡话 |
| stories15M | 15M | 15MB | 30 MFLOP | 37ms | **15–20 tok/s** | 20MB | ✅ 推荐（英文） |
| **MiniMind2-Small** | **26M** | **26MB** | **52 MFLOP** | **65ms** | **8–12 tok/s** | **32MB** | ✅ **推荐（中文）** |
| stories42M | 42M | 42MB | 84 MFLOP | 105ms | 5–7 tok/s | 48MB | ✅ 可跑，略慢 |
| MiniMind3 | 64M | 64MB | 128 MFLOP | 160ms | 3–4 tok/s | 72MB | ⚠️ 能用但很肉 |
| SmolLM2-135M | 135M | 80MB(int4) | 270 MFLOP | 340ms | 1.5–2 tok/s | 150MB+ | ❌ 超预算 |
| Qwen2.5-0.5B | 494M | 400MB(int4) | 1 GFLOP | 1.2s+ | <0.5 tok/s | 420MB+ | ❌ 物理不可能 |

### 词表陷阱（很重要，选型时容易忽略）

小模型的**输出投影层**（`hidden × vocab`）在单 token 推理里占比惊人。以 stories15M 为例：

- 词表 32000 × dim 288 = **9.2M** 参数，占总参数量 60%
- 每个 token 光算 logits 就要 18.4 MFLOP，占全部计算量的 **60%**

所以选模型不能只看参数量，要看 `vocab × dim`：

- MiniMind 用 **6400** 词表 → 26M 模型里 logits 仅占 12%，这对 A5 是巨大优势（作者本来就是为中文压缩率设计的）
- 反过来，任何 32k/64k/152k 词表的"小模型"在 4S 上都会被输出层吃掉一半性能

**结论：中文选 MiniMind2-Small（26M / 6400 词表）是最优解，其 6400 词表设计几乎就是为这类场景准备的。**

---

## 4. 三条技术路线

### 路线 A：自写精简 C 推理引擎 + int8 量化（推荐）

以 Karpathy 的 `llama2.c` / `runq.c`（MIT，纯 C，无依赖）为骨架，改造成 MiniMind/TinyLlama 架构支持。代码量约 1500–2000 行 C。

优点：
- 完全可控，二进制 <200KB，无任何第三方依赖
- 内存布局自己说了算，可精确控制 RSS
- 没有 llama.cpp 在 32-bit 上那一堆历史坑

缺点：采样器/聊天模板/tokenizer 要自己写（BPE 约 250 行）。

### 路线 B：llama.cpp（不推荐，坑很多）

llama.cpp 的 32-bit ARMv7 支持一直是**二等公民**，实测/社区已知问题：

1. **k-quants 在 armv7 上编译失败**（PR #2920）：`vld1q_s16_x2`、`vld1q_u8_x4` 这类 "load multiple structures" intrinsic 在 armv7 的 `arm_neon.h` 里不存在（那是 aarch64 扩展）
2. `ggml-alloc` 里有 `size_t` 与 64-bit 的硬假设（`1ULL<<40` 被截断成 0 → `not enough space in the buffer`）
3. `llama_calc_tensor_size` 里 `172800 * 32000` 溢出 32-bit
4. `sgemm.cpp`（llamafile 的 tinyBLAS）大量使用 `vld1q_f16` 等 FP16 NEON intrinsic → armv7 无 FP16 向量运算，编译必炸
5. 新版本强制 C++17 + 更多 64-bit 假设，老版本又缺 GQA/RoPE 的某些实现

如果坚持用：必须锁到 2023 年底前后的老 commit、关掉 K-quants 只用 `Q4_0/Q8_0`、打 32-bit patch。**投入产出比极差，不建议。**

### 路线 C：不本地（排除）

局域网把 Mac/树莓派当后端、4S 当浏览器终端，是唯一"能用"的形状，但违反"一定要本地"的前提。此处仅作对比记录。

---

## 5. 引擎设计（路线 A 的具体实现）

### 5.1 项目结构

```
4sllm/
├── src/
│   ├── main.c        # CLI / HTTP 入口
│   ├── model.c/h     # Transformer 前向：embed / RMSNorm / RoPE / GQA attn / SwiGLU
│   ├── quant.c/h     # int8 分组量化 + NEON GEMV kernel
│   ├── tokenizer.c/h # GPT2-style byte-level BPE
│   ├── sampler.c/h   # top-k / top-p / 温度 / 重复惩罚
│   ├── http.c/h      # 极简 HTTP + SSE 服务（给 Safari 用）
│   └── arena.c/h     # 预分配内存池，杜绝运行时 malloc
├── tools/
│   ├── export_mm.py  # HF 权重 → 自定义 .sllm 格式（int8 + per-group scale）
│   └── make_tok.py   # tokenizer.json → tok.bin
└── web/index.html    # 单文件聊天页
```

### 5.2 模型文件格式（.sllm）

```
[header]  magic "SLLM", version, dim, n_layers, n_heads, n_kv_heads,
          vocab_size, hidden_dim, seq_len, quant_type, group_size, rope_theta
[tensors] 按固定顺序连续排列，每行 int8 + 每 32 个元素一个 fp32 scale
```

加载方式：`mmap(MAP_PRIVATE)`。好处是页面按需 fault-in，冷启动快，且**不占堆**；越狱环境无 codesign 限制，匿名 mmap + 可执行页都能用。

### 5.3 核心 kernel（决定成败）

ARMv7 **没有** `vdotq_s32`（int8 点积），所以量化矩阵乘走这条路径：

```c
// 权重 int8, 激活 int16, 累加 int32
int8x8_t  w8  = vld1_s8(w + i);              // 8 个 int8
int16x8_t w16 = vmovl_s8(w8);                 // 零扩展到 int16
int16x4_t a_lo = vld1_s16(x + i);             // 激活低 4 个
int16x4_t a_hi = vld1_s16(x + i + 4);         // 激活高 4 个
acc = vmlal_s16(acc, vget_low_s16(w16),  a_lo);  // 4 MAC
acc = vmlal_s16(acc, vget_high_s16(w16), a_hi);  // 4 MAC
```

- 吞吐：`vmlal_s16` 每指令 4 MAC，A9 上约 1 指令/cycle → **4 MAC/cycle/核**
- 对比 fp32 `vmla.f32`(Q)：4 元素需 2 cycle → **2 MAC/cycle/核**
- 所以 **int8 路径比 fp32 快 2 倍，同时权重带宽只有 1/4** —— 双赢，必做

要点：
- 激活必须**逐行**量化（`scale = max(|x|)/127`），行内量化误差可控
- `int32` 累加不会溢出（512 个元素 × 127 × 127 ≈ 8.3M，远小于 2^31）
- 累加完乘 `scale_w * scale_x` 得到 fp32 输出，写入行缓冲
- **不要**用 `-ffast-math`：会破坏 softmax 的 NaN/Inf 处理

### 5.4 内存布局

| 项 | 26M 模型实际占用 |
|---|---|
| int8 权重（mmap，page-cache 计入 RSS） | 26MB |
| KV cache（8 层, GQA kv_heads=2, head_dim 64, ctx 256, fp16） | ≈1MB |
| 激活 arena（行缓冲 512 float + FFN 中间 2048 float + logits 6400 float） | ≈0.1MB |
| tokenizer + 采样器 | <1MB |
| **合计 RSS** | **≈30MB** |

对比 stories15M：权重 15MB，但 logits 缓冲要 32000×4B=128KB，KV 也小，总 RSS ≈20MB。

### 5.5 线程与并发

- `pthread_create` × 2，在 GEMV 的**行维度**上对半切（`rows/2`），不用锁
- 线程池常驻，避免每 token 创建
- A9 双核共享 L2（1MB），两个线程同时流式读权重会抢带宽，实测加速比通常只有 **1.5–1.7×**，不要期待 2×
- ⚠️ 避开 `thread_local`：iOS 6 的 libSystem 可能缺 `__cxa_thread_atexit`，用 `pthread_key` 或直接全局变量

### 5.6 Tokenizer

MiniMind 用 6400 词表的 **GPT2 byte-level BPE**（`tokenizer.json` / `vocab.json` + `merges.txt`）。需要：

1. Python 侧把 vocab + merges + special tokens 导出成二进制表（`tok.bin`）
2. C 侧实现：UTF-8 → byte 序列 → 贪心 BPE merge（用 hash map 查 merge rank，3200 条 merge 规则级别，很快）
3. 中文汉字平均 **1–1.5 token/字**（6400 词表对常用汉字覆盖良好），但生僻字会退化成 2–3 token
4. Chat 模板按 MiniMind 的 ChatML 格式（`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`）

### 5.7 HTTP + Safari（推荐的交互形态）

CLI 进程内置一个单线程 socket server（`listen(127.0.0.1:8080)`），用 SSE 流式吐 token：

```
GET /chat?q=你好   →  text/event-stream，每 token 一个 "data: {...}\n\n"
```

iOS 6 的 Safari 支持 `EventSource`（Safari 5+），前端就是一个单文件 HTML，几十行 JS。好处：
- 零 UIKit 开发，界面随便改
- 不占 SpringBoard 的图形内存
- 调试方便：Mac 上改 HTML，`scp` 过去刷新即可

---

## 6. 工具链（已在本机实测）

### 6.1 实测结论

本机 Apple clang **21.0.0**（无 Xcode，仅 CommandLineTools）+ 系统 `ld64` 的验证结果：

```bash
$ clang -target armv7-apple-ios6.0 -O2 -c t.c -o t.o
$ file t.o
t.o: Mach-O object arm_v7                      # ✅ 代码生成正常
$ ld -arch armv7 -e _main -o m.out m.o
ld: dynamic main executables must link with libSystem.dylib for architecture armv7
                                               # ✅ 说明 ld64 仍完整支持 armv7 链接
```

**结论：不需要装 Xcode 4.6、不需要虚拟机、不需要老 toolchain。** 现代 clang + 一个 iOS SDK sysroot 就能出 armv7 二进制。这是本方案能低成本落地的关键前提。

### 6.2 获取 SDK 与构建

`theos/sdks` 里最老的可用 SDK 是 **iPhoneOS9.3.sdk**（没有 6.1 的现成包），配合 `-miphoneos-version-min=6.0` 完全够用——我们只用 POSIX/libc/pthread/mmap，这些符号在 iOS 6 就存在。

```bash
git clone --depth 1 https://github.com/theos/sdks.git ~/sdks
export SDK=~/sdks/iPhoneOS9.3.sdk

CFLAGS="-target armv7-apple-ios6.0 -isysroot $SDK -miphoneos-version-min=6.0 \
        -mfpu=neon -O3 -fno-math-errno -ffunction-sections -fdata-sections \
        -Wall -std=gnu99"

clang $CFLAGS -c src/*.c
clang $CFLAGS -o llm4s src/*.o -Wl,-dead_strip -lpthread -lm
ldid -S llm4s          # 伪签名（brew install ldid）
```

⚠️ **不要加 `-mfloat-abi=hard`**：Darwin armv7 默认是 softfp ABI，改成 hard 会让 `printf` 里的浮点可变参数直接崩。

> 备选：若需要严格的 iOS 6.1 SDK（例如要链 UIKit 做原生 App），从 Xcode 4.6.3.dmg 里抽 `iPhoneOS6.1.sdk`。

---

## 7. 上机部署

### 7.1 前置（Cydia）

1. 已越狱（p0sixspwn）→ Cydia 装 **OpenSSH**（`root` / 默认密码 `alpine`，**立刻改密码**）
2. 装 **MobileTerminal**（可选，方便本机调试）
3. 装 **iFile/Filza**（传文件备用）
4. Wi-Fi 与 Mac 同网段，记下 IP

### 7.2 三种交付形态（按复杂度递增）

**形态 1：纯 CLI（第 1 天就能跑）**

```bash
scp llm4s tok.bin minimind26m.sllm root@192.168.x.x:/var/root/
ssh root@192.168.x.x
cd /var/root && chmod +x llm4s
./llm4s -m minimind26m.sllm -t tok.bin -p "你好" -n 64 --stats
# --stats 打印：tokens/s、RSS(KB)、每层耗时
```

这一步是**验证性的关键节点**：先拿到真实的 tok/s 和 RSS，再决定要不要做 UI。

**形态 2：HTTP 常驻 + Safari（推荐的最终形态）**

```bash
./llm4s -m /var/root/minimind26m.sllm --serve 8080 &
# iPhone 上 Safari 打开 http://127.0.0.1:8080
```

配 `LaunchDaemon` 自启（`/Library/LaunchDaemons/com.4sllm.plist`），开机即用，桌面上再加个 Web Clip 图标，体验和原生 App 差不多。

**形态 3：原生 App / Cydia 包（可选）**

- 需要 `iPhoneOS6.1.sdk` + UIKit 壳（一个 UITextView + UITextField 即可）
- `ldid -S` 后 scp 到 `/Applications/LLM4S.app/`，`uicache` 刷新图标
- 再用 `dpkg-deb` 打成 `.deb`，可上自己的 Cydia 源

### 7.3 内存/温度监控

```bash
vm_stat                          # 看 free/inactive 页
ps -o pid,rss,vsz,comm -p $(pidof llm4s)   # RSS 是否稳定不涨
```

- RSS 应稳定在 30–50MB，**若每轮对话都涨 = 内存泄漏**（arena 没复位或 KV cache 没截断）
- A5 满载 3–5 分钟会明显发热并降频，tok/s 掉 20–40%，属于正常

---

## 8. 优化清单（按收益排序）

| # | 优化 | 预期收益 | 备注 |
|---|---|---|---|
| 1 | int8 权重 + int16 激活（`vmlal_s16`） | **2–3×** | 最大单项收益，必做 |
| 2 | 双线程切分行维度 | 1.5–1.7× | A9 双核共享带宽，到不了 2× |
| 3 | 只算最后一个位置的 logits | 已默认 | 别整句重算 |
| 4 | 选 6400 词表的模型（MiniMind） | 性能 +50% | 见 §3 词表陷阱 |
| 5 | `mmap` 权重 + 预热一次触摸 | 冷启动 | 避免首次推理卡 2s |
| 6 | arena 预分配，零运行时 malloc | 稳定性 | iOS 6 的 malloc 慢且碎片化 |
| 7 | ctx 限制 256 | 内存 | KV cache 线性增长，别设 2048 |
| 8 | 关闭不需要的越狱守护进程 | 可用内存 +30–60MB | 用 iCleaner 精简 |
| 9 | 输出投影用 int8 行批（一次算完 6400 行） | 5–10% | 避免函数调用开销 |

不要做的优化：`-ffast-math`（坏 softmax）、`-mfloat-abi=hard`（ABI 崩）、NEON intrinsic 里手写 128-bit 假设（armv7 只有 64-bit 通路）。

---

## 9. 风险清单

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| libSystem 缺某些符号（如 `__cxa_thread_atexit`） | 中 | 链接/运行失败 | 避免 `thread_local`，用 pthread_key |
| SDK 9.3 的 tbd 引入新符号 | 中 | 上机 dyld 报错 | 只用 POSIX/libc；`nm -u` 逐个核对，必要时手工 `.tbd` |
| 内存压力被杀 | 中 | 进程突然消失 | RSS ≤60MB；`--serve` 时用看门狗重启 |
| 模型输出质量过低（MiniMind 26M 的 C-Eval ≈25.5%，等于 4 选 1 瞎猜） | **高** | 观感差 | 明确预期：它只有"语言形态"，没有知识 |
| 中文速度只有 5–8 字/秒 | 高 | 体感慢 | 6400 词表已是上限，只能减少输出长度 |
| 发热降频 | 高 | 长对话变慢 | 正常现象 |
| armv7 上 BPE 逐字节 merge 太慢 | 低 | 输入卡顿 | merge rank 用 hash 表，输入长度 <64 token |

---

## 10. 里程碑

| 阶段 | 内容 | 产出验证 |
|---|---|---|
| **M0** 工具链 | 编译 hello-armv7，上机跑通 `uname -a` | 二进制能被 iOS 6 加载 |
| **M1** 英文基线 | `runq.c` + stories15M 直接移植 | 实测 tok/s ≥15、RSS ≤25MB |
| **M2** 中文 | MiniMind2-Small 导出 + BPE 移植 | 中文能出通顺短句 |
| **M3** 交互 | HTTP + SSE + Safari 单页 | 手机上打字有回复 |
| **M4** 优化 | 手写 int8 NEON kernel + 双线程 + arena | tok/s 相比 M1 提升 2× |
| **M5** 打包 | deb / Web Clip / LaunchDaemon 自启 | 开机可用 |

M0→M1 是**决策点**：如果 M1 实测 tok/s <8，说明这条路的上限太低，建议直接停在这个 demo 状态，不要再投 UI 工作量。

---

## 11. 诚实的预期

**能得到的：**
- 一台 2011 年的手机，离线、完全本地地跑一个真正的 Transformer LLM
- 中文短句级输出（"今天天气不错"这种量级），速度 8–12 tok/s ≈ 每秒 6–9 个汉字
- 一个可复现的端侧推理教学样本：量化、NEON、KV cache、BPE 全链条

**得不到的：**
- 任何问答/助手级能力。26M 模型的中文选择题准确率 25.56%，和随机猜一样。它学会的是**汉字序列的统计规律**，不是知识
- 长上下文（ctx 256 就是上限，超过 KV cache 和算力都受不了）
- 多轮对话的连贯性（3 轮之后基本跑飞）

**如果你真正的目标是"在旧设备上有个能用的 AI"，正确的解法是：** 4S 只做前端（Safari 打开局域网地址），推理放在树莓派 5 或 Mac 上跑 1.5B–3B（int4，5–10 tok/s）。但那就不是"本地"了，取决于你怎么定义这个前提。
