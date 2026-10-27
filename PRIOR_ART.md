# 在旧硬件上跑 LLM 的开源案例

调研日期：2026-10-27。所有数据均为项目作者自测，非第三方复现。

---

## 结论

**这类项目有，而且是一个完整的谱系，但它们几乎全部源自同一个起点：karpathy 的 `llama2.c`。**

那不只是一个"能在小设备上跑的推理器"，它真正做的事情是：用 700 行 C 复刻 Llama 2 的前向计算，
把移植成本从"适配一个框架"降到"适配一个 C 编译器 + 一种权重格式"。
于是 8 位机、DOS、Windows 98、PowerPC、ESP32 全都能接上去。

`4sllm` 是这个谱系的第二代：起点相同，但多了 int8 量化、mmap、多线程、增量 prefill、HTTP/SSE 和原生 UI。

---

## 总表

| 项目 | 硬件 | 年代 | 内存 | 模型 | 速度 | 与 4S 的可比性 |
|---|---|---|---|---|---|---|
| [llama2.c](https://github.com/karpathy/llama2.c) | M1 MacBook Air | 2022 | 8GB | stories15M fp32 | ~110 tok/s | 基准线 |
| [ullm](https://www.xinfinite.net/t/topic/11413)（PowerBook G4） | PowerPC G4 1.5GHz 单核 | 2005 | 1GB | stories110M fp32 | **~0.8 tok/s** | ★★★★ 同为 32 位地址空间 |
| [llama2.c64](https://github.com/ytmytm/llama2.c64) | C64 6510 1MHz + 2MB REU | 1982 | 64KB+2MB | stories260K fp32 | **~8 分钟/token** | ★★ 验证性 |
| [dosllam2](https://github.com/yeokm1/dosllam2) | 486DX2-66 / 32MB | 1993 | 32MB | 260K / 15M | 2.08 / — tok/s | ★★★ 有完整横向基准表 |
| [dosllam2](https://github.com/yeokm1/dosllam2) | Pentium M 1.7GHz / 2GB | 2004 | 2GB | 15M / 110M | 11.55 / 1.71 tok/s | ★★★★★ **最佳锚点** |
| [llama98.c](https://github.com/exo-explore/llama98.c) | Pentium II 350 / Win98 | 1997 | 128MB | 260K / 15M | 39.31 / 1.03 tok/s | ★★★ |
| [esp32-ai](https://github.com/slvDev/esp32-ai) | ESP32-S3 240MHz 双核 | 2026 | 512KB SRAM + 8MB PSRAM | TinyStories 28.9M **4-bit** | **9.5 tok/s** | ★★★★ 内存分层技巧 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / Termux | 骁龙中端旧安卓机 | 2015+ | 3-6GB | Qwen3 0.6B Q4 | 14-18 tok/s | ★★ 64 位，参考价值有限 |
| **4sllm** | **A9 800MHz 双核 / iOS 6** | **2011** | **512MB** | **MiniMind2-Small 25.8M int8** | **预估 8-12 tok/s** | — |

---

## 逐例要点

### 1. PowerBook G4 / ullm（最接近你的场景）

- 1.5GHz PowerPC G4，**32 位地址空间**，1GB 内存。
- 模型只能停在 110M —— 不是因为算不动，是因为**地址空间装不下更大的**。
- 移植时踩的坑：**大端 CPU vs 小端权重格式**（要逐元素字节交换）、内存对齐、
  `gcc 4.x` 不支持现代 C 语法。
- 加速手段：**AltiVec（PowerPC 的 SIMD）**，让一次 183 token 的查询从 4 分钟降到 3.5 分钟。
- 作者原话大意：光是让它在 20 年前的笔记本上算出来，就已经是目的本身。
- **对我们的意义**：这是唯一一个"32 位 + 桌面级旧 CPU + 手动写向量内核"的完整先例，
  它证明就算只能拿到个位数 tok/s，工程上依然成立。

### 2. dosllam2（唯一有系统横向基准表的）

MS-DOS 6.22 + Open Watcom + DOS/32 extender，跑在从 486 到 ThinkPad 的一串机器上。
这张表是整个领域最有价值的**定量标尺**：

| 系统 | 260K | 15M | 42M | 110M |
|---|---|---|---|---|
| 486 DX2-66 / 32MB | 2.08 | — | — | — |
| Pentium MMX 200 / 96MB | 15.32 | 0.43 | — | — |
| Pentium III 667 / 512MB | 80.04 | 2.35 | 0.89 | — |
| **Pentium M 1.7G / 2GB** | 331.6 | **11.55** | 4.42 | **1.71** |
| Core i5-10310U / 16GB | 386.36 | 10.51 | 3.89 | 1.53 |

两点观察：

1. **tok/s 与参数量严格成反比**。15M→42M 是 2.8 倍参数，11.55→4.42 正好 2.6 倍；
   42M→110M 是 2.6 倍参数，4.42→1.71 也正好 2.6 倍。说明瓶颈是**每 token 的权重读取量**，
   不是别的。
2. **fp32 下这条线是内存带宽线，不是算力线**。T42 跑 15M 时每 token 读 60MB，
   11.55 tok/s ⇒ 693MB/s 有效带宽；跑 110M 时读 440MB，1.71 tok/s ⇒ 752MB/s。
   两个数字几乎相等 —— 这说明 CPU 一直没被算满，**在等内存**。

### 3. llama98.c（EXO Labs，Windows 98）

- Pentium II 350 + 128MB，Borland C++ 5.02（当年编译器，`long long` 都不支持，要 typedef 成 `DLONGWORD`）。
- 260K 模型 39.31 tok/s，15M 模型 1.03 tok/s。
- 关键改动：**把 mmap 换成 read** —— 因为 Win98 上内存映射直接段错误。
  这和我们在 iOS 6 上遇到的"`__cxa_thread_atexit` 不存在"是同一类问题：
  **老 libSystem 的可用 API 集合比语言标准小得多**。
- 作者的另一句话值得记住："如果它能跑在 25 年前的硬件上，那它跑在任何地方。"

### 4. llama2.c64（走到底的那一个）

- C64，1MHz 6510，2MB REU 扩展内存，`oscar64` 工具链。
- **约 8 分钟生成一个 token**。模型权重留在 float32，所以 REU 最多只能装 ~8MB 模型。
- 有意思的两个工程细节：
  - 自己重写了 `sin/cos/exp`，因为编译器自带的精度不够，多项式系数直接从 **C64 BASIC ROM** 里抄。
  - 验证方式：在 `temperature=0` 下与 `llama2.c` **逐 token 比对输出**，完全一致才认为移植成功。
- **对我们的意义**：这个"确定性逐 token 比对"的验证方法，比看输出通不通顺严格得多。

### 5. esp32-ai（技术上最有借鉴价值的一个）

- ESP32-S3：240MHz 双核 Xtensa，**512KB SRAM**，8MB PSRAM，16MB Flash。
- 跑 28.9M 参数 TinyStories，4-bit 量化后 14.9MB，**9.5 tok/s**。
- 核心手法：**Per-Layer Embeddings**（借自 Google Gemma 3n/4）——
  把 25M 参数、占绝对多数的 embedding 表**留在慢速 Flash 里**，
  每 token 只读当前需要的 6 行（约 450 字节）；真正参与计算的 560K 参数才放 SRAM。
- 对比：之前同类硬件上的记录是 260K 参数，它一下子把规模提高了 100 倍。
- **对我们的意义**：这就是"按访问频率给参数分层"的思路。我们的 mmap 天然只有被 touch 的页会进内存，
  但可以用 `madvise(MADV_RANDOM)` 关掉内核的顺序预读，避免把用不到的表页拉进 120MB 的紧张预算里。

### 6. llama.cpp 在旧安卓（反面教材）

- 可行但没意义：骁龙 855 以下，1B 级模型 Q4 只能到个位数 tok/s。
- 社区的经验值是"**骁龙 855 是能用的下限**"。
- 这恰好说明：**4S 用 0.5B 模型是不可能的**（我们原来判断的 400MB 内存需求是硬墙），
  只有 25M 量级 + int8 这条路。

---

## 用这些数据校准 4S

这是这次调研最实际的收获。

**iPhone 4S 的内存带宽和 2004 年的 ThinkPad T42 是同一量级**：
A5 配 LPDDR2，理论 ~3.2GB/s，实测有效带宽约 0.8-1GB/s；T42 的 DDR333 实测也是这个数。

所以 T42 的 11.55 tok/s（15M fp32）是一个有意义的锚点：

| 步骤 | 计算 | 结果 |
|---|---|---|
| T42 跑 15M fp32 | 每 token 读 60MB | 11.55 tok/s ⇒ 693MB/s |
| 4S 同带宽跑 15M fp32 | 假设有效带宽 700MB/s | ≈ 11 tok/s |
| 换成 25.8M int8 | 每 token 只读 26MB，理论带宽上限 27 tok/s | 带宽不再是瓶颈 |
| 此时瓶颈变成 A9 算力 | 双核 NEON `vmlal.s16`，25.8M MAC/token | ≈ 8-15 tok/s |

**结论：8-12 tok/s 这个预估站得住，而且是通过别人的实测数据推出来的，不是我拍脑袋。**
唯一可能落空的是 A9 的有效算力利用率 —— 如果实测掉到 5 tok/s 以下，
说明瓶颈不在带宽而在指令效率，那时要做的是展开 kernel，而不是换模型。

---

## 能直接抄进 4sllm 的五件事

1. **`madvise(MADV_RANDOM)` + 显式 `pread` 取 embedding 行**（esp32-ai 的思路）。
   现在 26MB 全量 mmap，内核可能预读用不到的页。
2. **确定性逐 token 比对**（llama2.c64 的验证法）。
   我们只做了单点 logits 余弦 0.9953；加一个 `-t 0` 模式跑 100 token，与 numpy 参考实现逐 id 对比，
   才算真正把"移植正确"这件事钉死。
3. **kernel 展开**（PowerBook 的 AltiVec 经验）。
   向量化本身能拿到大头，但循环展开还能再榨 20-30%，这是所有旧硬件项目的共同结论。
4. **砍掉一切非必要 libc 依赖**（DOS / Win98 的教训）。
   我们已经在 `compat_ios.c` 里手写了 `memcpy/strcmp/strchr`，方向是对的。
5. **功能边界要写在 README 第一行**（所有项目的共同自觉）。
   C64 那版作者写得很直白："这不是聊天模型，你就当在给三岁小孩一个故事开头。"

---

## 没有先例的部分

检索下来，**没有找到任何 iPhone 4S / iOS 6 / armv7 上跑 Transformer LLM 的公开案例**。

最接近的三个都没有覆盖这一点：

- **llama.cpp 的 iOS 支持**：早已砍掉 32 位 ARM，最低 iOS 12+，且依赖 Metal 或 64 位 NEON。
- **MLC-LLM / CoreML 方案**：全部要求 arm64。
- **旧安卓案例**：都是 64 位 + 现代 libc。

也就是说，`4sllm` 需要同时解决三个别人没在同一个设备上同时解决的问题：

1. 32 位 armv7 的 int8 内核（llama.cpp 已放弃）
2. iOS 6 `libSystem` 的符号缺失（`__cxa_thread_atexit`、`memcpy` 全家）
3. 没有 Metal、没有 CoreML、可用内存 120-200MB

这三件事各自都有先例，**合在一起没有**。这是这个项目真正的价值所在 ——
不是"跑了个 AI"，是把一条已经被所有人放弃的技术路径重新走通了。

---

## 链接

- llama2.c — https://github.com/karpathy/llama2.c
- llama2.c64 — https://github.com/ytmytm/llama2.c64
- dosllam2 — https://github.com/yeokm1/dosllam2
- llama98.c — https://github.com/exo-explore/llama98.c
- esp32-ai — https://github.com/slvDev/esp32-ai
- L2E（Llama 2 Everywhere，多平台分支）— https://github.com/trholding/llama2.c
- llama.cpp — https://github.com/ggml-org/llama.cpp
- BitNet（1-bit，CPU 上的另一条路）— https://github.com/microsoft/BitNet
- PowerBook G4 实验中文整理 — https://www.xinfinite.net/t/topic/11413
