# LLM 4 4s — 在 iPhone 4S / iOS 6.1.3 上跑本地 Transformer LLM

一台 2011 年的手机，512MB 内存，ARMv7 双核 800MHz，离线跑真正的 Transformer 语言模型。

名字就是它的履历：**LLM，for 4S**。别人眼里的电子垃圾，还能重新燃起一点光。

三种形态，同一套引擎：

- **原生 App（`LLM44S.ipa`）** —— Material Design 3 界面，双模型可切换，装完点图标就用。
  预编译包在 [Releases](https://github.com/ach1llea/llm44s/releases) 里，越狱机直接装
- **CLI** —— 无界面版本，带 `--stats`，适合压测和调 kernel
- **CLI + Safari** —— 内置 HTTP/SSE 服务，用浏览器当界面

```
  LLM44S.app (UIKit, MD3) ─┐
                          ├─▶  libsllm (armv7)  ──mmap──▶  minimind2s_q8.sllm  26M   int8  24.8 MB
  Safari ─HTTP+SSE──▶ CLI ┘                              minimind2_i8.sllm  104M  int8  99.7 MB
                                                         tokenizer.tok                  92 KB
```

两个模型共用同一个分词器（两个 `tokenizer.json` 字节级相同），App 里点顶部的模型 chip 切换，选择会记住。

| 模型 | 参数 | 量化 | 体积 | 4S decode | 定位 |
|---|---|---|---|---|---|
| MiniMind2-S | 25.8M | int8 | 24.8 MB | **20.5 tok/s** | 快，适合连续对话 |
| MiniMind2 | 104.0M | int8 | 99.7 MB | 预计 4–5 tok/s | 慢，但回答明显更靠谱 |

> **为什么 104M 用 int8 而不是 4-bit**：int4 的 logits 余弦只有 0.962，int8 是 0.995 且贪心输出与
> numpy fp32 逐字一致。体积翻倍不亏——4S 是带宽瓶颈，int4 解包还要额外指令，实测两者
> decode 速度几乎相同。所以默认发 int8。

## 实测数据

| 项目 | 结果 |
|---|---|
| 数值校验（26M int8，vs numpy fp32） | top-1 一致，余弦相似度 **0.9953** |
| 数值校验（104M int8，vs numpy fp32） | top-1 一致，余弦相似度 **0.9950**，贪心输出与 fp32 逐字相同 |
| 纯 int4 对照（不推荐） | 余弦 **0.913**；int4 + attn/emb 留 int8 为 **0.962** |
| 分词器（vs HF `tokenizers`） | 17/17 用例 id 完全一致（含中文/标点/全角） |
| 每 token 计算量 | 51.6 MFLOP（26M）/ 208.0 MFLOP（104M），与 2×参数量一致 |
| **4S · 26M int8** | prefill 14.1 tok/s，decode **20.5 tok/s**（48.9 ms/token），RSS 30 MB，ctx 512 |
| **4S · 104M int8** | decode **4–5 tok/s**（≈190 ms/token），RSS 83 MB；model 100 MB + KV 12 MB ≈ 118 MB 常驻 |
| Mac 2 线程（Apple Silicon） | 26M 349 tok/s，104M 282 tok/s（int8 与 int4 同速：都卡在指令而非带宽） |
| iOS 二进制 | 72KB(CLI) / 140KB(App)，armv7，`LC_VERSION_MIN_IPHONEOS 6.0`，`LC_UNIXTHREAD` → `start` |

104M 那档的有效带宽约 **347 MB/s**，低于 26M 的 459 MB/s：ARMv7 每周期只能取有限个 32 位字，
模型越大越吃不满总线，而不是内存到顶。

## 目录

```
src/            C 引擎（约 1900 行，无第三方依赖）
  model.c       .sllm 加载 + 前向（embed → N×[GQA attn, SwiGLU] → norm → logits）
  quant.c       int8 / int4 NEON 点积（vmull_s8 + vpadalq_s16）+ 自建线程池
  tokenizer.c   GPT-2 byte-level BPE，按 merge rank 贪心合并
  sampler.c     temperature / top-k / top-p / 重复惩罚（常驻内存 <1KB）
  http.c        HTTP + SSE，多轮上下文、增量 prefill
  compat_ios.c  iOS 构建专用：memcpy/memset/strcmp 等（SDK stub 里没有）
tools/
  export_model.py      HF safetensors → .sllm（手写解析，不需要 torch；--q4 出 int4）
  export_tokenizer.py  tokenizer.json → tok.bin
  ref_forward.py       numpy 参考实现（fp32），用于数值对拍
  ref_chat.py          fp32 贪心生成，判断"模型笨"还是"量化坏了"
  appsim.c             复现 App 的多轮 / KV-cache 流程（改 _conv 逻辑时跑它）
  verify_sllm.py       从 .sllm 反量化后跑参考前向，区分"量化损失"和"实现问题"
  test_tokenizer.py    分词器一致性测试
  fix_entry.py         LC_MAIN → LC_UNIXTHREAD（iOS 6 的 dyld 只认后者）
  patch_stub_libs.py   把 iOS 6 不存在的 libsystem_* 改写成 libSystem.B.dylib
app/            原生 iOS App（UIKit，MD3，无 storyboard，纯代码布局）
  main.m              AppDelegate + UIApplicationMain + 崩溃日志兜底
  RootViewController.m   MD3 界面：气泡 / 模型面板 / 运行日志 / 键盘处理 / 生成线程
  Info.plist  Icon.png  Icon@2x.png  make_icons.py
web/index.html  Safari 前端（纯 ES5，iOS 6 可用）
build_mac.sh    本机构建（CLI）
build_ios6.sh   armv7 / iOS 6 交叉构建（CLI）
build_ipa.sh    构建 LLM44S.ipa（原生 App，两个 int8 模型都打进去，114MB）
install_ipa.sh  通过 ssh 安装到设备
deploy.sh       拷贝 CLI 版到设备 + 可选 LaunchDaemon
deploy/com.4sllm.plist  开机自启配置
```

## 准备：下权重、转格式

仓库里只有代码，`.sllm` / `.tok` 是要自己生成的。以 26M 为例：

```sh
python3 -m venv .venv && .venv/bin/pip install numpy

mkdir -p models/minimind2s && cd models/minimind2s
for f in config.json model.safetensors tokenizer.json; do
  curl -LO "https://huggingface.co/jingyaogong/MiniMind2-Small/resolve/main/$f"
done
cd ../..

.venv/bin/python tools/export_model.py     models/minimind2s models/minimind2s_q8.sllm --seq 512
.venv/bin/python tools/export_tokenizer.py models/minimind2s/tokenizer.json models/minimind2s.tok
```

104M 同理，把 `MiniMind2-Small` 换成 `MiniMind2`、目录换成 `models/minimind2`。

`--seq` 决定 KV cache 上限，**默认 256**，本项目用 512；不写就会比这里的实测数据少一半上下文。
`export_model.py` 默认输出 int8，不需要 torch——safetensors 是手写解析的，只要 numpy。

## 快速开始（先在 Mac 上跑通）

```sh
./build_mac.sh
./llm4s -m models/minimind2s_q8.sllm -t models/minimind2s.tok -p "你好" -n 60 --threads 2
```

```
[model] dim=512 layers=8 heads=8 kv=2 ffn=1408 vocab=6400 ctx=512
[prompt] 25 tokens
您好！很高兴能帮助您。请问有什么我可以帮助您的吗？
[stats] prefill 25 tok in 44 ms (562.3 tok/s)
[stats] decode  9 tok in 16 ms (558.49 tok/s, 1.8 ms/tok)
[stats] flops   51.6 MFLOP/token
[stats] rss     29728 KB
```

网页模式：

```sh
./llm4s -m models/minimind2s_q8.sllm -t models/minimind2s.tok --serve 8080 --web web --threads 2
open http://127.0.0.1:8080
```

## 打包原生 App（推荐）

不想自己构建的话，[Releases](https://github.com/ach1llea/llm44s/releases) 里有构建好的
`LLM44S.ipa`。

```sh
./build_ipa.sh            # → LLM44S.ipa（114MB，两个 int8 模型都在包里）
./install_ipa.sh 192.168.x.x
```

构建机需要 `ldid`（`brew install ldid`）做伪签名：`build_ipa.sh` 会自动调用，缺了也能出包，
但设备端要装 AppSync Unified 才装得上。

安装脚本优先用 Cydia 的 `ipainstaller`，没有就解包到 `/Applications` + `uicache`。
装完桌面出现「LLM 4 4s」，点开即用，全程离线。

界面按 Material Design 3 深色主题做的，色板从一枚暖橙 seed 生成：

- **顶部 app bar**：`LLM 4 4s` 大字 + 模型 chip（显示当前参数量，点开是 MD3 bottom sheet 选择器）
- **仪表行**（14 pt，两行，每 0.5 s 刷新一次，生成时也不停止）：
  ```
  20.5 tok/s · CPU 87%
  ctx 42/512 · 294/504MB
  ```
  CPU 是整机负载，来自 `host_processor_info` 的 tick 差分（单次采样没有意义，必须按固定节奏取两次）。
  内存是「已用/总物理内存」，`total` 来自 `hw.memsize`，`used = total - (free + inactive)`。
- **生成时**：助手气泡里三个点依次明暗循环（0.55 s 一轮，错开 0.18 s），第一个 token 到达就换成文字
- **对话**：不对称圆角气泡，用户侧 primaryContainer，助手侧 surfaceContainer
- **输入**：胶囊输入框 + 圆形发送按钮（48pt 触控区）
- **日志**：点右上角菜单图标进入，等宽字体，每行带毫秒时间戳

模型选择面板从底部滑入，带 scrim，点空白处关闭。除了内置的两个模型，
App 还会扫描沙盒的 `Documents/models/*.sllm`，所以你自己导出的模型丢进去就能出现在列表里。

日志按 `HH:mm:ss` + 30 字符以内的描述逐行输出，节选：

```
13:46:02  ready rss=64224 KB, free=97 MB
13:46:09  gen   prompt=32 (cache 0) max_new=96 temp=0.85 top_p=0.90
13:46:14  tok   1  id=5134   191.1 ms  ctx=33
13:46:20  done  25 tok in 4778 ms  5.23 tok/s  rss=64224 KB
```

日志同时写入沙盒的 `Documents/engine.log`，未捕获的 ObjC 异常会写进 `Documents/crash.log`，
进程被杀也能事后查：

```sh
ssh root@<ip> "find /var/mobile/Applications -maxdepth 3 -name '*.log'"
```

崩溃报告在 `/var/mobile/Library/Logs/CrashReporter/`。

### iOS 6 的二进制要求

`build_ios6.sh` 与 `build_ipa.sh` 都内置了下面三步，普通交叉编译不满足其中任何一条都跑不起来：

- **入口点必须是 `start`**：直接调 `ld` 时不会自动链入 `crt1.o`，要显式加上；也不能用 `-dead_strip`
  （会把没人引用的 `start` 当死代码删掉）。
- **加载命令用 `LC_UNIXTHREAD`**：`ld` 给 armv7 生成的是 `LC_MAIN`，iOS 6 的 dyld 不认。
  `tools/fix_entry.py` 把它原地替换成完整的 `ARM_THREAD_STATE`，`pc` 指向 `start`。
- **`libsystem_*` 重定向到伞形库**：iPhoneOS9.3 SDK 把 libc 拆成 `libsystem_malloc` / `_pthread` 等，
  而 iOS 6 的共享缓存里只有 `libSystem.B.dylib`。`tools/patch_stub_libs.py` 复制一份 tbd 并改写
  `install-name`，再把这个目录放到链接器搜索路径最前面。

另外 `src/compat_ios.c` 补了 SDK stub 里没有的 `memcpy` / `memset` / `strcmp`，
编译时要带 `-fno-builtin-*`，否则编译器会把这些调用优化成对自身的调用。

## 部署到 iPhone 4S（CLI 版）

前置：已越狱（p0sixspwn）、Cydia 装好 OpenSSH（`root`，默认密码 `alpine`，**先改密码**）。

```sh
# 1. 一次性拉 SDK（约 56MB）
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/theos/sdks.git ~/.ios-sdks
cd ~/.ios-sdks && git sparse-checkout init --cone
git sparse-checkout set iPhoneOS9.3.sdk && git checkout && cd -

# 2. 交叉编译（armv7 / iOS 6.0）
./build_ios6.sh

# 3. 拷贝到设备（58MB，wifi 慢的话耐心等）
./deploy.sh 192.168.x.x
./deploy.sh 192.168.x.x --daemon     # 顺便装开机自启
```

在手机上的 MobileTerminal 或 ssh 里：

```sh
cd /var/root/4sllm
./llm4s-ios -m minimind2s_q8.sllm -t minimind2s.tok --serve 8080 --web . --threads 2
```

然后 Safari 打开 `http://127.0.0.1:8080`。桌面加个别名（Web Clip）就和 App 一样。

> 服务默认只绑 `127.0.0.1`。想在 Mac 上调试就用隧道，别把端口暴露到 wifi：
> `ssh -L 8080:127.0.0.1:8080 root@192.168.x.x`

## 换模型 / 重新导出

```sh
# int8（默认，≈1 字节/参数，问答与 fp32 逐字一致）
.venv/bin/python tools/export_model.py models/minimind2 out_i8.sllm --seq 512

# int4 只给 gate/up（省 20% 体积，余弦 0.977，仍连贯；内存吃紧时的折中）
.venv/bin/python tools/export_model.py models/minimind2 out_mix2.sllm --seq 512 --q4 32 --mix attn,emb,down

# int4 给整个 MLP（68MB，余弦 0.962，不推荐）
.venv/bin/python tools/export_model.py models/minimind2 out_mix.sllm --seq 512 --q4 32 --mix attn,emb

# 纯 int4（体积最小，余弦 0.913，不推荐）
.venv/bin/python tools/export_model.py models/minimind2 out_q4.sllm --seq 512 --q4 32 --mix none

.venv/bin/python tools/export_tokenizer.py models/minimind2/tokenizer.json out.tok
```

要求：Llama 架构（GQA + RoPE + RMSNorm + SwiGLU）、无 bias、词表被 token embedding 复用（tied）。
`--seq` 决定 KV cache 上限（`2 × L × kv_heads × head_dim × seq × 4B`）：256 时 104M 只要 6.3 MB，
512 翻倍到 12.6 MB，`--seq 1024` 是 25 MB。**本项目的模型用 512**——12.6 MB 换来两倍上下文，值得；
1024 会把 RSS 推到 95 MB 以上，在 4S 上就开始和 Jetsam 抢地盘了。

`--q4` 的组大小必须是 32 的倍数，且每个矩阵的列数要能被它整除——不满足会明确报错而不是
悄悄产出坏文件。导出后跑一次 `verify_sllm.py`：

```sh
.venv/bin/python tools/verify_sllm.py models/minimind2 out_q4.sllm "你好"
# cosine dequant-vs-fp32 : 0.91286   (pure quantization loss)
```

### 量化档位对比

| 方案 | 体积 | logits 余弦 | 与 fp32 贪心输出 |
|---|---|---|---|
| fp32 参考 | 208 MB | 1.000 | — |
| **int8** | 100 MB | **0.995** | **逐字相同** |
| int4（仅 gate/up） | 78 MB | 0.977 | 短句仍连贯 |
| int4（整个 MLP） | 68 MB | 0.962 | 明显跑偏 |

余弦是整条 logits 向量的相似度，而生成只看 top-1 附近那十几个候选：0.96 意味着几十个 token
之后就会走进完全不同的语义分支。所以默认发 int8，`--mix attn,emb,down` 留作内存吃紧时的折中。

同一个文件里混着两种精度不需要改格式：引擎在 `mat_get` 里比较 `nbytes` 和 `rows*cols`，
一字节一个权重就是 int8，半个字节就是 int4。

`verify_sllm.py` 会把 `.sllm` 自己反量化一遍再跑参考前向，因此量出的是**纯粹的量化损失**；
如果 C 引擎的 logits 和它对不上，那才是 kernel 的问题。

## 仓库说明

```sh
git clone git@github.com:ach1llea/llm44s.git
cd llm44s
git tag                          # v0.3 = 真机验证过的 iOS 6 构建
git diff v0.3 -- src/            # 和某个版本比差异
```

**不入库的东西**（都能重新生成，见 `.gitignore`）：`models/minimind2*/`（HF 权重，579 MB）、
`models/*.sllm`、`models/*.tok`、`*.ipa`、`build-*/`、`.venv/`。代价是**回退代码不等于回退
二进制**：回到旧版本后要重新 `./build_ipa.sh`，如果 `.sllm` 格式动过还要重跑
`tools/export_model.py`。

## 工程要点

**为什么要自写引擎而不是 llama.cpp**：llama.cpp 的 32-bit ARMv7 支持是二等公民——k-quants 用了
aarch64 专属的 `vld1q_*_x2` intrinsic 编译不过，`ggml-alloc` 里有 `size_t` 64 位假设，`sgemm.cpp`
依赖 FP16 NEON。自写反而更快落地。

**量化方案**：权重 int8 按行（per output channel）缩放，激活 int8 按向量缩放、存成 int16 供
`vmlal_s16` 使用。ARMv7 没有 int8 点积（`vdotq_s32` 是 ARMv8.2 才有），所以走 int8→int16 零扩展：
每指令 4 MAC，是 fp32 的两倍，同时权重带宽只有 1/4。

**上下文处理**：KV cache 跨轮保留，每轮只 prefill 新增的用户消息（实测第 2、3 轮的 tok/s 反而更高，
因为省掉了重复 prefill）。上下文写满就回滚到 system 前缀重来。

**重复惩罚只算「出现过」这一个事实**：最近 64 token 内每个 id 只罚一次，而不是按出现次数累乘。
后者会让高频虚词（`的`、`我`、换行）被压掉一个数量级，把模型推向越来越生僻的词。

**`--temp 0` 走 argmax**：贪心模式下 C 引擎可以与 numpy fp32 参考逐 token 比对，这是区分
「模型本身弱」和「量化/实现有问题」最快的办法。int8 104M 的贪心输出与 fp32 逐字相同。

**日志一行一条**：屏幕只有 320 pt，Courier 13 一行 38 列。引擎侧所有 `sllm_log` 格式串都压到
30 字符以内（`tok 12 803 3.8ms c26`），App 只加 9 字符的 `HH:mm:ss`。超宽会折行，一屏只能看到
几个 token，等于没日志。

**词表大小决定成败**：26M 模型的输出投影层是 `hidden × vocab`。MiniMind 用 6400 词表，
logits 只占 12% 算力；换成 32k 词表的同规模模型，一半算力会烧在输出层。

## .sllm 格式

所有整数小端，每个张量 64 字节对齐。

```
Header  68B : magic "SLLM" | version | dim | n_layers | n_heads | n_kv_heads |
              head_dim | hidden_dim | vocab_size | max_seq | rms_eps | rope_theta |
              n_tensors | quant | qgroup | pad[8]
Dir     36B : id, ndim, shape[4], nbytes, off, soff     （每张量一条）
Blob        : soff 处是 scale，off 处是数据
```

`quant` 为 1 时是 int8：每行一个 fp32 scale，数据是 `rows*cols` 字节。
`quant` 为 2 时表示「有 int4 参与」，`qgroup` 是组大小（本项目用 32）：

**每个张量可以是不同精度**，引擎靠 `nbytes` 区分（`rows*cols` → int8，
`rows*cols/2` → int4），所以一个文件里可以 attention 走 int8、MLP 走 int4，
不需要额外的格式字段。下面描述的是 int4 张量的布局：

```
每组 32 个权重共用 1 个 bf16 scale，打包成 16 字节：
byte[i] 的低 4 位 = 组内第 i 个元素，高 4 位 = 第 16+i 个
nibble 是有符号补码，范围 -8..7
```

bf16 scale 是为了省内存——ARMv7 的 NEON 没有 fp16 转换指令，但把 bf16 的高 16 位
左移 16 位再 bitcast 成 fp32 是零成本的。int4 让 104M 模型降到 55MB，
刚好塞进 4S 那点剩余内存。

## 已知限制

- 模型能力：26M 的中文 C-Eval 约 25.6%，四选一瞎猜也是 25%。它会说通顺的中文短句，但没有知识，
  三轮之后开始跑题。104M 好一些——问「深圳」它答「南端城市、东南沿海、多元文化」，
  26M 答「10 万家汽车制造商和航空航天工业」——但也只是「不那么离谱」。
- **104M 的 prefill 很慢**：32 token 要 6 秒才吐出第一个字。26M 只要 2.3 秒。
- 上下文 512 token（约 400 汉字）。再往上就是内存和 attention 的线性开销问题：
  1024 会让 104M 的 KV cache 涨到 25 MB，RSS 逼近 95 MB。
- 4S 上 20.5 tok/s（26M int8）/ 4–5 tok/s（104M int8）。满载 3–5 分钟会发热降频，
  速度掉 20–40%。104M 单轮对话 100 字左右约 20–30 秒。
- 单线程处理请求：生成时 `/stats` 会排队。

## 许可

本项目以 [MIT](LICENSE) 许可发布。

- `src/` 与 `tools/` 为本项目原创；推理思路参考 Karpathy 的 `llama2.c`（MIT）。
- 模型权重来自 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0），
  权重文件本身不入库，按上文「快速开始」自行转换。

## 贡献

Issue 和 PR 都欢迎。改动量化或 kernel 之前先看一遍上面「实测数据」——4S 上的每个数字都来自
实际测量，别凭直觉推翻。
