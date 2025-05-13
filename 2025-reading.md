## 2025-01

## 2025-02

## 2025-03

- [<mark>端侧推理，npu+cpu，量化int8</mark>] **Fast On-device LLM Inference with NPUs**. Daliang Xu et.al. **ASPLOS25**, **2024**, ([link](http://arxiv.org/abs/2407.05858v2)).

这篇是很经典的一篇利用NPU在端侧加速推理的一篇文章，已经被ASPLOS25录用。因为隐私计算的需求，以及手机性能的提高，端侧运行llm越来越可行，但是端到端延迟不符合要求。本文重点关注端侧的prefill阶段加速，在长序列的任务中，这部分计算延时占比可以达到80%以上。因为prefill阶段是计算密集型，且npu擅长整数并行运算且广泛存在于手机中，另外手机cpu/gpu并行计算能力相对于消费级gpu并行能力差，所以用npu加速prefill阶段的推理是很符合直觉的。

核心思想：在量化的模型下，在prefill阶段将尽可能多的整数计算移动到npu上执行，同时将浮点运算在cpu/gpu上执行。挑战：1.对于变长的prompt，npu只支持静态大小的操作，准备工作(构建优化计算图)耗时；2.npu计算与sota量化方法不匹配，常用的per-group量化无法在npu上执行，需要拆分矩阵运算；3.npu不擅长浮点运算，而attention和layernorm往往需要浮点计算以保持精度。

对于挑战1，提出了分块chunk与共享计算图，对于静态的计算图（如ffn），在不同chunk间共享，对于有序列位置依赖的计算图（如attention），不共享计算图；对于挑战2，提出了Shadow outlier execution，mllm.npu采取了per-tensor的量化方法以适应npu的结构，对于激活值中的离群值，将浮点部分移动到cpu上计算，最后合并结果。对于npu,cpu内存地址空间不共享，需要复制离群值对应的权重，这里采用了统计分析哪些channel容易出现离群值（hot,集中），预先复制，cold的权重按需从磁盘加载；对于挑战3，主要是解决如何协调npu/cpu间计算的问题，按chunk顺序调度然后重合计算肯定是不可行的，因为ATTENTION计算会变化，这里采用的是乱序执行，只要满足数据依赖关系，就可以执行，这里考虑的依赖包括chunk内部的依赖以及跨chunk的依赖。因为实验发现npu执行时间占推理的主要部分，所以是关键路径，我们要尽可能减少npu执行的停滞。所以对于具体调度来说，优先在cpu上调度subgraph(满足的后续依赖可以让npu执行最久)，在npu上则优先调度执行时间最长的subgraph。

- <mark>端侧推理(pc)，存储优化，激活稀疏，参数offload</mark> **LLM in a flash: Efficient Large Language Model Inference with Limited Memory**. Keivan Alizadeh et.al. **arxiv**, **2023**, ([link](http://arxiv.org/abs/2312.11514v3)).

端侧设备（个人pc）内存有限，需要offload模型参数到flash中，本文关注激活稀疏场景下从flash（nvme nand）加载参数到dram的通信优化。nvme特点是线程越多chunk越大带宽就越大，同时激活稀疏利用预测器可以只加载少量神经元。本文提出1.滑动窗口作为缓存的依据，因为神经元有一定agggregate的特性，所以保留窗口大小以前激活的神经元，这样可以减少传输的数据大小。2.bind neuron，增加chunk size，提高带宽。3.提出了一个在dram放置管理neurons的方法，不过感觉作用不大。

- <mark>serving，分布式kv-cache存储，prefill-decode分离</mark> **Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot**. Ruoyu Qin et.al. **FAST**, **2025**, ([link](https://www.usenix.org/system/files/fast25-qin.pdf)).

本文获得了fast25最佳论文奖，还是很牛的。本文的场景是巨大的Mass，该场景的特点是在满足SLO下最大化吞吐量。当然，这里有一个前置的配置就是说prefill与decode instance分离（普遍做法）。

在prefill阶段，kvcache前缀重用可以大大减小计算量，原始计算量$\text{flops}(n) = l \times (a n^2 d + b n d^2)$，假设n token中有前p个token被重用，则计算量变为原来的$(\frac{p}{n})^2$。假如kvcache不在prefill节点，则需要加载而只要传输时间小于减少部分计算时间，就有收益，经过计算llama3-70b + 1node(8*h800)只要20GB/s带宽（min NIC&h2d）即可，而这是很容易满足的。所以本文就提出核心思想`trading more storage for less computation`，利用上一切可以用上的存储资源（DRAM,SSD,RDMA）来持久化存储kv cache，尽可能提高prefix hit rate。挑战无非是：要提高hit rate,存储的kvcache就要很大，怎么存的问题；相对应的就是分布式环境不同设备传输kvcache的带宽问题；另一个就是基于serving的环境以及目标如何最优化调度的问题喽。

当一个请求到来时，调度器根据工作负载与kv cache状态调度到一个prefill与decode node pair。与传统的区别就是prefill前会进行Kvcache重用的加载，以及decode前会通过文件系统MOONCAKE Store传输到decode node(与prefill重叠)。

这里再补充一点RDMA的知识:
传统网络传输路径`用户空间缓冲区 → 内核空间缓冲区 → 协议栈处理 → NIC发送`
RDMA 传输路径`用户空间缓冲区 → RDMA NIC直接访问（通过虚拟地址映射）`

**MOONCAKE Store**，page block存储（page attention），hash id标记自己及prefix,**LRU逐出策略**。这里设计到分布式的文件传输api设计，一个是kv cache包装后的存取复制等api,一个是跨节点的传输api。为了实现这些api,需要设计一个transfer engine,对于跨节点传输（local dram/vram to renote location using RDMA NIC）的带宽限制的问题（本地DRAM/VRAM → UPI → PCIe Switch → RDMA NIC → 网络），通过Topology-aware path selection的方法实现动态RDMA路径的选择，同时将传输请求分片均衡RDMA NIC负载。

对于prefill,文章说是用了一个chunked pipeline parallelism (CPP) ，不过我没感觉这个东西和序列并行的区别？有待讨论。

对于prefill调度，给予该请求prefill time（预测而来） + local排队时间，prefill time通过该Instance重用前缀数量及请求长度预测得到（offline训练）。这里并不会以最大的匹配前缀来确定调度到哪个instance，有一个kv-cache负载平衡的算法，同时这样做可以做到自动复制热点的前缀。

实验来说设备很豪华，16 * node( eight NVIDIA-A800-SXM4-80GB GPUs and four 200 Gbps RDMA NICs.),模型LLaMA3-70B
这里的一个亮点就是细分了使用场景：Conversation Tool&Agent Synthetic（长序列），mooncake遵循slo还是很不错的。
对于llama3-70b,假设1node-1T mooncake-cache- 3M tokens - 50% theoretical maximum hit rate，而且100%max hit rate则需约50M tokens - 20node,太恐怖了。
这里对于hot cache的分析也很有意思，比如agent场景，hot key几乎每个node都保存了。

- <mark>并行编码，training-free，kv-cache相似</mark> **APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding**. Xinyu Yang et.al. **ICLR Poster**, **2025**, ([link](http://arxiv.org/abs/2502.05431v2))([open review](https://openreview.net/forum?id=yUC8pU508S)).

因为说发现trainable的方法对于处理复杂推理任务的性能不行，本文采用了一种training-free的方法。基于不同context的kvcache的相似性，
内在性质证明了不同上下文的kvcache是可以组合的。并针对并行解码效果下降的原因进行了分析。针对头部attention sink导致的异常高的attention score,
采用共享前缀；为使不同上下文拼接后attention score更平滑，对注意力计算增加了温度参数，并针对该温度参数影响attention score绝对值的问题在attention score后增加了一个缩放因子抵消该影响。

- **KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse**. Jingbo Yang et.al. **arxiv**, **2025**, ([link](http://arxiv.org/abs/2502.16002v1)).

- **Speculative Decoding and Beyond: An In-Depth Survey of Techniques**. Yunhai Hu et.al. **arxiv**, **2025**, ([link](http://arxiv.org/abs/2502.19732v3)).

- **LongSpec: Long-Context Speculative Decoding with Efficient Drafting and Verification**. Penghui Yang et.al. **arxiv**, **2025**, ([link](http://arxiv.org/abs/2502.17421v1)).

- **Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding**. Heming Xia et.al. **arxiv**, **2024**, ([link](http://arxiv.org/abs/2401.07851v3)).

- **KVDirect: Distributed Disaggregated LLM Inference**. Shiyang Chen et.al. **arxiv**, **2024**, ([link](http://arxiv.org/abs/2501.14743v1)).

- **vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention**. Ramya Prabhu et.al. **arxiv**, **2024**, ([link](http://arxiv.org/abs/2405.04437v3)).

- **IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference**. Weijian Chen et.al. **FAST25**, ([link](https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress)).

- [<mark>并行编码，trainable</mark>] **Long-Context Language Modeling with Parallel Context Encoding**. Howard Yen et.al. **ACL**, **2024**, ([link](http://arxiv.org/abs/2402.16617v2)).

比较早提出并行编码方式的工作。使用的trainable的方法，对于原本的大模型，增加了一个encoder编码不同上下文，同时在大模型的每一层增加了一个cross-attention layer,用于将prompt关注外部context。

- **Attention Entropy is a Key Factor: An Analysis of Parallel Context
  Encoding with Full-attention-based Pre-trained Language Models**. Zhisong Zhang et.al. **arxiv**, **2024**, ([link](http://arxiv.org/abs/2412.16545v1)).

- **LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism**. Bingyang Wu et.al. **sosp24**, **2024**, ([link](http://arxiv.org/abs/2404.09526v2)).

- **FastDecode: High-Throughput GPU-Efficient LLM Serving using
  Heterogeneous Pipelines**. Jiaao He et.al. **arxiv**, **2024**, ([link](http://arxiv.org/abs/2403.11421v1)).

- **Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed Large Model Inference**. Du Jiangsu et.al. **ppopp24**, **2024-2-20**, ([link](https://doi.org/10.1145/3627535.3638466)).

## 2025-04

- **Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning**. Wei An et.al. **SC24**, **2024**, ([link](http://arxiv.org/abs/2408.14158v2)).

deepseek的萤火AI系统的设计，主要介绍了cost-effective的AI-HPC设计（万卡PCIE A100），对标英伟达的DGX-A100设计。主要包括5方面的设计，
一个是网络的设计上，一个是计算通信重叠的HFREDUCE（对标NCCL），一个是存储3FS，一个是HaiScale用于并行方法优化，一个是HAI调度系统错误恢复等。

## 2025-05
