# 😃 Agent 系统优化调研

### Cataglories

* multi-agent workflow design
  * Magentic-one: A generalist multi-agent system for solving complex tasks. 2024.
  * Optimizing Sequential Multi-Step Tasks with Parallel LLM Agents. 2025.
  * PEER: Expertizing Domain-Specific Tasks with a Multi-Agent Framework and Tuning Methods. 2024.
  * Very Large-Scale Multi-Agent Simulation in AgentScope. arXiv:2407
* RAG optimization (specific)
  * llm prefill latency
    * RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation. arXiv:2404
      * rag prefix kv-cache
* Compound AI optimization
  * framework design
    * Towards Resource-Efficient Compound AI Systems. HOTOS 25
      * AIWaas design, Workflow-Aware Cluster Management.
  * SLO Goodput
    *
  * e2e latency
    * Towards Efficient Compound Large Language Model System Serving in the Wild. IWQoS'2024
      * DAG is uncertain, including dependency and exec time; priority schedule most uncertain request to parallel consequential stage;
    * LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications. arXiv:2504
      * uncertainty aware schdule to minimize average JCT
    * Efficient Serving of LLM Applications with Probabilistic Demand Modeling. Arxiv:2506
      * uncertainty modeling to better schdule and prewarm backend to minimize JCT
  * llm prefill latency
    * KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. Arxiv:2507
      * agent prefix kv-cache cache

### papers

> Tempo: Application-aware LLM Serving with Mixed SLO Requirements

现有3种请求类型

* **延迟敏感型请求** ：如交互式聊天机器人（ChatGPT）、实时语音转文字服务，关注 **逐 token 生成延迟** （TTFT、TBT），需逐步流式响应。
* **吞吐密集型请求** ：如大规模代码生成、批量数据处理 API，需在指定时间内完成整个响应（TTLT）。
* **集体请求** ：如多智能体系统或复杂推理任务（Tree of Thoughts），涉及多个 LLM 调用的动态依赖关系DAG，需整体端到端延迟满足 SLO。

现有调度器（如 Sarathi-Serve、Autellix）通常针对单一类型请求优化：

* **延迟感知调度器** ：优先处理低 TTFT/TBT 请求，但可能牺牲吞吐密集型请求的性能。
* **吞吐感知调度器** ：优化 TTLT，但可能导致延迟敏感请求的 SLO 违规。
* **资源浪费** ：为确保 SLO 严格满足，可能过度分配资源，降低其他请求的效率。

motivation：**LLM请求的多样性与SLO需求的复杂性**

* **跨应用差异** ：不同应用对SLO的需求显著不同（如聊天机器人侧重延迟，批量处理侧重吞吐）。
* **同一应用内差异** ：
  * 聊天机器人用户因阅读速度不同，TBT需求差异显著（38.1%用户偏好流式代码交付，30.5%用户需快速全响应）。
  * 定价影响相应时间
  * 复杂推理任务中，初始“思考”阶段需快速响应（如5秒内完成内部推理），而后续总结阶段需流畅生成。

核心思想：通过量化服务增益（Service Gain），动态分配“恰到好处”的资源满足SLO，最大化剩余资源服务其他请求。

挑战：

* **请求不确定性的挑战：包括DAG，长度预测、请求负载变化**

> Optimizing SLO-oriented LLM Serving with PD-Multiplexing

场景：多轮对话kv-cache重用，长序列

解决的核心问题是 LLM 服务中 **SLO 保证**与**高吞吐量**的冲突

PD slo的区别导致天然需要PD分离，计算特性原因导致SLO下的并行策略、PD实例比例的不同等

问题：kv cache重用下，需要将D→P，无法layer-wise传输（P→D），导致多轮对话请求高latency。否则就需要重计算kv-cache

另一个方法是chunk-prefill，但是在SLO下无法满足GPU的高利用率→吞吐量不够,同时attention计算复杂度变化

本文核心思想：spatial prefill-decode (PD) multiplexing approach.，利用nvidia官方`GreenContext`分割方法，分割计算核，但不分割存储。既可以PD分离，又可以无kv-cache迁移开销

挑战：如何调度请求满足SLO并MAX吞吐量，即建模；如何划分；如何同步，即请求从p→d

架构：

* **离线建模**：通过 GreenContext（NVIDIA 提供的 GPU 内部空间划分技术）预训练 latency预测模型，量化不同计算分片比例下 prefill/decode 的执行时间。参考loongserve建模
* **在线服务**：
  * **SLO 感知调度器**：根据预测器的预测信息，动态分配计算资源（划分方案），优先保证 decode 阶段的 SLO。
  * **自适应组调度**：将 prefill 阶段切分为多个块（blocks），与 decode 阶段并行执行，减少资源浪费。

#### **自适应Gang调度（Adaptive Gang Scheduling）**

* **分块预填充**：将长预填充阶段拆分为多个块（PBs），每个块独立调度。减小decode等待prefill完成的气泡，同时减小kernel launch导致的decode间的气泡
* **动态同步**：通过查询CUDA事件状态实现非阻塞同步，避免因预填充和解码阶段完成时间差异导致的“气泡”（资源空闲）。
* **优先级调度**：短请求可抢占长预填充块的执行，避免SLO违反。

![image.png](attachment:3ffa1521-a821-4ee1-8ae5-a5b090a8290e:image.png)

这部分为什么不能并行的启动kernel有点疑问。可能是这个原因：GreenContext 是 NVIDIA 提供的一种**进程内 GPU 计算资源分区技术**，允许在同一个 CUDA 进程中创建多个虚拟的“上下文”（Green Contexts），每个上下文可以独立分配 SMs（流处理器）资源。其核心设计目标是**在同一进程中实现多个任务的空间复用**，而非跨进程隔离。

> Chameleon: a Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models

**第一类 RALM：检索“文本块”，单词检索，多次间隔检索**

**第二类 RALM：仅检索“下一个 token”，检索数据库类似上下文的下一个token，query是最后一层的hidden state，logits与下一个token做平均**

向量搜索的定义

A vector search takes a 𝐷-dimensional query vector 𝑥 as input and retrieves 𝐾 similar vector(s) from a database 𝑌, populated with many 𝐷-dimensional vectors, based on metrics like L2 distances or cosine similarity.

𝐾 nearest neighbor太昂贵，业界往往用approximate nearest neighbor (ANN)。ANN效果一般用recall at 𝐾 (𝑅@𝐾)来评估。本文采用IVF-PQ这种ANN的方法，原因是它有压缩。

RALM推理的问题：

1. LLM与RAG推理特性不同，LLM主要是计算、带宽等；RAG主要是搜索算法、存储
2. 广泛的RARL配置，比如查询的频率，数据库的规模，模型的大小

motivation：

* CPUs are slow in scanning PQ codes during query time，即使用了SIMD；且占用大量内存带宽1G/core。GPU计算也不行，主要是显存容量以及CPU与GPU间带宽的问题。异构算力加速计算，如FPGA作为RAG的加速器
* 广泛的RARL配置导致两者所需资源是变化的，需要可以独立地弹性扩容解决瓶颈问题。

核心思想：异构算力加速、同时PD分离的类似思想，分离RAG与LLM推理

主要组成是3部分，一部分是GPU上的推理引擎以及IVF list；一部分是CPU上的协调进程；一部分是FPGA的向量查找节点，通过网线连接；

这里本文在RAG查询时GPU是闲置的，并没有用micro batch等，作者说是会降低吞吐量且二者时延不匹配。同时模型并没有用并行技术，不过这也不是本文关心的部分，但是如果并行的话这部分闲置的开销是不是就不可忽略呢？

> RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving

也是IVF-PQ

Empirical RAG Performance Trade-off Analysis，数学建模FLOPsinference，Bretrieval，End-to-end RAG performance

RAG问题

* **（C1）内在异构性 (Heterogeneity)**：RAG系统由多个不同类型的组件构成，如用于向量搜索的检索器、生成式LLM，以及可选的文档编码器、查询重写器和结果重排序器等 。这些组件通常运行在不同的硬件上（例如，检索器在CPU上，LLM在GPU/TPU等加速器上），这使得优化变得极其复杂 。
* **（C2）性能多变性 (Variability)**：不同的RAG配置（如知识库大小、检索频率、模型选择等）会导致性能瓶颈在不同组件之间转移 。例如，有时瓶颈在检索，有时在LLM推理 。

RAGSchema：为复杂的RAG工作负载建立统一语言，利用RAGSchema定义并分析了四种代表性的RAG模式，揭示了它们的性能瓶颈所在：

*   **场景一：超大规模检索**

    **特点**：中小型LLM搭配极大知识库（数万亿令牌）。

    **发现**：**检索是主要瓶颈**，尤其当LLM较小或处理多查询时，检索可占总时间80%以上。
*   **场景二：长上下文处理**

    **特点**：实时处理长文档（>10万令牌）作临时知识库。

    **发现**：检索时间几乎可忽略（<1%），瓶颈转移至

    **文档编码**。即使编码器仅1.2亿参数，处理大量文本的耗时也超过700亿参数主LLM。
*   **场景三：迭代式检索**

    **特点**：生成过程中周期性多次检索，获取精准上下文。

    **发现**：频繁暂停解码等待检索结果**严重影响生成速度**。检索批次大小设置关键：太小增加延迟，太大导致解码器空闲。
*   **场景四：增强型RAG**

    **特点**：检索前后分别加入查询重写器和重排序器优化质量。

    **发现**：**查询重写器显著增加TTFT**，因其本身是自回归生成过程。重排序器计算量小，影响不大。

核心思想：根据**RAGSchema**和给定的**硬件资源**，为其量身定制一套最优的**调度策略**

为了应对上述挑战，论文提出了RAGO框架，它能根据给定的RAGSchema和硬件资源，自动探索并找出最优的系统调度策略 。RAGO主要优化以下三个维度的决策 ：

1. **任务放置 (Task Placement)**：决定RAG流水线中的不同组件应该\*\*合并部署（collocated）**在同一组加速器上，还是**分离部署（disaggregated）\*\*在不同的硬件资源上 。例如，是否应该将计算特性相似的文档编码器和LLM前缀计算（prefix）阶段放在一起 。
2. **资源分配 (Resource Allocation)**：为每个（合并或分离的）任务分配合适数量的硬件资源（如XPU加速器的数量或CPU服务器的数量） 。
3. **批处理策略 (Batching Policy)**：为流水线中的每个阶段设置最佳的批处理大小，以平衡延迟和吞吐量 。

RAGO的工作流程是：首先，利用一个经过校准的性能分析模型，对每个RAG组件在不同资源和批次大小下的性能进行分析 。然后，它会穷举所有可能的“任务放置、资源分配、批处理”组合，计算每种组合的端到端性能，并最终筛选出性能最优的一系列配置



> Towards End-to-End Optimization of LLM-based Applications with Ayo. Asplos'25

* CUHK



> Towards Efficient Compound Large Language Model System Serving in the Wild. IWQoS'2024

* SJTU
* workshop
* optim e2e latency
* Challenge：DAG的不确定性->topology;exec duration;
* motivation：“信息的生成（llm planer）”本身就是一个关键的、需要被优先保障的计算过程 。优先生成DAG方便调度后续步骤，提高资源利用率。
* Solution：**PS-TCS** (Priority-based Scheduling with Topological Complexity Sensing)，不同类型的APP的不确定性不一样，优先调度不确定性最高的

> Circinus: Efficient Query Planner for Compound ML Serving. ArXiv:2504

* UIUC
* not open-source
* keywords: slo aware edge-cloud schedule; agent serving; optim slo throughput;
* brainstorm
  * no edge? just device + cloud?
  * no pipeline concurrency detail

> Efficient Serving of LLM Applications with Probabilistic Demand Modeling. Arxiv:2506

* SJTU & HUAWEI CLOUD
* not open-source
* keywords: Probabilistic Demand Graph; LLM app serving; optimize e2e latency；
* 问题：
  * 队列调度效率低下，因为不知道任务的执行时间只能当做黑盒
  * DAG动态性，后端容器预热延迟问题
* insight：
  * 模块    输入输出以及并行度的稳定性，高斯分布
  * 仅仅利用离线测量的均值不够，输入输出与之前的组件有关
* 核心思想：建模DAG图，预测每个组件的资源需求，以及其后续依赖的概率关系，优化请求调度顺序以及容器预热判断
* method
  * DAG建模：offline记录多个原始值，使用蒙特卡洛生成组件资源预计消耗；三种跨单元需求相关性模式，皮尔森相关分析，Online时当单元执行完成后，PDGraph进行条件预测优化，调整下游需求预测估计；
  * PDGraph-based Queue Management：
* brainstorm：
  * 端侧多模型切换的问题
  * 组件利用频率的问题；预热有效性概率

> LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications. arXiv:2504

* SJTU
* not open-source
* keywords: DAG scheduling; optimize e2e latency；
* background
  * schedule Challenge：DAG uncertainty->topology; node exec duration;&#x20;
  * SFJ is useful to reduce ave latecy, but topology uncertainty makes wrong decision.
  * compound type
    * #### Predefined applications, fixed (rag)
    * #### chain like (react, iteration)
    * #### plan app (task-automation)
  * #### key insight: scheduling stages that <mark style="color:red;">reduce uncertainty</mark>, we can obtain valuable information once they’re completed.
  * #### design
    * #### DAG define
      * Regular Stage, llm stage, <mark style="color:red;">dynamic stage</mark>&#x20;
    * #### BN-based Profiler
      * #### offline, to identify stage reduces uncertainty
      * motivation: heatmaps show relation between stages, leverage inter-stage correlations
    * #### Entropy-based Uncertainty Quantification
      * #### to calculate which stage to schedule first
    * #### Uncertainty-aware Scheduler
      * **ε-greedy to mix schedule of SRFT and uncertainty schedule**
  * **brainstorm**
    * **only consider schdule layer, lacks fine-grained batch policy**

> Optimizing Sequential Multi-Step Tasks with Parallel LLM Agents. ICML 2025 Workshop on MAS





> Towards Resource-Efficient Compound AI Systems. HOTOS 25

* MIT & Azure Research
* not open source
* keyword:&#x20;
  * focus on resource utilization
  * AI Workflows-as-a-Service (AIWaaS) design, similar to Faas. 选择最优模型、配置和调度GPU/CPU资源、控制成本、保证质量。
  * **core: Workflow-Aware Cluster Management -> Cost/latency/quality/Energy...**

> KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. Arxiv:2507

* UCSD & AWS
* not open source
* keyword: multi-agent serve, prefix kv cache, <mark style="color:red;">optim prefill latency</mark>
* <mark style="color:red;">key insight: cache the agent fixed prefix prompt shared by requests and consider workflow info to reduce cache miss to reduce prefilll latency</mark>
* backgroud
  * agent kv cache(_**tree-based**_)，fixed part(large, agent’s role, behavioral instructions, task description, and few-shot learning examples) +task-specific dynamic part(user inpput, small)， what we cache? _**KV of the fixed parts**_
  * different user lead to different fiexed part for the same agent, eg 2 executor instruct
  * LRU not fits agentic workflow, can not capture workflow info, leads to cache miss
* prefix cache management for agentic workflows
  * a workflow-aware eviction policy
    * DAG无法描述multi agent中的分支关系，是AND还是OR，无法准确计算该node计算是在后面几个位置，从而kv cache的优先级无法确认
    * 提出Agent Step Graph abstraction，**agent invocation as node level.** agents with larger steps-to-execution are more likely to be evicted.
  * overlapped KV prefetching mechanism
    * proactivate offload kvcache, like infinigen
* brainstrom
  * only optim llm prefill latency. useful for long sequence, but as the number of output tokens increases, the relative gain from KVFlow diminishes.
  * **not consider tool call time**, which may be the critical path

> RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation. arXiv:2404

* PKU & ByteDance
* not open-source
* keyword: rag prefix cache, serving, multi-level cache system, <mark style="color:red;">optim prefill latency</mark>
* <mark style="color:red;">key insight: cache the prefix kv-cache of the retrival</mark> <mark style="color:red;"></mark><mark style="color:red;">**hot**</mark> <mark style="color:red;"></mark><mark style="color:red;">doc to reduce prefill latency</mark>
* background
  * rag bottleneck lies in prefill step for long seq len rag
  * recurrence of identical _**hot**_ documents across multiple requests
* method
  * Greedy-Dual-Size-Frequency (PGDSF) **replacement policy** to minimize cache miss
    * tree based; cost aware;
  * cache-aware request reordering
    * priority queue; cached length / recompute length
  * dynamic speculative pipelining.
    * vector search may produce the final results early in the retrieval step
    * overlap the retrieval and generation steps
* brainstorm
  * not consider different rag config, not all situation bottleneck lies in prefill
  * 0.3M docs in vector database, top-1 or 2 selection, is a representive config?
  * may be consider agent workflow info? powerinfer? on device?

> HeterRAG: Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation. ISCA ’25

* HUST & NUS
* not open-source
* keywords: PIM based rag system; HBM-based PIM + DIMM-based PIM;



> Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC. arxiv:2506

* PKU
* not open-source
* keywords: heterogeneous SoCs; PD disaggregate; on-device; Intel Core Ultra with llama-3B;
* background
  * **端侧两种workload，一种被动，一种主动；reactive-latency， proactive-throughput**
  * 端侧上资源争用；SOC特性（如NPU）与LLM不匹配

### Idea

8卡机之间nvlink共享显存，不同组件之间充分利用能力, aqua like schudling，optim placement；Fair schedule；agent上下文迁移看看有什么能做的

multi-agent

GPU 拆分？multiplexing? nvidia-green split in on device senario

muti agent kv-cache + aqua

serveless + agent ?

could + edge workflow optim

dynamic resource allocation and load balancing做这个？现在好像都是静态的

利用概率建模加速Agent kv cache复用

