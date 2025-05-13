---
title: "Interested Paper"
author : "tutu"
description:
date: '2025-03-08'
lastmod: '2025-03-08'
image:
math: true
hidden: false
comments: true
draft: false
categories : ['paper reading']
---


`M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism,” Mar. 13, 2020, _arXiv_: arXiv:1909.08053. doi: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053).`

- NLP发展--考虑阅读公众号
- http://arxiv.org/abs/1606.08415 gelu，gpt-2  bert
- Albert: A lite bert for self-supervised learning of language representations bert改进
- waek scaling

[1]

S. Choi, I. Koo, J. Ahn, M. Jeon, and Y. Kwon, “{EnvPipe}: Performance-preserving {DNN} Training Framework for Saving Energy,” presented at the 2023 USENIX Annual Technical Conference (USENIX ATC 23), 2023, pp. 851–864. Accessed: Oct. 24, 2024. [Online]. Available: [https://www.usenix.org/conference/atc23/presentation/choi](https://www.usenix.org/conference/atc23/presentation/choi)


A. Faiz _et al._, “LLMCarbon: Modeling the end-to-end Carbon Footprint of Large Language Models,” Jan. 19, 2024, _arXiv_: arXiv:2309.14393. doi: [10.48550/arXiv.2309.14393](https://doi.org/10.48550/arXiv.2309.14393).


`A. K. Kakolyris, D. Masouros, P. Vavaroutsos, S. Xydis, and D. Soudris, “SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving,” Aug. 05, 2024, _arXiv_: arXiv:2408.05235. doi: [10.48550/arXiv.2408.05235](https://doi.org/10.48550/arXiv.2408.05235).`

```raw

- Recent studies indicate that inference dominates the computational workload, accounting for over 90% of the overall LLM compute cycles within data centers

P. Patel, E. Choukse, C. Zhang, ´ I. Goiri, B. Warrier, N. Mahalingam, and R. Bianchini, “Characterizing power management opportunities for llms in the cloud,” in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, 2024, pp. 207–222.

S. Samsi, D. Zhao, J. McDonald, B. Li, A. Michaleas, M. Jones,
W. Bergeron, J. Kepner, D. Tiwari, and V. Gadepally, “From words
to watts: Benchmarking the energy costs of large language model
inference,” in IEEE High Performance Extreme Computing Conference,
HPEC 2023, Boston, MA, USA, September 25-29, 2023. IEEE, 2023,
pp. 1–9. [Online]. Available: https://doi.org/10.1109/HPEC58863.2023.
10363447

- 何为slo
- 为什么prefill是计算密集 

[43] P. Patel, E. Choukse, C. Zhang, A. Shah, I. Goiri, S. Maleki, and R. Bianchini, “Splitwise: Efficient generative llm inference using phase splitting,” in ISCA, June 2024

- decode为什么是memory bound？

A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani,
A. Tumanov, and R. Ramjee, “Taming {Throughput-Latency} tradeoff
in {LLM} inference with {Sarathi-Serve},” in 18th USENIX Symposium
on Operating Systems Design and Implementation (OSDI 24), 2024, pp.

- 传统的功率上限[31]和功率超订阅[27]、[41]技术的系统在这种情况下存在不足

[31] S. Li, X. Wang, X. Zhang, V. Kontorinis, S. Kodakara, D. Lo,
and P. Ranganathan, “Thunderbolt: Throughput-optimized, quality-ofservice-aware power capping at scale,” in 14th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2020, Virtual Event, November 4-6, 2020. USENIX Association, 2020, pp. 1241–1255. [Online]. Available: https://www.usenix.org/conference/ osdi20/presentation/li-shaohong

[27] A. G. Kumbhare, R. Azimi, I. Manousakis, A. Bonde, F. V. Frujeri,
N. Mahalingam, P. A. Misra, S. A. Javadi, B. Schroeder, M. Fontoura,
and R. Bianchini, “Prediction-based power oversubscription in cloud
platforms,” in 2021 USENIX Annual Technical Conference, USENIX
ATC 2021, July 14-16, 2021, I. Calciu and G. Kuenning, Eds.
USENIX Association, 2021, pp. 473–487. [Online]. Available:
https://www.usenix.org/conference/atc21/presentation/kumbhare

- stream of request, workload 

[59] Y. Wang, Y. Chen, Z. Li, X. Kang, Z. Tang, X. He, R. Guo, X. Wang,
Q. Wang, A. C. Zhou, and X. Chu, “Burstgpt: A real-world workload
dataset to optimize llm serving systems,” 2024. [Online]. Available:
https://arxiv.org/abs/2401.17644

- inflight batch ,应该是动态批处理

G.-I. Yu et al., “Orca: A distributed serving system for TransformerBased generative models,” in 16th USENIX Symposium on Operating
Systems Design and Implementation (OSDI 22). Carlsbad, CA:
USENIX Association, Jul. 2022, pp. 521–538. [Online]. Available:
https://www.usenix.org/conference/osdi22/presentation/yu

- “shadow instancing” 避免scale的开销（如启动服务器）

[12] M. Chow, A. Jahanshahi, and D. Wong, “Krisp: Enabling kernel-wise
right-sizing for spatial partitioned gpu inference servers,” in 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2023, pp. 624–637.

[18] A. Dhakal, S. G. Kulkarni, and K. Ramakrishnan, “Gslice: controlled
spatial sharing of gpus for a scalable inference platform,” in Proceedings of the 11th ACM Symposium on Cloud Computing, 2020, pp. 492–506
```


B. Li, S. Samsi, V. Gadepally, and D. Tiwari, “Clover: Toward Sustainable AI with Carbon-Aware Machine Learning Inference Service,” in _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_, in SC ’23. New York, NY, USA: Association for Computing Machinery, Nov. 2023, pp. 1–15. doi: [10.1145/3581784.3607034](https://doi.org/10.1145/3581784.3607034).


J. Stojkovic, C. Zhang, Í. Goiri, J. Torrellas, and E. Choukse, “DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency,” Aug. 01, 2024, _arXiv_: arXiv:2408.00741. doi: [10.48550/arXiv.2408.00741](https://doi.org/10.48550/arXiv.2408.00741).

`infinigen`

```raw
FlexGen: Highthroughput generative inference of large language models with a single gpu.



```

`B. Li, S. Samsi, V. Gadepally, and D. Tiwari, “Clover: Toward Sustainable AI with Carbon-Aware Machine Learning Inference Service,” in _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_, in SC ’23. New York, NY, USA: Association for Computing Machinery, Nov. 2023, pp. 1–15. doi: [10.1145/3581784.3607034](https://doi.org/10.1145/3581784.3607034).`

```raw
P. Patel, Z. Gong, S. Rizvi, E. Choukse, P. Misra, T. Anderson, and
A. Sriraman, “Towards improved power management in cloud gpus,”
IEEE Computer Architecture Letters, 2023.

```