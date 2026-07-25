# naddod [NVIDIA GPU Architecture Evolution: From Volta to Blackwell](https://www.naddod.com/ai-insights/nvidia-gpu-architecture-evolution-from-volta-to-blackwell)

NVIDIA's GPU architecture has evolved through multiple generations, from Volta to Ampere, then to Ada Lovelace, Hopper, and Blackwell, each specifically optimized for the ever-growing and diverse data center workloads. These architectures have not only continuously improved compute performance, interconnect bandwidth, and memory efficiency, but also achieved greater scalability and energy efficiency in scenarios such as AI inference, training, HPC, and graphics rendering.  

## Volta: The First Introduction of Tensor Cores

The Volta architecture marks a significant starting point for NVIDIA GPUs' transformation towards AI computing. This architecture was the first to introduce Tensor Cores, specifically designed to accelerate matrix multiply-accumulate (GEMM) operations common in deep learning, improving training efficiency at **FP16 precision** from a hardware perspective.  

The addition of Tensor Cores means that GPUs no longer rely entirely on traditional CUDA Cores for general-purpose operations in AI computing, laying the foundation for more explicitly AI-specific designs in subsequent architectures.  

Core technological innovations include:

- Tensor Cores: The Volta architecture integrates 640 Tensor Cores on a single GPU, providing over 100 TFLOPS of deep learning computing power, more than five times that of the previous generation NVIDIA Pascal architecture.

- NVIDIA NVLink: Volta introduces NVIDIA NVLink high-speed interconnect technology, enabling faster results delivery. Compared to the previous generation NVLink, its throughput is increased by 2x. It provides a more efficient communication foundation for model parallelism and data parallelism in multi-GPU systems, helping to improve the overall scalability of large-scale computing tasks.

- Software Optimized for Volta: The Volta architecture, combined with optimized CUDA and the NVIDIA deep learning software stack (such as cuDNN, NCCL, and TensorRT), allows mainstream deep learning frameworks and applications to fully leverage their hardware capabilities. This hardware-software co-design lowers the development threshold and improves practical computing efficiency in data centers and research scenarios.
   

The introduction of the Volta architecture transformed GPUs from general-purpose accelerators into AI-dedicated computing platforms, laying the foundation for subsequent Turing and Ampere architectures.  

## Turing: The Fusion of AI Inference and Graphics Computing

Turing is NVIDIA's GPU architecture designed for AI inference, graphics, and professional computing scenarios. Representative products include the GeForce RTX 20 series and the data center T4 Tensor Core GPU. This architecture builds upon Volta Tensor Core technology, further expanding inference capabilities and graphics computing efficiency.  

ey innovations include:

- **Second-generation Tensor Cores:** Optimized AI inference performance, supporting accelerated INT8/INT4 precision inference.



## Ampere: Scaling Up AI Training

Ampere is NVIDIA's GPU architecture for data centers and AI training, with the A100 Tensor Core GPU as its flagship product. This architecture features comprehensive upgrades in performance, interconnectivity, and accuracy support, making it a mainstream computing platform for AI training and HPC systems.  



Key innovations include:



- Third-generation Tensor Core: NVIDIA Tensor Core technology was first used in the NVIDIA Volta architecture. Building upon these innovations, the NVIDIA Ampere architecture employs Tensor Float 32 (TF32) and 64-bit floating-point (FP64) to accelerate large-scale AI model training.
  

- Third-generation NVLink: The Ampere architecture integrates third-generation NVLink technology, enabling higher-bandwidth direct interconnects between GPUs. Interconnect bandwidth between single GPUs can be increased to 600 GB/s. When used in conjunction with NVIDIA NVSwitch, all GPUs within a server can achieve full interconnect communication via NVLink, meeting the high-speed data exchange requirements of large-scale parallel computing.
  

- HBM2e High-Bandwidth Memory: Ampere adopts the HBM2e high-bandwidth memory solution, further improving memory bandwidth and data access efficiency.
  

- Mixed-precision training optimization: By optimizing mixed-precision computation, Ampere can accelerate the training process and reduce model migration and optimization costs without modifying the original FP32 training code.
  

- Data center-oriented system optimization: The Ampere architecture is deeply integrated with CUDA, cuDNN, NCCL, and the complete AI software stack to achieve multi-GPU system-level collaborative computing, optimizing the performance of large-scale AI and HPC workloads.  



## Ada Lovelace: Energy Efficiency Optimization and Inference Performance Enhancements

Key innovations include:

- Fourth-generation Tensor Cores and FP8 Acceleration: Supports mixed-precision computing (FP8/FP16), improving computational efficiency for AI inference while optimizing deep learning inference performance and throughput.  



## Hopper: Born for Large Model Training

NVIDIA Hopper is NVIDIA's ninth-generation data center GPU architecture, launched in 2022 and first appearing in the H100 Tensor Core GPU. The H100, based on the Hopper architecture, is manufactured using TSMC's 4N process and integrates approximately 80 billion transistors, primarily for large-scale AI training, inference, and high-performance computing (HPC) workloads.
 
Core technological innovations include:

- Transformer Engine: The Hopper architecture incorporates a new Transformer Engine into Tensor Cores, enabling mixed-precision acceleration of FP8 and FP16, significantly improving the training and inference performance of Transformer-like models (such as large language models). Compared to its predecessor, this engine delivers significant speedups for specific AI workloads.
  

- DPX Instructions: The new DPX instruction set accelerates dynamic programming algorithms (such as Smith-Waterman and Floyd-Warshall), making these algorithms several times faster on the H100 than previous generation GPUs, benefiting fields such as scientific computing, genome analysis, and path optimization.
  

- Fourth-Generation NVLink and NVSwitch: Hopper introduces fourth-generation NVLink high-speed GPU interconnect technology, supporting high-bandwidth, low-latency communication between multiple GPUs on the same system or across nodes through extensions such as NVSwitch. The H100 GPU achieves up to 900 GB/s bidirectional bandwidth per GPU, far exceeding PCIe Gen5.



- Multi-Instance GPU (MIG) and Confidential Computing: The Hopper architecture further enhances MIG (Multi-Instance GPU) segmentation capabilities and introduces Confidential Computing(机密计算) for the first time in data center GPUs to protect the privacy of running data and models.
  

- HBM3 Memory and Cache Optimization: The H100 configuration offers up to 80 GB of HBM3 (SXM5 version), providing over 3 TB/s of memory bandwidth, and combined with large-capacity L2 cache technology to reduce external memory access, thereby improving data transfer efficiency.  



## Blackwell: Designed for Trillion-Parameter Models

Blackwell, the next-generation architecture following Hopper, was released on March 18, 2024, focusing on further improving AI inference, training efficiency, energy efficiency, and scalability. Blackwell builds upon Hopper in terms of computing resources, interconnect architecture, and hardware-software co-optimization, supporting larger-scale AI model deployments. Named after the renowned mathematician David Harold Blackwell, Blackwell, with its powerful AI computing capabilities and exceptional energy efficiency, is a key cornerstone supporting the training and inference of trillion-parameter models.  

Key technological innovations include:


- Second-generation Transformer Engine: Blackwell introduces the second-generation Transformer Engine, combining a custom Blackwell Tensor Core, TensorRT-LLM, and NeMo framework to simultaneously accelerate training and inference tasks, particularly improving performance and efficiency in Large Language Models (LLM) and Mixture-of-Experts (MoE) models. The engine supports finer-grained micro-tensor scaling and low-precision formats (such as FP4), expanding model size and throughput while maintaining accuracy.
  

- Dual-chip interconnect design and high-density transistors: Blackwell GPUs are manufactured using TSMC's custom 4NP process, integrating approximately 208 billion transistors per GPU. Inter-chip interconnects of up to 10 TB/s combine two reticle-constrained chips into a single unified GPU, significantly improving computational density and architectural scalability.
  

- Advanced Interconnect and Cluster Expansion: Fifth-generation NVLink interconnect and NVLink Switch technology support large-scale interconnects of up to 576 GPUs, providing up to 130 TB/s of bandwidth in a single 72-GPU NVLink domain (NVL72), and enabling efficient aggregation and data communication via the SHARP protocol, thus providing infrastructure-level interconnect performance for ultra-large-scale model training.
  

- Secure and Confidential Computation: Blackwell is the industry's first GPU to support Trusted Execution Environment I/O (TEE-I/O), protecting sensitive data and AI models from unauthorized access at the hardware level while maintaining near-throughput performance in both encrypted and unencrypted modes, providing strong security guarantees for enterprise-grade confidential AI training, inference, and federated learning.
  

- Accelerated Decompression and Data Analytics: The integrated Decompression Engine accesses large-scale memory via a high-bandwidth interconnect (900 GB/s bidirectional bandwidth) with NVIDIA Grace CPUs, accelerating database query and analytics workloads, and supporting common compression formats such as LZ4, Snappy, and Deflate, unlocking data pipeline performance.
  

- Reliability, Availability, and Maintainability (RAS) Engine: The built-in RAS engine uses intelligent prediction and diagnostic capabilities to proactively identify potential failures at the hardware and system levels, reducing downtime and improving overall system availability and service efficiency, making it suitable for large-scale distributed deployment environments.  



## Rubin: System-Level Compute Scaling for the AI Factory Era



Rubin is NVIDIA’s next-generation GPU architecture following Blackwell, built on the Vera Rubin platform and designed to support the computing demands of AI factories in the Transformer era. The architecture is jointly optimized across compute, memory, and interconnect to support high-communication workloads such as MoE, long-context inference, and Agentic AI. Its core objective is to address the sustained throughput bottlenecks in large-scale AI scenarios caused by limitations in compute, memory bandwidth, and communication efficiency.  



> 翻译: 鲁宾（Rubin）是英伟达继布莱克韦尔（Blackwell）之后推出的下一代 GPU 架构，依托维拉・鲁宾（Vera Rubin）计算平台打造，专为 Transformer 时代人工智能工厂的算力需求设计。该架构对**计算单元、显存、高速互联**三大维度做一体化协同优化，可承载混合专家模型（MoE）、超长上下文推理、智能体 AI 等高通信密集型负载。其核心目标是破解大规模 AI 场景下，受算力、显存带宽、通信效率限制而持续存在的吞吐性能瓶颈。


