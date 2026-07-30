# [1. Introduction to CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/part1.html)

A GPU provides much higher instruction throughput and memory bandwidth than a CPU within a similar price and power envelope.

翻译: 在成本与功耗相近的前提下，GPU 的指令吞吐率与内存带宽远高于 CPU

## 1.1.2. The Benefits of Using GPUs

### GPUs VS CPUs

GPUs and CPUs are designed with different goals in mind. While a CPU is designed to excel at executing a serial sequence of operations (called a thread) as fast as possible and can execute a few tens of these threads in parallel, a GPU is designed to excel at executing thousands of threads in parallel, trading off lower single-thread performance to achieve much greater total throughput.


