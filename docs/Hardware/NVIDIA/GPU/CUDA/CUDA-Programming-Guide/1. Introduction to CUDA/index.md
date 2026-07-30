# [1. Introduction to CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/part1.html)

A GPU provides much higher instruction throughput and memory bandwidth than a CPU within a similar price and power envelope.

翻译: 在成本与功耗相近的前提下，GPU 的指令吞吐率与内存带宽远高于 CPU

## 1.1.2. The Benefits of Using GPUs

### GPUs VS CPUs

> NOTE: 在"Architecture&Computation&DNN"中也对这个topic进行了讨论

GPUs and CPUs are designed with different goals in mind. While a CPU is designed to excel at executing a serial sequence of operations (called a thread) as fast as possible and can execute a few tens of these threads in parallel, a GPU is designed to excel at executing thousands of threads in parallel, trading off lower single-thread performance to achieve much greater total throughput.

## 1.1.3. Getting Started Quickly

An ever-growing collection of algorithms and routines from a variety of domains is available through specialized libraries. When a library has already been implemented—especially those provided by NVIDIA—using it is often more productive and performant than reimplementing algorithms from scratch. Libraries like cuBLAS, cuFFT, cuDNN, and CUTLASS are just a few examples of libraries that help developers avoid reimplementing well-established algorithms. These libraries have the added benefit of being optimized for each GPU architecture, providing an ideal mix of productivity, performance, and portability.

# 1.2. Programming Model

## 1.2.1. Heterogeneous(异构) Systems

The CUDA programming model assumes a heterogeneous(异构) computing system, which means a system that includes both GPUs and CPUs. The CPU and the memory directly connected to it are called the *host* and *host memory*, respectively. A GPU and the memory directly connected to it are referred to as the *device* and *device memory*, respectively. In some system-on-chip (SoC) systems, these may be part of a single package. In larger systems, there may be multiple CPUs or GPUs.

CUDA applications execute some part of their code on the GPU, but applications always start execution on the CPU. The host code, which is the code that runs on the CPU, can use CUDA APIs to copy data between the **host memory** and **device memory**, start code executing on the GPU, and wait for data copies or GPU code to complete. The CPU and GPU can both be executing code simultaneously, and best performance is usually found by maximizing utilization of both CPUs and GPUs.

### Kernel

The code an application executes on the GPU is referred to as *device code*, and a function that is invoked for execution on the GPU is, for historical reasons, called a *kernel*. The act of starting a kernel running is called *launching* the kernel. A **kernel launch** can be thought of as starting many threads executing the **kernel code** in parallel on the GPU. GPU threads operate similarly to threads on CPUs, though there are some differences important to both correctness and performance that will be covered in later sections (see [Section 3.2.2.1.1](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling)).

### 总结

|     |        | memory        | code        |
| --- | ------ | ------------- | ----------- |
| CPU | host   | host memory   |             |
| GPU | device | device memory | device code |

## 1.2.2. GPU Hardware Model

Like any programming model, CUDA relies on a conceptual model of the underlying hardware. For the purposes of CUDA programming, the GPU can be considered to be a collection of *Streaming Multiprocessors* (SMs) which are organized into groups called *Graphics Processing Clusters* (GPCs). Each SM contains a **local register file**, a **unified data cache**, and a number of **functional units** that perform computations. The **unified data cache** provides the physical resources for *shared memory* and L1 cache. The allocation of the **unified data cache** to L1 and shared memory can be configured at runtime. The sizes of different types of memory and the number of functional units within an SM can vary across GPU architectures.

[![The CUDA programming model view of CPU and GPU components and connection](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)

Figure 2 A GPU has many streaming multiprocessors (SMs), each of which contains many functional units. Graphics processing clusters (GPCs) are collections of SMs. A GPU is a set of GPCs connected to the GPU memory. A CPU typically has several cores and a memory controller which connects to the system memory. A CPU and a GPU are connected by an interconnect such as PCIe or NVLINK.

### 1.2.2.1. Thread Blocks and Grids

When an application launches a kernel, it does so with many threads, often millions of threads. These threads are organized into blocks. A block of threads is referred to, perhaps unsurprisingly, as a *thread block*. Thread blocks are organized into a *grid*. All the thread blocks in a grid have the same size and dimensions. [Figure 3](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-hierarchy-grid-of-thread-blocks) shows an illustration of a grid of thread blocks.

[![Grid of Thread Blocks](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)

Figure 3 Grid of Thread Blocks. Each arrow represents a thread (the number of arrows is not representative of actual number of threads).
