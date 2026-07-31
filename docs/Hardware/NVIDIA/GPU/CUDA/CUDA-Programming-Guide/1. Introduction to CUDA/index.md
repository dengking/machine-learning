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

Like any programming model, CUDA relies on a conceptual model of the underlying hardware. For the purposes of CUDA programming, the GPU can be considered to be a collection of *Streaming Multiprocessors* (SMs) which are organized into groups called *Graphics Processing Clusters* (GPCs). Each SM contains:

- a **local register file**

- **unified data cache**

- a number of **functional units**(对应下图中的SM的小格子) that perform computations

The **unified data cache** provides the physical resources for *shared memory* and L1 cache. The allocation of the **unified data cache** to L1 and shared memory can be configured at runtime. The sizes of different types of memory and the number of **functional units** within an SM can vary across GPU architectures.

[![The CUDA programming model view of CPU and GPU components and connection](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)

Figure 2 A GPU has many streaming multiprocessors (SMs), each of which contains many functional units. Graphics processing clusters (GPCs) are collections of SMs. A GPU is a set of GPCs connected to the GPU memory. A CPU typically has several cores and a memory controller which connects to the system memory. A CPU and a GPU are connected by an interconnect such as PCIe or NVLINK.

#### Q: 上述GPU hardware model中，SM中的小格子代表的是什么？

A: **functional unit**，它负责实际的computation

### 1.2.2.1. Thread Blocks and Grids

When an application launches a kernel, it does so with many threads, often millions of threads. These threads are organized into blocks. A block of threads is referred to, perhaps unsurprisingly, as a *thread block*. **Thread blocks** are organized into a *grid*. All the **thread blocks** in a grid have the same **size** and **dimensions**. [Figure 3](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-hierarchy-grid-of-thread-blocks) shows an illustration of a grid of thread blocks.

[![Grid of Thread Blocks](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)

Figure 3 Grid of Thread Blocks. Each arrow represents a thread (the number of arrows is not representative of actual number of threads).

#### why need dimension?

Thread blocks and grids may be 1, 2, or 3 dimensional. These dimensions can simplify mapping of individual threads to units of work or data items.

#### execution configuration

When a kernel is launched, it is launched using a specific *execution configuration* which specifies the grid and thread block dimensions. The execution configuration may also include optional parameters such as cluster size, stream, and SM configuration settings, which will be introduced in later sections.

#### location

Using **built-in variables**, each thread executing the kernel can determine its location within its containing block and the location of its block within the containing grid. A thread can also use these built-in variables to determine the dimensions of the thread blocks and the grid on which the kernel was launched. This gives each thread a unique identity among all the threads running the kernel. This identity is frequently used to determine what data or operations a thread is responsible for.

#### execution: thread block<->SM

All threads of a **thread block** are executed in a single **SM**. This allows threads within a thread block to communicate and synchronize with each other efficiently. Threads within a **thread block** all have access to the **on-chip shared memory**, which can be used for exchanging information between threads of a thread block.

A grid may consist of millions of **thread blocks**, while the GPU executing the grid may have only tens or hundreds of SMs. All threads of a thread block are executed by a single SM and, in most cases [[1]](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#fn-non-completion), run to completion on that SM. There is no guarantee of scheduling between **thread blocks**, so a **thread block** cannot rely on results from other **thread blocks**, as they may not be able to be scheduled until that **thread block** has completed. [Figure 4](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-scheduling) shows an example of how thread blocks from a grid are assigned to an SM.

[![Thread blocks scheduled on SMs](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling.png)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling.png)

Figure 4 Each SM has one or more active thread blocks. In this example, each SM has three **thread blocks** scheduled simultaneously. There are no guarantees about the order in which thread blocks from a grid are assigned to SMs.

The **CUDA programming model** enables arbitrarily large grids to run on GPUs of any size, whether it has only one SM or thousands of SMs. To achieve this, the **CUDA programming model**, with some exceptions, requires that there be no **data dependencies** between threads in different thread blocks. That is, a thread should not depend on results from or synchronize with a thread in a different **thread block** of the same grid. All the threads within a **thread block** run on the same SM at the same time. Different thread blocks within the grid are scheduled among the available SMs and may be executed in any order. In short, the CUDA programming model requires that it be possible to execute thread blocks in any order, in parallel or in series.

#### 1.2.2.1.1. Thread Block Clusters



### 1.2.2.2. Warps(术) and SIMT

Within a thread block, threads are organized into groups of 32 threads called *warps*. A warp executes the kernel code in a *Single-Instruction Multiple-Threads* (SIMT) paradigm. In SIMT, all threads in the warp are executing the same **kernel code**, but each thread may follow different branches through the code. That is, though all threads of the program execute the same code, threads do not need to follow the same **execution path**.

When threads are executed by a warp, they are assigned a **warp lane**. Warp lanes are numbered 0 to 31 and threads from a thread block are assigned to warps in a predictable fashion detailed in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-hardware-multithreading).

#### warp divergence

All threads in the warp execute the same instruction simultaneously. If some threads within a warp follow a **control flow** branch in execution while others do not, the threads which do not follow the branch will be masked off while the threads which follow the branch are executed. For example, if a conditional is only true for half the threads in a warp, the other half of the warp would be masked off while the active threads execute those instructions. This situation is illustrated in [Figure 7](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#active-warp-lanes). When different threads in a warp follow different code paths, this is sometimes called **warp divergence**. It follows that utilization of the GPU is maximized when threads within a warp follow the same control flow path.

[![Warp lanes are masked off when not active](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/active-warp-lanes.png)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/active-warp-lanes.png)

Figure 7 In this example, only threads with even thread index execute the body of the if statement, the others are masked off while the body is executed.



In the SIMT model, all threads in a warp progress through the kernel in lock step. Hardware execution may differ. See the sections on [Independent Thread Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-independent-thread-scheduling) for more information on where this distinction is important. Exploiting knowledge of how warp execution is actually mapped to real hardware is discouraged. The CUDA programming model and SIMT say that all threads in a warp progress through the code together. Hardware may optimize masked lanes in ways that are transparent to the program so long as the programming model is followed. If the program violates this model, this can result in undefined behavior that can be different in different GPU hardware.





### 1.2.2.3. Tile Programming in CUDA

In addition to the **SIMT model** described in the preceding sections, CUDA supports a **tile programming model**. In tile programming, the programmer writes code at the level of an entire **thread block**, describing operations on multidimensional collections of data called **tiles**. The compiler maps these operations to the individual threads of the block.

> NOTE: 操作单位不同

**Tile kernels** are launched on a **grid of blocks**, as described in the [Thread Blocks and Grids](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#programming-model-threads-grids) section. Each block executes the **tile kernel** and can query its position within the grid to determine which portion of the data it is responsible for. The programmer specifies only the **grid dimensions**; the number of threads per block is determined by the compiler based on the **tile operations** in the kernel ([Figure 8](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#figure-tile-programming-abstraction)).

[![Programmer's view in the SIMT and tile programming models](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/tile-simt.png)](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/tile-simt.png)

Figure 8 Programmer’s view in the **SIMT** and **tile programming models**. In SIMT, the programmer writes **per-thread code** and controls how each thread accesses data. In tile programming, the programmer writes **per-block code** that operates on **tiles**; the compiler maps operations to the threads of the block.

Within a tile kernel, the block executes a **single control flow**. The programmer specifies operations on tiles, and the compiler distributes the work across the threads of the block. Standard control flow constructs such as conditionals and loops are supported, but because the block follows a **single control flow**, there is no concept of **warp divergence**. Scalar operations, such as computing an index or a loop bound, are executed by a single thread of the block. Tile operations, such as adding two tiles element by element, are collectively executed in parallel by all threads of the block.

#### tile VS block

It is important not to confuse blocks—units of execution—with tiles—units of data. A single block may create and operate on many tiles of different shapes and data types.

翻译: 切勿将**线程块**（执行单元）与**分块瓦片**（数据单元）混淆。单个线程块可创建多种形状、不同数据类型的瓦片并对其进行运算。

#### 1.2.2.3.1. Arrays and tiles

Tile kernels work with two types of data: **arrays** and **tiles**. An array (or global array) is a multidimensional container of elements stored in device memory. Arrays are mutable: their contents can be modified by store operations within a kernel. An array has a shape and a data type.

A tile is a multidimensional collection of values that exists only within **tile code** and is local to a single block. **Tiles** are immutable: every operation on a tile produces a new tile rather than modifying an existing one. Unlike an array, a tile does not necessarily have a representation in memory—the compiler decides how tile data is stored, and may use registers, shared memory, or other resources of the SM. Each dimension of a tile must be a power of two and must be known at **compile time** (that is, its value must be determinable before the kernel executes, rather than computed during execution). Tiles cannot be passed as kernel parameters; they are created and consumed entirely within **tile code**.
