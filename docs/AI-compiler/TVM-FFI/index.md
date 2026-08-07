# [tvm-ffi](https://github.com/apache/tvm-ffi)

**Apache TVM FFI** is an open ABI and FFI for machine learning systems. It is a minimal, framework-agnostic(框架无关), yet flexible open convention with the following systems in mind:

- **Kernel libraries** - ship one wheel to support multiple frameworks, Python versions, and different languages. [[FlashInfer](https://docs.flashinfer.ai/)]
- **Kernel DSLs** - reusable open ABI for JIT and AOT kernel exposure frameworks and runtimes. [[TileLang](https://tilelang.com/)][[cuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html)]
- **Frameworks and runtimes** - a uniform extension point for ABI-compliant libraries and DSLs. [[PyTorch](https://tvm.apache.org/ffi/get_started/quickstart.html#ship-to-pytorch)][[JAX](https://tvm.apache.org/ffi/get_started/quickstart.html#ship-to-jax)][[PaddlePaddle](https://tvm.apache.org/ffi/get_started/quickstart.html#ship-to-paddle)][[NumPy/CuPy](https://tvm.apache.org/ffi/get_started/quickstart.html#ship-to-numpy)]
- **ML infrastructure** - out-of-box bindings and interop across languages. [[Python](https://tvm.apache.org/ffi/get_started/quickstart.html#ship-to-python)][[C++](https://tvm.apache.org/ffi/get_started/quickstart.html#ship-to-cpp)][[Rust](https://tvm.apache.org/ffi/get_started/quickstart.html#ship-to-rust)][[XGrammar](https://github.com/mlc-ai/xgrammar)]
- **Coding agents** - a unified mechanism for shipping generated code in production.



## see also

[构建机器学习系统的开放ABI和FFI标准](https://zhuanlan.zhihu.com/p/81570886474?share_code=hwgF567Z5YkU&utm_psn=1965081659973011118)


