# tensorrt_tutorial


When accelerating the inference in TensorFlow with TensorRT (TF-TRT), you may experience problems with tf.estimator and standard allocator (BFC allocator). It may cause crashes and nondeterministic accuracy. If you run into issues, use cuda_malloc as an allocator 

```
export TF_GPU_ALLOCATOR="cuda_malloc"
```

https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html
