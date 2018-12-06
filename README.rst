tensor4 - pytorch to C++ convertor using lightweight templated tensor library
================================

This project was born as a fun experiment and can be useful because of it is extreamly lightweight.

Idea:
 * Use pytorch trace to generate C++ code that defines the network.
 * Single header library
 * No dependencies
 * Inference only, no gradients.
 * Easy to use, simple to embed.

What it can do?:
 * Convert most of pytorch graphs to C++ code
 * Can run DenseNet, ResNet, AlexNet, Vgg16.
 * Produces very small binary footprint onto executable. Executable that can run DenseNet is about 100kb.
 
 
 
