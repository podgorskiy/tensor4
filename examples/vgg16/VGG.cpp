#include "VGG.h"


VGG VGGLoad(const char* filename)
{
	VGG ctx;
	t4::model_dict dict = t4::load(filename);
	dict.load(ctx.features_0_weight, "features.0.weight", 64, 3, 3, 3);
	dict.load(ctx.features_0_bias, "features.0.bias", 64);
	dict.load(ctx.features_2_weight, "features.2.weight", 64, 64, 3, 3);
	dict.load(ctx.features_2_bias, "features.2.bias", 64);
	dict.load(ctx.features_5_weight, "features.5.weight", 128, 64, 3, 3);
	dict.load(ctx.features_5_bias, "features.5.bias", 128);
	dict.load(ctx.features_7_weight, "features.7.weight", 128, 128, 3, 3);
	dict.load(ctx.features_7_bias, "features.7.bias", 128);
	dict.load(ctx.features_10_weight, "features.10.weight", 256, 128, 3, 3);
	dict.load(ctx.features_10_bias, "features.10.bias", 256);
	dict.load(ctx.features_12_weight, "features.12.weight", 256, 256, 3, 3);
	dict.load(ctx.features_12_bias, "features.12.bias", 256);
	dict.load(ctx.features_14_weight, "features.14.weight", 256, 256, 3, 3);
	dict.load(ctx.features_14_bias, "features.14.bias", 256);
	dict.load(ctx.features_17_weight, "features.17.weight", 512, 256, 3, 3);
	dict.load(ctx.features_17_bias, "features.17.bias", 512);
	dict.load(ctx.features_19_weight, "features.19.weight", 512, 512, 3, 3);
	dict.load(ctx.features_19_bias, "features.19.bias", 512);
	dict.load(ctx.features_21_weight, "features.21.weight", 512, 512, 3, 3);
	dict.load(ctx.features_21_bias, "features.21.bias", 512);
	dict.load(ctx.features_24_weight, "features.24.weight", 512, 512, 3, 3);
	dict.load(ctx.features_24_bias, "features.24.bias", 512);
	dict.load(ctx.features_26_weight, "features.26.weight", 512, 512, 3, 3);
	dict.load(ctx.features_26_bias, "features.26.bias", 512);
	dict.load(ctx.features_28_weight, "features.28.weight", 512, 512, 3, 3);
	dict.load(ctx.features_28_bias, "features.28.bias", 512);
	dict.load(ctx.classifier_0_weight, "classifier.0.weight", 4096, 25088);
	dict.load(ctx.classifier_0_bias, "classifier.0.bias", 4096);
	dict.load(ctx.classifier_3_weight, "classifier.3.weight", 4096, 4096);
	dict.load(ctx.classifier_3_bias, "classifier.3.bias", 4096);
	dict.load(ctx.classifier_6_weight, "classifier.6.weight", 1000, 4096);
	dict.load(ctx.classifier_6_bias, "classifier.6.bias", 1000);
	return ctx;
}


t4::tensor2f VGGForward(const VGG& ctx, t4::tensor4f x0)
{
	t4::tensor4f x33 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x0, ctx.features_0_weight, ctx.features_0_bias); //features.0
	t4::release(x0);
	t4::tensor4f x34 = t4::Relu(x33); //features.1
	t4::release(x33);
	t4::tensor4f x35 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x34, ctx.features_2_weight, ctx.features_2_bias); //features.2
	t4::release(x34);
	t4::tensor4f x36 = t4::Relu(x35); //features.3
	t4::release(x35);
	t4::tensor4f x37 = t4::MaxPool2d<2, 2, 2, 2, 0, 0>(x36); //features.4
	t4::release(x36);
	t4::tensor4f x38 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x37, ctx.features_5_weight, ctx.features_5_bias); //features.5
	t4::release(x37);
	t4::tensor4f x39 = t4::Relu(x38); //features.6
	t4::release(x38);
	t4::tensor4f x40 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x39, ctx.features_7_weight, ctx.features_7_bias); //features.7
	t4::release(x39);
	t4::tensor4f x41 = t4::Relu(x40); //features.8
	t4::release(x40);
	t4::tensor4f x42 = t4::MaxPool2d<2, 2, 2, 2, 0, 0>(x41); //features.9
	t4::release(x41);
	t4::tensor4f x43 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x42, ctx.features_10_weight, ctx.features_10_bias); //features.10
	t4::release(x42);
	t4::tensor4f x44 = t4::Relu(x43); //features.11
	t4::release(x43);
	t4::tensor4f x45 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x44, ctx.features_12_weight, ctx.features_12_bias); //features.12
	t4::release(x44);
	t4::tensor4f x46 = t4::Relu(x45); //features.13
	t4::release(x45);
	t4::tensor4f x47 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x46, ctx.features_14_weight, ctx.features_14_bias); //features.14
	t4::release(x46);
	t4::tensor4f x48 = t4::Relu(x47); //features.15
	t4::release(x47);
	t4::tensor4f x49 = t4::MaxPool2d<2, 2, 2, 2, 0, 0>(x48); //features.16
	t4::release(x48);
	t4::tensor4f x50 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x49, ctx.features_17_weight, ctx.features_17_bias); //features.17
	t4::release(x49);
	t4::tensor4f x51 = t4::Relu(x50); //features.18
	t4::release(x50);
	t4::tensor4f x52 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x51, ctx.features_19_weight, ctx.features_19_bias); //features.19
	t4::release(x51);
	t4::tensor4f x53 = t4::Relu(x52); //features.20
	t4::release(x52);
	t4::tensor4f x54 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x53, ctx.features_21_weight, ctx.features_21_bias); //features.21
	t4::release(x53);
	t4::tensor4f x55 = t4::Relu(x54); //features.22
	t4::release(x54);
	t4::tensor4f x56 = t4::MaxPool2d<2, 2, 2, 2, 0, 0>(x55); //features.23
	t4::release(x55);
	t4::tensor4f x57 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x56, ctx.features_24_weight, ctx.features_24_bias); //features.24
	t4::release(x56);
	t4::tensor4f x58 = t4::Relu(x57); //features.25
	t4::release(x57);
	t4::tensor4f x59 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x58, ctx.features_26_weight, ctx.features_26_bias); //features.26
	t4::release(x58);
	t4::tensor4f x60 = t4::Relu(x59); //features.27
	t4::release(x59);
	t4::tensor4f x61 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x60, ctx.features_28_weight, ctx.features_28_bias); //features.28
	t4::release(x60);
	t4::tensor4f x62 = t4::Relu(x61); //features.29
	t4::release(x61);
	t4::tensor4f x63 = t4::MaxPool2d<2, 2, 2, 2, 0, 0>(x62); //features.30
	t4::release(x62);
	t4::tensor2f x64 = t4::Flatten<1>(x63);
	t4::release(x63);
	t4::tensor2f x65 = t4::Linear(x64, ctx.classifier_0_weight, ctx.classifier_0_bias); //classifier.0
	t4::release(x64);
	t4::tensor2f x66 = t4::Relu(x65); //classifier.1
	t4::release(x65);
	t4::tensor2f x67 = t4::Dropout(x66, 0.5f); //classifier.2
	t4::release(x66);
	t4::tensor2f x69 = t4::Linear(x67, ctx.classifier_3_weight, ctx.classifier_3_bias); //classifier.3
	t4::release(x67);
	t4::tensor2f x70 = t4::Relu(x69); //classifier.4
	t4::release(x69);
	t4::tensor2f x71 = t4::Dropout(x70, 0.5f); //classifier.5
	t4::release(x70);
	t4::tensor2f x73 = t4::Linear(x71, ctx.classifier_6_weight, ctx.classifier_6_bias); //classifier.6
	t4::release(x71);
	return x73;
}
