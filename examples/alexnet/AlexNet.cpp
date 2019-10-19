#include "AlexNet.h"


AlexNet AlexNetLoad(const char* filename)
{
	AlexNet ctx;
	t4::model_dict dict = t4::load(filename);
	dict.load(ctx.features_0_weight, "features.0.weight", 64, 3, 11, 11);
	dict.load(ctx.features_0_bias, "features.0.bias", 64);
	dict.load(ctx.features_3_weight, "features.3.weight", 192, 64, 5, 5);
	dict.load(ctx.features_3_bias, "features.3.bias", 192);
	dict.load(ctx.features_6_weight, "features.6.weight", 384, 192, 3, 3);
	dict.load(ctx.features_6_bias, "features.6.bias", 384);
	dict.load(ctx.features_8_weight, "features.8.weight", 256, 384, 3, 3);
	dict.load(ctx.features_8_bias, "features.8.bias", 256);
	dict.load(ctx.features_10_weight, "features.10.weight", 256, 256, 3, 3);
	dict.load(ctx.features_10_bias, "features.10.bias", 256);
	dict.load(ctx.classifier_1_weight, "classifier.1.weight", 4096, 9216);
	dict.load(ctx.classifier_1_bias, "classifier.1.bias", 4096);
	dict.load(ctx.classifier_4_weight, "classifier.4.weight", 4096, 4096);
	dict.load(ctx.classifier_4_bias, "classifier.4.bias", 4096);
	dict.load(ctx.classifier_6_weight, "classifier.6.weight", 1000, 4096);
	dict.load(ctx.classifier_6_bias, "classifier.6.bias", 1000);
	return ctx;
}


t4::tensor2f AlexNetForward(const AlexNet& ctx, t4::tensor4f xinput_1)
{
	t4::tensor4f x17 = t4::Conv2d<11, 11, 4, 4, 2, 2, 1, 1>(xinput_1, ctx.features_0_weight, ctx.features_0_bias); //features.0
	t4::release(xinput_1);
	t4::tensor4f x18 = t4::ReluInplace(x17); //features.1
	t4::release(x17);
	t4::tensor4f x19 = t4::MaxPool2d<3, 3, 2, 2, 0, 0>(x18); //features.2
	t4::release(x18);
	t4::tensor4f x20 = t4::Conv2d<5, 5, 1, 1, 2, 2, 1, 1>(x19, ctx.features_3_weight, ctx.features_3_bias); //features.3
	t4::release(x19);
	t4::tensor4f x21 = t4::ReluInplace(x20); //features.4
	t4::release(x20);
	t4::tensor4f x22 = t4::MaxPool2d<3, 3, 2, 2, 0, 0>(x21); //features.5
	t4::release(x21);
	t4::tensor4f x23 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x22, ctx.features_6_weight, ctx.features_6_bias); //features.6
	t4::release(x22);
	t4::tensor4f x24 = t4::ReluInplace(x23); //features.7
	t4::release(x23);
	t4::tensor4f x25 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x24, ctx.features_8_weight, ctx.features_8_bias); //features.8
	t4::release(x24);
	t4::tensor4f x26 = t4::ReluInplace(x25); //features.9
	t4::release(x25);
	t4::tensor4f x27 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x26, ctx.features_10_weight, ctx.features_10_bias); //features.10
	t4::release(x26);
	t4::tensor4f x28 = t4::ReluInplace(x27); //features.11
	t4::release(x27);
	t4::tensor4f x29 = t4::MaxPool2d<3, 3, 2, 2, 0, 0>(x28); //features.12
	t4::release(x28);
	t4::tensor4f x30 = t4::AveragePool2d<1, 1, 1, 1, 0, 0>(x29); //avgpool
	t4::release(x29);
	t4::tensor0i x31 = t4::Constant<t4::int64>(0);
	t4::tensor1i x32 = t4::Shape(x30);
	t4::tensor0i x33 = t4::Gather(x32, x31);
	t4::release(x32, x31);
	t4::tensor0i x34 = t4::Constant<t4::int64>(9216);
	t4::tensor1i x35 = t4::Unsqueeze<0>(x33);
	t4::release(x33);
	t4::tensor1i x36 = t4::Unsqueeze<0>(x34);
	t4::release(x34);
	t4::tensor1i x37 = t4::Concat<0>(x35, x36);
	t4::release(x35, x36);
	t4::tensor2f x38 = t4::Reshape<2>(x30, x37); //classifier.0
	t4::release(x30, x37);
	t4::tensor2f x39 = t4::Linear(x38, ctx.classifier_1_weight, ctx.classifier_1_bias); //classifier.1
	t4::release(x38);
	t4::tensor2f x40 = t4::ReluInplace(x39); //classifier.3
	t4::release(x39);
	t4::tensor2f x41 = t4::Linear(x40, ctx.classifier_4_weight, ctx.classifier_4_bias); //classifier.4
	t4::release(x40);
	t4::tensor2f x42 = t4::ReluInplace(x41); //classifier.5
	t4::release(x41);
	t4::tensor2f x43 = t4::Linear(x42, ctx.classifier_6_weight, ctx.classifier_6_bias); //classifier.6
	t4::release(x42);
	return x43;
}
