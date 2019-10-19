#include "tensor4.h"


struct VGG
{
	t4::tensor4f features_0_weight;
	t4::tensor1f features_0_bias;
	t4::tensor4f features_2_weight;
	t4::tensor1f features_2_bias;
	t4::tensor4f features_5_weight;
	t4::tensor1f features_5_bias;
	t4::tensor4f features_7_weight;
	t4::tensor1f features_7_bias;
	t4::tensor4f features_10_weight;
	t4::tensor1f features_10_bias;
	t4::tensor4f features_12_weight;
	t4::tensor1f features_12_bias;
	t4::tensor4f features_14_weight;
	t4::tensor1f features_14_bias;
	t4::tensor4f features_17_weight;
	t4::tensor1f features_17_bias;
	t4::tensor4f features_19_weight;
	t4::tensor1f features_19_bias;
	t4::tensor4f features_21_weight;
	t4::tensor1f features_21_bias;
	t4::tensor4f features_24_weight;
	t4::tensor1f features_24_bias;
	t4::tensor4f features_26_weight;
	t4::tensor1f features_26_bias;
	t4::tensor4f features_28_weight;
	t4::tensor1f features_28_bias;
	t4::tensor2f classifier_0_weight;
	t4::tensor1f classifier_0_bias;
	t4::tensor2f classifier_3_weight;
	t4::tensor1f classifier_3_bias;
	t4::tensor2f classifier_6_weight;
	t4::tensor1f classifier_6_bias;
};


VGG VGGLoad(const char* filename);

t4::tensor2f VGGForward(const VGG& ctx, t4::tensor4f xinput_1);
