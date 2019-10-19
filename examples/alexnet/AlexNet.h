#include "tensor4.h"


struct AlexNet
{
	t4::tensor4f features_0_weight;
	t4::tensor1f features_0_bias;
	t4::tensor4f features_3_weight;
	t4::tensor1f features_3_bias;
	t4::tensor4f features_6_weight;
	t4::tensor1f features_6_bias;
	t4::tensor4f features_8_weight;
	t4::tensor1f features_8_bias;
	t4::tensor4f features_10_weight;
	t4::tensor1f features_10_bias;
	t4::tensor2f classifier_1_weight;
	t4::tensor1f classifier_1_bias;
	t4::tensor2f classifier_4_weight;
	t4::tensor1f classifier_4_bias;
	t4::tensor2f classifier_6_weight;
	t4::tensor1f classifier_6_bias;
};


AlexNet AlexNetLoad(const char* filename);

t4::tensor2f AlexNetForward(const AlexNet& ctx, t4::tensor4f xinput_1);
