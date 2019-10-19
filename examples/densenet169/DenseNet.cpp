#include "DenseNet.h"


DenseNet DenseNetLoad(const char* filename)
{
	DenseNet ctx;
	t4::model_dict dict = t4::load(filename);
	dict.load(ctx.features_conv0_weight, "features.conv0.weight", 64, 3, 7, 7);
	dict.load(ctx.features_norm0_weight, "features.norm0.weight", 64);
	dict.load(ctx.features_norm0_bias, "features.norm0.bias", 64);
	dict.load(ctx.features_norm0_running_mean, "features.norm0.running_mean", 64);
	dict.load(ctx.features_norm0_running_var, "features.norm0.running_var", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_weight, "features.denseblock1.denselayer1.norm1.weight", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_bias, "features.denseblock1.denselayer1.norm1.bias", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_running_mean, "features.denseblock1.denselayer1.norm1.running_mean", 64);
	dict.load(ctx.features_denseblock1_denselayer1_norm1_running_var, "features.denseblock1.denselayer1.norm1.running_var", 64);
	dict.load(ctx.features_denseblock1_denselayer1_conv1_weight, "features.denseblock1.denselayer1.conv1.weight", 128, 64, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_weight, "features.denseblock1.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_bias, "features.denseblock1.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_running_mean, "features.denseblock1.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer1_norm2_running_var, "features.denseblock1.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer1_conv2_weight, "features.denseblock1.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_weight, "features.denseblock1.denselayer2.norm1.weight", 96);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_bias, "features.denseblock1.denselayer2.norm1.bias", 96);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_running_mean, "features.denseblock1.denselayer2.norm1.running_mean", 96);
	dict.load(ctx.features_denseblock1_denselayer2_norm1_running_var, "features.denseblock1.denselayer2.norm1.running_var", 96);
	dict.load(ctx.features_denseblock1_denselayer2_conv1_weight, "features.denseblock1.denselayer2.conv1.weight", 128, 96, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_weight, "features.denseblock1.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_bias, "features.denseblock1.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_running_mean, "features.denseblock1.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer2_norm2_running_var, "features.denseblock1.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer2_conv2_weight, "features.denseblock1.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_weight, "features.denseblock1.denselayer3.norm1.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_bias, "features.denseblock1.denselayer3.norm1.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_running_mean, "features.denseblock1.denselayer3.norm1.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm1_running_var, "features.denseblock1.denselayer3.norm1.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer3_conv1_weight, "features.denseblock1.denselayer3.conv1.weight", 128, 128, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_weight, "features.denseblock1.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_bias, "features.denseblock1.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_running_mean, "features.denseblock1.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer3_norm2_running_var, "features.denseblock1.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer3_conv2_weight, "features.denseblock1.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_weight, "features.denseblock1.denselayer4.norm1.weight", 160);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_bias, "features.denseblock1.denselayer4.norm1.bias", 160);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_running_mean, "features.denseblock1.denselayer4.norm1.running_mean", 160);
	dict.load(ctx.features_denseblock1_denselayer4_norm1_running_var, "features.denseblock1.denselayer4.norm1.running_var", 160);
	dict.load(ctx.features_denseblock1_denselayer4_conv1_weight, "features.denseblock1.denselayer4.conv1.weight", 128, 160, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_weight, "features.denseblock1.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_bias, "features.denseblock1.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_running_mean, "features.denseblock1.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer4_norm2_running_var, "features.denseblock1.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer4_conv2_weight, "features.denseblock1.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_weight, "features.denseblock1.denselayer5.norm1.weight", 192);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_bias, "features.denseblock1.denselayer5.norm1.bias", 192);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_running_mean, "features.denseblock1.denselayer5.norm1.running_mean", 192);
	dict.load(ctx.features_denseblock1_denselayer5_norm1_running_var, "features.denseblock1.denselayer5.norm1.running_var", 192);
	dict.load(ctx.features_denseblock1_denselayer5_conv1_weight, "features.denseblock1.denselayer5.conv1.weight", 128, 192, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_weight, "features.denseblock1.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_bias, "features.denseblock1.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_running_mean, "features.denseblock1.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer5_norm2_running_var, "features.denseblock1.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer5_conv2_weight, "features.denseblock1.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_weight, "features.denseblock1.denselayer6.norm1.weight", 224);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_bias, "features.denseblock1.denselayer6.norm1.bias", 224);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_running_mean, "features.denseblock1.denselayer6.norm1.running_mean", 224);
	dict.load(ctx.features_denseblock1_denselayer6_norm1_running_var, "features.denseblock1.denselayer6.norm1.running_var", 224);
	dict.load(ctx.features_denseblock1_denselayer6_conv1_weight, "features.denseblock1.denselayer6.conv1.weight", 128, 224, 1, 1);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_weight, "features.denseblock1.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_bias, "features.denseblock1.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_running_mean, "features.denseblock1.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock1_denselayer6_norm2_running_var, "features.denseblock1.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock1_denselayer6_conv2_weight, "features.denseblock1.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_transition1_norm_weight, "features.transition1.norm.weight", 256);
	dict.load(ctx.features_transition1_norm_bias, "features.transition1.norm.bias", 256);
	dict.load(ctx.features_transition1_norm_running_mean, "features.transition1.norm.running_mean", 256);
	dict.load(ctx.features_transition1_norm_running_var, "features.transition1.norm.running_var", 256);
	dict.load(ctx.features_transition1_conv_weight, "features.transition1.conv.weight", 128, 256, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_weight, "features.denseblock2.denselayer1.norm1.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_bias, "features.denseblock2.denselayer1.norm1.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_running_mean, "features.denseblock2.denselayer1.norm1.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm1_running_var, "features.denseblock2.denselayer1.norm1.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer1_conv1_weight, "features.denseblock2.denselayer1.conv1.weight", 128, 128, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_weight, "features.denseblock2.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_bias, "features.denseblock2.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_running_mean, "features.denseblock2.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer1_norm2_running_var, "features.denseblock2.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer1_conv2_weight, "features.denseblock2.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_weight, "features.denseblock2.denselayer2.norm1.weight", 160);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_bias, "features.denseblock2.denselayer2.norm1.bias", 160);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_running_mean, "features.denseblock2.denselayer2.norm1.running_mean", 160);
	dict.load(ctx.features_denseblock2_denselayer2_norm1_running_var, "features.denseblock2.denselayer2.norm1.running_var", 160);
	dict.load(ctx.features_denseblock2_denselayer2_conv1_weight, "features.denseblock2.denselayer2.conv1.weight", 128, 160, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_weight, "features.denseblock2.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_bias, "features.denseblock2.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_running_mean, "features.denseblock2.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer2_norm2_running_var, "features.denseblock2.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer2_conv2_weight, "features.denseblock2.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_weight, "features.denseblock2.denselayer3.norm1.weight", 192);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_bias, "features.denseblock2.denselayer3.norm1.bias", 192);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_running_mean, "features.denseblock2.denselayer3.norm1.running_mean", 192);
	dict.load(ctx.features_denseblock2_denselayer3_norm1_running_var, "features.denseblock2.denselayer3.norm1.running_var", 192);
	dict.load(ctx.features_denseblock2_denselayer3_conv1_weight, "features.denseblock2.denselayer3.conv1.weight", 128, 192, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_weight, "features.denseblock2.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_bias, "features.denseblock2.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_running_mean, "features.denseblock2.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer3_norm2_running_var, "features.denseblock2.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer3_conv2_weight, "features.denseblock2.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_weight, "features.denseblock2.denselayer4.norm1.weight", 224);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_bias, "features.denseblock2.denselayer4.norm1.bias", 224);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_running_mean, "features.denseblock2.denselayer4.norm1.running_mean", 224);
	dict.load(ctx.features_denseblock2_denselayer4_norm1_running_var, "features.denseblock2.denselayer4.norm1.running_var", 224);
	dict.load(ctx.features_denseblock2_denselayer4_conv1_weight, "features.denseblock2.denselayer4.conv1.weight", 128, 224, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_weight, "features.denseblock2.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_bias, "features.denseblock2.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_running_mean, "features.denseblock2.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer4_norm2_running_var, "features.denseblock2.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer4_conv2_weight, "features.denseblock2.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_weight, "features.denseblock2.denselayer5.norm1.weight", 256);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_bias, "features.denseblock2.denselayer5.norm1.bias", 256);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_running_mean, "features.denseblock2.denselayer5.norm1.running_mean", 256);
	dict.load(ctx.features_denseblock2_denselayer5_norm1_running_var, "features.denseblock2.denselayer5.norm1.running_var", 256);
	dict.load(ctx.features_denseblock2_denselayer5_conv1_weight, "features.denseblock2.denselayer5.conv1.weight", 128, 256, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_weight, "features.denseblock2.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_bias, "features.denseblock2.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_running_mean, "features.denseblock2.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer5_norm2_running_var, "features.denseblock2.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer5_conv2_weight, "features.denseblock2.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_weight, "features.denseblock2.denselayer6.norm1.weight", 288);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_bias, "features.denseblock2.denselayer6.norm1.bias", 288);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_running_mean, "features.denseblock2.denselayer6.norm1.running_mean", 288);
	dict.load(ctx.features_denseblock2_denselayer6_norm1_running_var, "features.denseblock2.denselayer6.norm1.running_var", 288);
	dict.load(ctx.features_denseblock2_denselayer6_conv1_weight, "features.denseblock2.denselayer6.conv1.weight", 128, 288, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_weight, "features.denseblock2.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_bias, "features.denseblock2.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_running_mean, "features.denseblock2.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer6_norm2_running_var, "features.denseblock2.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer6_conv2_weight, "features.denseblock2.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_weight, "features.denseblock2.denselayer7.norm1.weight", 320);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_bias, "features.denseblock2.denselayer7.norm1.bias", 320);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_running_mean, "features.denseblock2.denselayer7.norm1.running_mean", 320);
	dict.load(ctx.features_denseblock2_denselayer7_norm1_running_var, "features.denseblock2.denselayer7.norm1.running_var", 320);
	dict.load(ctx.features_denseblock2_denselayer7_conv1_weight, "features.denseblock2.denselayer7.conv1.weight", 128, 320, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_weight, "features.denseblock2.denselayer7.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_bias, "features.denseblock2.denselayer7.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_running_mean, "features.denseblock2.denselayer7.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer7_norm2_running_var, "features.denseblock2.denselayer7.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer7_conv2_weight, "features.denseblock2.denselayer7.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_weight, "features.denseblock2.denselayer8.norm1.weight", 352);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_bias, "features.denseblock2.denselayer8.norm1.bias", 352);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_running_mean, "features.denseblock2.denselayer8.norm1.running_mean", 352);
	dict.load(ctx.features_denseblock2_denselayer8_norm1_running_var, "features.denseblock2.denselayer8.norm1.running_var", 352);
	dict.load(ctx.features_denseblock2_denselayer8_conv1_weight, "features.denseblock2.denselayer8.conv1.weight", 128, 352, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_weight, "features.denseblock2.denselayer8.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_bias, "features.denseblock2.denselayer8.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_running_mean, "features.denseblock2.denselayer8.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer8_norm2_running_var, "features.denseblock2.denselayer8.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer8_conv2_weight, "features.denseblock2.denselayer8.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_weight, "features.denseblock2.denselayer9.norm1.weight", 384);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_bias, "features.denseblock2.denselayer9.norm1.bias", 384);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_running_mean, "features.denseblock2.denselayer9.norm1.running_mean", 384);
	dict.load(ctx.features_denseblock2_denselayer9_norm1_running_var, "features.denseblock2.denselayer9.norm1.running_var", 384);
	dict.load(ctx.features_denseblock2_denselayer9_conv1_weight, "features.denseblock2.denselayer9.conv1.weight", 128, 384, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_weight, "features.denseblock2.denselayer9.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_bias, "features.denseblock2.denselayer9.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_running_mean, "features.denseblock2.denselayer9.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer9_norm2_running_var, "features.denseblock2.denselayer9.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer9_conv2_weight, "features.denseblock2.denselayer9.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_weight, "features.denseblock2.denselayer10.norm1.weight", 416);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_bias, "features.denseblock2.denselayer10.norm1.bias", 416);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_running_mean, "features.denseblock2.denselayer10.norm1.running_mean", 416);
	dict.load(ctx.features_denseblock2_denselayer10_norm1_running_var, "features.denseblock2.denselayer10.norm1.running_var", 416);
	dict.load(ctx.features_denseblock2_denselayer10_conv1_weight, "features.denseblock2.denselayer10.conv1.weight", 128, 416, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_weight, "features.denseblock2.denselayer10.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_bias, "features.denseblock2.denselayer10.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_running_mean, "features.denseblock2.denselayer10.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer10_norm2_running_var, "features.denseblock2.denselayer10.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer10_conv2_weight, "features.denseblock2.denselayer10.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_weight, "features.denseblock2.denselayer11.norm1.weight", 448);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_bias, "features.denseblock2.denselayer11.norm1.bias", 448);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_running_mean, "features.denseblock2.denselayer11.norm1.running_mean", 448);
	dict.load(ctx.features_denseblock2_denselayer11_norm1_running_var, "features.denseblock2.denselayer11.norm1.running_var", 448);
	dict.load(ctx.features_denseblock2_denselayer11_conv1_weight, "features.denseblock2.denselayer11.conv1.weight", 128, 448, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_weight, "features.denseblock2.denselayer11.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_bias, "features.denseblock2.denselayer11.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_running_mean, "features.denseblock2.denselayer11.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer11_norm2_running_var, "features.denseblock2.denselayer11.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer11_conv2_weight, "features.denseblock2.denselayer11.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_weight, "features.denseblock2.denselayer12.norm1.weight", 480);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_bias, "features.denseblock2.denselayer12.norm1.bias", 480);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_running_mean, "features.denseblock2.denselayer12.norm1.running_mean", 480);
	dict.load(ctx.features_denseblock2_denselayer12_norm1_running_var, "features.denseblock2.denselayer12.norm1.running_var", 480);
	dict.load(ctx.features_denseblock2_denselayer12_conv1_weight, "features.denseblock2.denselayer12.conv1.weight", 128, 480, 1, 1);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_weight, "features.denseblock2.denselayer12.norm2.weight", 128);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_bias, "features.denseblock2.denselayer12.norm2.bias", 128);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_running_mean, "features.denseblock2.denselayer12.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock2_denselayer12_norm2_running_var, "features.denseblock2.denselayer12.norm2.running_var", 128);
	dict.load(ctx.features_denseblock2_denselayer12_conv2_weight, "features.denseblock2.denselayer12.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_transition2_norm_weight, "features.transition2.norm.weight", 512);
	dict.load(ctx.features_transition2_norm_bias, "features.transition2.norm.bias", 512);
	dict.load(ctx.features_transition2_norm_running_mean, "features.transition2.norm.running_mean", 512);
	dict.load(ctx.features_transition2_norm_running_var, "features.transition2.norm.running_var", 512);
	dict.load(ctx.features_transition2_conv_weight, "features.transition2.conv.weight", 256, 512, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_weight, "features.denseblock3.denselayer1.norm1.weight", 256);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_bias, "features.denseblock3.denselayer1.norm1.bias", 256);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_running_mean, "features.denseblock3.denselayer1.norm1.running_mean", 256);
	dict.load(ctx.features_denseblock3_denselayer1_norm1_running_var, "features.denseblock3.denselayer1.norm1.running_var", 256);
	dict.load(ctx.features_denseblock3_denselayer1_conv1_weight, "features.denseblock3.denselayer1.conv1.weight", 128, 256, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_weight, "features.denseblock3.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_bias, "features.denseblock3.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_running_mean, "features.denseblock3.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer1_norm2_running_var, "features.denseblock3.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer1_conv2_weight, "features.denseblock3.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_weight, "features.denseblock3.denselayer2.norm1.weight", 288);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_bias, "features.denseblock3.denselayer2.norm1.bias", 288);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_running_mean, "features.denseblock3.denselayer2.norm1.running_mean", 288);
	dict.load(ctx.features_denseblock3_denselayer2_norm1_running_var, "features.denseblock3.denselayer2.norm1.running_var", 288);
	dict.load(ctx.features_denseblock3_denselayer2_conv1_weight, "features.denseblock3.denselayer2.conv1.weight", 128, 288, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_weight, "features.denseblock3.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_bias, "features.denseblock3.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_running_mean, "features.denseblock3.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer2_norm2_running_var, "features.denseblock3.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer2_conv2_weight, "features.denseblock3.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_weight, "features.denseblock3.denselayer3.norm1.weight", 320);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_bias, "features.denseblock3.denselayer3.norm1.bias", 320);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_running_mean, "features.denseblock3.denselayer3.norm1.running_mean", 320);
	dict.load(ctx.features_denseblock3_denselayer3_norm1_running_var, "features.denseblock3.denselayer3.norm1.running_var", 320);
	dict.load(ctx.features_denseblock3_denselayer3_conv1_weight, "features.denseblock3.denselayer3.conv1.weight", 128, 320, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_weight, "features.denseblock3.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_bias, "features.denseblock3.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_running_mean, "features.denseblock3.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer3_norm2_running_var, "features.denseblock3.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer3_conv2_weight, "features.denseblock3.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_weight, "features.denseblock3.denselayer4.norm1.weight", 352);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_bias, "features.denseblock3.denselayer4.norm1.bias", 352);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_running_mean, "features.denseblock3.denselayer4.norm1.running_mean", 352);
	dict.load(ctx.features_denseblock3_denselayer4_norm1_running_var, "features.denseblock3.denselayer4.norm1.running_var", 352);
	dict.load(ctx.features_denseblock3_denselayer4_conv1_weight, "features.denseblock3.denselayer4.conv1.weight", 128, 352, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_weight, "features.denseblock3.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_bias, "features.denseblock3.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_running_mean, "features.denseblock3.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer4_norm2_running_var, "features.denseblock3.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer4_conv2_weight, "features.denseblock3.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_weight, "features.denseblock3.denselayer5.norm1.weight", 384);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_bias, "features.denseblock3.denselayer5.norm1.bias", 384);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_running_mean, "features.denseblock3.denselayer5.norm1.running_mean", 384);
	dict.load(ctx.features_denseblock3_denselayer5_norm1_running_var, "features.denseblock3.denselayer5.norm1.running_var", 384);
	dict.load(ctx.features_denseblock3_denselayer5_conv1_weight, "features.denseblock3.denselayer5.conv1.weight", 128, 384, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_weight, "features.denseblock3.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_bias, "features.denseblock3.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_running_mean, "features.denseblock3.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer5_norm2_running_var, "features.denseblock3.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer5_conv2_weight, "features.denseblock3.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_weight, "features.denseblock3.denselayer6.norm1.weight", 416);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_bias, "features.denseblock3.denselayer6.norm1.bias", 416);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_running_mean, "features.denseblock3.denselayer6.norm1.running_mean", 416);
	dict.load(ctx.features_denseblock3_denselayer6_norm1_running_var, "features.denseblock3.denselayer6.norm1.running_var", 416);
	dict.load(ctx.features_denseblock3_denselayer6_conv1_weight, "features.denseblock3.denselayer6.conv1.weight", 128, 416, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_weight, "features.denseblock3.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_bias, "features.denseblock3.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_running_mean, "features.denseblock3.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer6_norm2_running_var, "features.denseblock3.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer6_conv2_weight, "features.denseblock3.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_weight, "features.denseblock3.denselayer7.norm1.weight", 448);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_bias, "features.denseblock3.denselayer7.norm1.bias", 448);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_running_mean, "features.denseblock3.denselayer7.norm1.running_mean", 448);
	dict.load(ctx.features_denseblock3_denselayer7_norm1_running_var, "features.denseblock3.denselayer7.norm1.running_var", 448);
	dict.load(ctx.features_denseblock3_denselayer7_conv1_weight, "features.denseblock3.denselayer7.conv1.weight", 128, 448, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_weight, "features.denseblock3.denselayer7.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_bias, "features.denseblock3.denselayer7.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_running_mean, "features.denseblock3.denselayer7.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer7_norm2_running_var, "features.denseblock3.denselayer7.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer7_conv2_weight, "features.denseblock3.denselayer7.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_weight, "features.denseblock3.denselayer8.norm1.weight", 480);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_bias, "features.denseblock3.denselayer8.norm1.bias", 480);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_running_mean, "features.denseblock3.denselayer8.norm1.running_mean", 480);
	dict.load(ctx.features_denseblock3_denselayer8_norm1_running_var, "features.denseblock3.denselayer8.norm1.running_var", 480);
	dict.load(ctx.features_denseblock3_denselayer8_conv1_weight, "features.denseblock3.denselayer8.conv1.weight", 128, 480, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_weight, "features.denseblock3.denselayer8.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_bias, "features.denseblock3.denselayer8.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_running_mean, "features.denseblock3.denselayer8.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer8_norm2_running_var, "features.denseblock3.denselayer8.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer8_conv2_weight, "features.denseblock3.denselayer8.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_weight, "features.denseblock3.denselayer9.norm1.weight", 512);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_bias, "features.denseblock3.denselayer9.norm1.bias", 512);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_running_mean, "features.denseblock3.denselayer9.norm1.running_mean", 512);
	dict.load(ctx.features_denseblock3_denselayer9_norm1_running_var, "features.denseblock3.denselayer9.norm1.running_var", 512);
	dict.load(ctx.features_denseblock3_denselayer9_conv1_weight, "features.denseblock3.denselayer9.conv1.weight", 128, 512, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_weight, "features.denseblock3.denselayer9.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_bias, "features.denseblock3.denselayer9.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_running_mean, "features.denseblock3.denselayer9.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer9_norm2_running_var, "features.denseblock3.denselayer9.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer9_conv2_weight, "features.denseblock3.denselayer9.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_weight, "features.denseblock3.denselayer10.norm1.weight", 544);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_bias, "features.denseblock3.denselayer10.norm1.bias", 544);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_running_mean, "features.denseblock3.denselayer10.norm1.running_mean", 544);
	dict.load(ctx.features_denseblock3_denselayer10_norm1_running_var, "features.denseblock3.denselayer10.norm1.running_var", 544);
	dict.load(ctx.features_denseblock3_denselayer10_conv1_weight, "features.denseblock3.denselayer10.conv1.weight", 128, 544, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_weight, "features.denseblock3.denselayer10.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_bias, "features.denseblock3.denselayer10.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_running_mean, "features.denseblock3.denselayer10.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer10_norm2_running_var, "features.denseblock3.denselayer10.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer10_conv2_weight, "features.denseblock3.denselayer10.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_weight, "features.denseblock3.denselayer11.norm1.weight", 576);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_bias, "features.denseblock3.denselayer11.norm1.bias", 576);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_running_mean, "features.denseblock3.denselayer11.norm1.running_mean", 576);
	dict.load(ctx.features_denseblock3_denselayer11_norm1_running_var, "features.denseblock3.denselayer11.norm1.running_var", 576);
	dict.load(ctx.features_denseblock3_denselayer11_conv1_weight, "features.denseblock3.denselayer11.conv1.weight", 128, 576, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_weight, "features.denseblock3.denselayer11.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_bias, "features.denseblock3.denselayer11.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_running_mean, "features.denseblock3.denselayer11.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer11_norm2_running_var, "features.denseblock3.denselayer11.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer11_conv2_weight, "features.denseblock3.denselayer11.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_weight, "features.denseblock3.denselayer12.norm1.weight", 608);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_bias, "features.denseblock3.denselayer12.norm1.bias", 608);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_running_mean, "features.denseblock3.denselayer12.norm1.running_mean", 608);
	dict.load(ctx.features_denseblock3_denselayer12_norm1_running_var, "features.denseblock3.denselayer12.norm1.running_var", 608);
	dict.load(ctx.features_denseblock3_denselayer12_conv1_weight, "features.denseblock3.denselayer12.conv1.weight", 128, 608, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_weight, "features.denseblock3.denselayer12.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_bias, "features.denseblock3.denselayer12.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_running_mean, "features.denseblock3.denselayer12.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer12_norm2_running_var, "features.denseblock3.denselayer12.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer12_conv2_weight, "features.denseblock3.denselayer12.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_weight, "features.denseblock3.denselayer13.norm1.weight", 640);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_bias, "features.denseblock3.denselayer13.norm1.bias", 640);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_running_mean, "features.denseblock3.denselayer13.norm1.running_mean", 640);
	dict.load(ctx.features_denseblock3_denselayer13_norm1_running_var, "features.denseblock3.denselayer13.norm1.running_var", 640);
	dict.load(ctx.features_denseblock3_denselayer13_conv1_weight, "features.denseblock3.denselayer13.conv1.weight", 128, 640, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_weight, "features.denseblock3.denselayer13.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_bias, "features.denseblock3.denselayer13.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_running_mean, "features.denseblock3.denselayer13.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer13_norm2_running_var, "features.denseblock3.denselayer13.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer13_conv2_weight, "features.denseblock3.denselayer13.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_weight, "features.denseblock3.denselayer14.norm1.weight", 672);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_bias, "features.denseblock3.denselayer14.norm1.bias", 672);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_running_mean, "features.denseblock3.denselayer14.norm1.running_mean", 672);
	dict.load(ctx.features_denseblock3_denselayer14_norm1_running_var, "features.denseblock3.denselayer14.norm1.running_var", 672);
	dict.load(ctx.features_denseblock3_denselayer14_conv1_weight, "features.denseblock3.denselayer14.conv1.weight", 128, 672, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_weight, "features.denseblock3.denselayer14.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_bias, "features.denseblock3.denselayer14.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_running_mean, "features.denseblock3.denselayer14.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer14_norm2_running_var, "features.denseblock3.denselayer14.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer14_conv2_weight, "features.denseblock3.denselayer14.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_weight, "features.denseblock3.denselayer15.norm1.weight", 704);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_bias, "features.denseblock3.denselayer15.norm1.bias", 704);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_running_mean, "features.denseblock3.denselayer15.norm1.running_mean", 704);
	dict.load(ctx.features_denseblock3_denselayer15_norm1_running_var, "features.denseblock3.denselayer15.norm1.running_var", 704);
	dict.load(ctx.features_denseblock3_denselayer15_conv1_weight, "features.denseblock3.denselayer15.conv1.weight", 128, 704, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_weight, "features.denseblock3.denselayer15.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_bias, "features.denseblock3.denselayer15.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_running_mean, "features.denseblock3.denselayer15.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer15_norm2_running_var, "features.denseblock3.denselayer15.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer15_conv2_weight, "features.denseblock3.denselayer15.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_weight, "features.denseblock3.denselayer16.norm1.weight", 736);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_bias, "features.denseblock3.denselayer16.norm1.bias", 736);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_running_mean, "features.denseblock3.denselayer16.norm1.running_mean", 736);
	dict.load(ctx.features_denseblock3_denselayer16_norm1_running_var, "features.denseblock3.denselayer16.norm1.running_var", 736);
	dict.load(ctx.features_denseblock3_denselayer16_conv1_weight, "features.denseblock3.denselayer16.conv1.weight", 128, 736, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_weight, "features.denseblock3.denselayer16.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_bias, "features.denseblock3.denselayer16.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_running_mean, "features.denseblock3.denselayer16.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer16_norm2_running_var, "features.denseblock3.denselayer16.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer16_conv2_weight, "features.denseblock3.denselayer16.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_weight, "features.denseblock3.denselayer17.norm1.weight", 768);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_bias, "features.denseblock3.denselayer17.norm1.bias", 768);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_running_mean, "features.denseblock3.denselayer17.norm1.running_mean", 768);
	dict.load(ctx.features_denseblock3_denselayer17_norm1_running_var, "features.denseblock3.denselayer17.norm1.running_var", 768);
	dict.load(ctx.features_denseblock3_denselayer17_conv1_weight, "features.denseblock3.denselayer17.conv1.weight", 128, 768, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_weight, "features.denseblock3.denselayer17.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_bias, "features.denseblock3.denselayer17.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_running_mean, "features.denseblock3.denselayer17.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer17_norm2_running_var, "features.denseblock3.denselayer17.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer17_conv2_weight, "features.denseblock3.denselayer17.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_weight, "features.denseblock3.denselayer18.norm1.weight", 800);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_bias, "features.denseblock3.denselayer18.norm1.bias", 800);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_running_mean, "features.denseblock3.denselayer18.norm1.running_mean", 800);
	dict.load(ctx.features_denseblock3_denselayer18_norm1_running_var, "features.denseblock3.denselayer18.norm1.running_var", 800);
	dict.load(ctx.features_denseblock3_denselayer18_conv1_weight, "features.denseblock3.denselayer18.conv1.weight", 128, 800, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_weight, "features.denseblock3.denselayer18.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_bias, "features.denseblock3.denselayer18.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_running_mean, "features.denseblock3.denselayer18.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer18_norm2_running_var, "features.denseblock3.denselayer18.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer18_conv2_weight, "features.denseblock3.denselayer18.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_weight, "features.denseblock3.denselayer19.norm1.weight", 832);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_bias, "features.denseblock3.denselayer19.norm1.bias", 832);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_running_mean, "features.denseblock3.denselayer19.norm1.running_mean", 832);
	dict.load(ctx.features_denseblock3_denselayer19_norm1_running_var, "features.denseblock3.denselayer19.norm1.running_var", 832);
	dict.load(ctx.features_denseblock3_denselayer19_conv1_weight, "features.denseblock3.denselayer19.conv1.weight", 128, 832, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_weight, "features.denseblock3.denselayer19.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_bias, "features.denseblock3.denselayer19.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_running_mean, "features.denseblock3.denselayer19.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer19_norm2_running_var, "features.denseblock3.denselayer19.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer19_conv2_weight, "features.denseblock3.denselayer19.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_weight, "features.denseblock3.denselayer20.norm1.weight", 864);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_bias, "features.denseblock3.denselayer20.norm1.bias", 864);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_running_mean, "features.denseblock3.denselayer20.norm1.running_mean", 864);
	dict.load(ctx.features_denseblock3_denselayer20_norm1_running_var, "features.denseblock3.denselayer20.norm1.running_var", 864);
	dict.load(ctx.features_denseblock3_denselayer20_conv1_weight, "features.denseblock3.denselayer20.conv1.weight", 128, 864, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_weight, "features.denseblock3.denselayer20.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_bias, "features.denseblock3.denselayer20.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_running_mean, "features.denseblock3.denselayer20.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer20_norm2_running_var, "features.denseblock3.denselayer20.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer20_conv2_weight, "features.denseblock3.denselayer20.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_weight, "features.denseblock3.denselayer21.norm1.weight", 896);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_bias, "features.denseblock3.denselayer21.norm1.bias", 896);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_running_mean, "features.denseblock3.denselayer21.norm1.running_mean", 896);
	dict.load(ctx.features_denseblock3_denselayer21_norm1_running_var, "features.denseblock3.denselayer21.norm1.running_var", 896);
	dict.load(ctx.features_denseblock3_denselayer21_conv1_weight, "features.denseblock3.denselayer21.conv1.weight", 128, 896, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_weight, "features.denseblock3.denselayer21.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_bias, "features.denseblock3.denselayer21.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_running_mean, "features.denseblock3.denselayer21.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer21_norm2_running_var, "features.denseblock3.denselayer21.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer21_conv2_weight, "features.denseblock3.denselayer21.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_weight, "features.denseblock3.denselayer22.norm1.weight", 928);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_bias, "features.denseblock3.denselayer22.norm1.bias", 928);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_running_mean, "features.denseblock3.denselayer22.norm1.running_mean", 928);
	dict.load(ctx.features_denseblock3_denselayer22_norm1_running_var, "features.denseblock3.denselayer22.norm1.running_var", 928);
	dict.load(ctx.features_denseblock3_denselayer22_conv1_weight, "features.denseblock3.denselayer22.conv1.weight", 128, 928, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_weight, "features.denseblock3.denselayer22.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_bias, "features.denseblock3.denselayer22.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_running_mean, "features.denseblock3.denselayer22.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer22_norm2_running_var, "features.denseblock3.denselayer22.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer22_conv2_weight, "features.denseblock3.denselayer22.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_weight, "features.denseblock3.denselayer23.norm1.weight", 960);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_bias, "features.denseblock3.denselayer23.norm1.bias", 960);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_running_mean, "features.denseblock3.denselayer23.norm1.running_mean", 960);
	dict.load(ctx.features_denseblock3_denselayer23_norm1_running_var, "features.denseblock3.denselayer23.norm1.running_var", 960);
	dict.load(ctx.features_denseblock3_denselayer23_conv1_weight, "features.denseblock3.denselayer23.conv1.weight", 128, 960, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_weight, "features.denseblock3.denselayer23.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_bias, "features.denseblock3.denselayer23.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_running_mean, "features.denseblock3.denselayer23.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer23_norm2_running_var, "features.denseblock3.denselayer23.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer23_conv2_weight, "features.denseblock3.denselayer23.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_weight, "features.denseblock3.denselayer24.norm1.weight", 992);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_bias, "features.denseblock3.denselayer24.norm1.bias", 992);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_running_mean, "features.denseblock3.denselayer24.norm1.running_mean", 992);
	dict.load(ctx.features_denseblock3_denselayer24_norm1_running_var, "features.denseblock3.denselayer24.norm1.running_var", 992);
	dict.load(ctx.features_denseblock3_denselayer24_conv1_weight, "features.denseblock3.denselayer24.conv1.weight", 128, 992, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_weight, "features.denseblock3.denselayer24.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_bias, "features.denseblock3.denselayer24.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_running_mean, "features.denseblock3.denselayer24.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer24_norm2_running_var, "features.denseblock3.denselayer24.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer24_conv2_weight, "features.denseblock3.denselayer24.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_weight, "features.denseblock3.denselayer25.norm1.weight", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_bias, "features.denseblock3.denselayer25.norm1.bias", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_running_mean, "features.denseblock3.denselayer25.norm1.running_mean", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_norm1_running_var, "features.denseblock3.denselayer25.norm1.running_var", 1024);
	dict.load(ctx.features_denseblock3_denselayer25_conv1_weight, "features.denseblock3.denselayer25.conv1.weight", 128, 1024, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_weight, "features.denseblock3.denselayer25.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_bias, "features.denseblock3.denselayer25.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_running_mean, "features.denseblock3.denselayer25.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer25_norm2_running_var, "features.denseblock3.denselayer25.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer25_conv2_weight, "features.denseblock3.denselayer25.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_weight, "features.denseblock3.denselayer26.norm1.weight", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_bias, "features.denseblock3.denselayer26.norm1.bias", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_running_mean, "features.denseblock3.denselayer26.norm1.running_mean", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_norm1_running_var, "features.denseblock3.denselayer26.norm1.running_var", 1056);
	dict.load(ctx.features_denseblock3_denselayer26_conv1_weight, "features.denseblock3.denselayer26.conv1.weight", 128, 1056, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_weight, "features.denseblock3.denselayer26.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_bias, "features.denseblock3.denselayer26.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_running_mean, "features.denseblock3.denselayer26.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer26_norm2_running_var, "features.denseblock3.denselayer26.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer26_conv2_weight, "features.denseblock3.denselayer26.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_weight, "features.denseblock3.denselayer27.norm1.weight", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_bias, "features.denseblock3.denselayer27.norm1.bias", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_running_mean, "features.denseblock3.denselayer27.norm1.running_mean", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_norm1_running_var, "features.denseblock3.denselayer27.norm1.running_var", 1088);
	dict.load(ctx.features_denseblock3_denselayer27_conv1_weight, "features.denseblock3.denselayer27.conv1.weight", 128, 1088, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_weight, "features.denseblock3.denselayer27.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_bias, "features.denseblock3.denselayer27.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_running_mean, "features.denseblock3.denselayer27.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer27_norm2_running_var, "features.denseblock3.denselayer27.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer27_conv2_weight, "features.denseblock3.denselayer27.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_weight, "features.denseblock3.denselayer28.norm1.weight", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_bias, "features.denseblock3.denselayer28.norm1.bias", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_running_mean, "features.denseblock3.denselayer28.norm1.running_mean", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_norm1_running_var, "features.denseblock3.denselayer28.norm1.running_var", 1120);
	dict.load(ctx.features_denseblock3_denselayer28_conv1_weight, "features.denseblock3.denselayer28.conv1.weight", 128, 1120, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_weight, "features.denseblock3.denselayer28.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_bias, "features.denseblock3.denselayer28.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_running_mean, "features.denseblock3.denselayer28.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer28_norm2_running_var, "features.denseblock3.denselayer28.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer28_conv2_weight, "features.denseblock3.denselayer28.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_weight, "features.denseblock3.denselayer29.norm1.weight", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_bias, "features.denseblock3.denselayer29.norm1.bias", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_running_mean, "features.denseblock3.denselayer29.norm1.running_mean", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_norm1_running_var, "features.denseblock3.denselayer29.norm1.running_var", 1152);
	dict.load(ctx.features_denseblock3_denselayer29_conv1_weight, "features.denseblock3.denselayer29.conv1.weight", 128, 1152, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_weight, "features.denseblock3.denselayer29.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_bias, "features.denseblock3.denselayer29.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_running_mean, "features.denseblock3.denselayer29.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer29_norm2_running_var, "features.denseblock3.denselayer29.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer29_conv2_weight, "features.denseblock3.denselayer29.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_weight, "features.denseblock3.denselayer30.norm1.weight", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_bias, "features.denseblock3.denselayer30.norm1.bias", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_running_mean, "features.denseblock3.denselayer30.norm1.running_mean", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_norm1_running_var, "features.denseblock3.denselayer30.norm1.running_var", 1184);
	dict.load(ctx.features_denseblock3_denselayer30_conv1_weight, "features.denseblock3.denselayer30.conv1.weight", 128, 1184, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_weight, "features.denseblock3.denselayer30.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_bias, "features.denseblock3.denselayer30.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_running_mean, "features.denseblock3.denselayer30.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer30_norm2_running_var, "features.denseblock3.denselayer30.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer30_conv2_weight, "features.denseblock3.denselayer30.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_weight, "features.denseblock3.denselayer31.norm1.weight", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_bias, "features.denseblock3.denselayer31.norm1.bias", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_running_mean, "features.denseblock3.denselayer31.norm1.running_mean", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_norm1_running_var, "features.denseblock3.denselayer31.norm1.running_var", 1216);
	dict.load(ctx.features_denseblock3_denselayer31_conv1_weight, "features.denseblock3.denselayer31.conv1.weight", 128, 1216, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_weight, "features.denseblock3.denselayer31.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_bias, "features.denseblock3.denselayer31.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_running_mean, "features.denseblock3.denselayer31.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer31_norm2_running_var, "features.denseblock3.denselayer31.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer31_conv2_weight, "features.denseblock3.denselayer31.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_weight, "features.denseblock3.denselayer32.norm1.weight", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_bias, "features.denseblock3.denselayer32.norm1.bias", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_running_mean, "features.denseblock3.denselayer32.norm1.running_mean", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_norm1_running_var, "features.denseblock3.denselayer32.norm1.running_var", 1248);
	dict.load(ctx.features_denseblock3_denselayer32_conv1_weight, "features.denseblock3.denselayer32.conv1.weight", 128, 1248, 1, 1);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_weight, "features.denseblock3.denselayer32.norm2.weight", 128);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_bias, "features.denseblock3.denselayer32.norm2.bias", 128);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_running_mean, "features.denseblock3.denselayer32.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock3_denselayer32_norm2_running_var, "features.denseblock3.denselayer32.norm2.running_var", 128);
	dict.load(ctx.features_denseblock3_denselayer32_conv2_weight, "features.denseblock3.denselayer32.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_transition3_norm_weight, "features.transition3.norm.weight", 1280);
	dict.load(ctx.features_transition3_norm_bias, "features.transition3.norm.bias", 1280);
	dict.load(ctx.features_transition3_norm_running_mean, "features.transition3.norm.running_mean", 1280);
	dict.load(ctx.features_transition3_norm_running_var, "features.transition3.norm.running_var", 1280);
	dict.load(ctx.features_transition3_conv_weight, "features.transition3.conv.weight", 640, 1280, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_weight, "features.denseblock4.denselayer1.norm1.weight", 640);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_bias, "features.denseblock4.denselayer1.norm1.bias", 640);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_running_mean, "features.denseblock4.denselayer1.norm1.running_mean", 640);
	dict.load(ctx.features_denseblock4_denselayer1_norm1_running_var, "features.denseblock4.denselayer1.norm1.running_var", 640);
	dict.load(ctx.features_denseblock4_denselayer1_conv1_weight, "features.denseblock4.denselayer1.conv1.weight", 128, 640, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_weight, "features.denseblock4.denselayer1.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_bias, "features.denseblock4.denselayer1.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_running_mean, "features.denseblock4.denselayer1.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer1_norm2_running_var, "features.denseblock4.denselayer1.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer1_conv2_weight, "features.denseblock4.denselayer1.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_weight, "features.denseblock4.denselayer2.norm1.weight", 672);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_bias, "features.denseblock4.denselayer2.norm1.bias", 672);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_running_mean, "features.denseblock4.denselayer2.norm1.running_mean", 672);
	dict.load(ctx.features_denseblock4_denselayer2_norm1_running_var, "features.denseblock4.denselayer2.norm1.running_var", 672);
	dict.load(ctx.features_denseblock4_denselayer2_conv1_weight, "features.denseblock4.denselayer2.conv1.weight", 128, 672, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_weight, "features.denseblock4.denselayer2.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_bias, "features.denseblock4.denselayer2.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_running_mean, "features.denseblock4.denselayer2.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer2_norm2_running_var, "features.denseblock4.denselayer2.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer2_conv2_weight, "features.denseblock4.denselayer2.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_weight, "features.denseblock4.denselayer3.norm1.weight", 704);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_bias, "features.denseblock4.denselayer3.norm1.bias", 704);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_running_mean, "features.denseblock4.denselayer3.norm1.running_mean", 704);
	dict.load(ctx.features_denseblock4_denselayer3_norm1_running_var, "features.denseblock4.denselayer3.norm1.running_var", 704);
	dict.load(ctx.features_denseblock4_denselayer3_conv1_weight, "features.denseblock4.denselayer3.conv1.weight", 128, 704, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_weight, "features.denseblock4.denselayer3.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_bias, "features.denseblock4.denselayer3.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_running_mean, "features.denseblock4.denselayer3.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer3_norm2_running_var, "features.denseblock4.denselayer3.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer3_conv2_weight, "features.denseblock4.denselayer3.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_weight, "features.denseblock4.denselayer4.norm1.weight", 736);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_bias, "features.denseblock4.denselayer4.norm1.bias", 736);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_running_mean, "features.denseblock4.denselayer4.norm1.running_mean", 736);
	dict.load(ctx.features_denseblock4_denselayer4_norm1_running_var, "features.denseblock4.denselayer4.norm1.running_var", 736);
	dict.load(ctx.features_denseblock4_denselayer4_conv1_weight, "features.denseblock4.denselayer4.conv1.weight", 128, 736, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_weight, "features.denseblock4.denselayer4.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_bias, "features.denseblock4.denselayer4.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_running_mean, "features.denseblock4.denselayer4.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer4_norm2_running_var, "features.denseblock4.denselayer4.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer4_conv2_weight, "features.denseblock4.denselayer4.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_weight, "features.denseblock4.denselayer5.norm1.weight", 768);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_bias, "features.denseblock4.denselayer5.norm1.bias", 768);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_running_mean, "features.denseblock4.denselayer5.norm1.running_mean", 768);
	dict.load(ctx.features_denseblock4_denselayer5_norm1_running_var, "features.denseblock4.denselayer5.norm1.running_var", 768);
	dict.load(ctx.features_denseblock4_denselayer5_conv1_weight, "features.denseblock4.denselayer5.conv1.weight", 128, 768, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_weight, "features.denseblock4.denselayer5.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_bias, "features.denseblock4.denselayer5.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_running_mean, "features.denseblock4.denselayer5.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer5_norm2_running_var, "features.denseblock4.denselayer5.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer5_conv2_weight, "features.denseblock4.denselayer5.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_weight, "features.denseblock4.denselayer6.norm1.weight", 800);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_bias, "features.denseblock4.denselayer6.norm1.bias", 800);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_running_mean, "features.denseblock4.denselayer6.norm1.running_mean", 800);
	dict.load(ctx.features_denseblock4_denselayer6_norm1_running_var, "features.denseblock4.denselayer6.norm1.running_var", 800);
	dict.load(ctx.features_denseblock4_denselayer6_conv1_weight, "features.denseblock4.denselayer6.conv1.weight", 128, 800, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_weight, "features.denseblock4.denselayer6.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_bias, "features.denseblock4.denselayer6.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_running_mean, "features.denseblock4.denselayer6.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer6_norm2_running_var, "features.denseblock4.denselayer6.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer6_conv2_weight, "features.denseblock4.denselayer6.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_weight, "features.denseblock4.denselayer7.norm1.weight", 832);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_bias, "features.denseblock4.denselayer7.norm1.bias", 832);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_running_mean, "features.denseblock4.denselayer7.norm1.running_mean", 832);
	dict.load(ctx.features_denseblock4_denselayer7_norm1_running_var, "features.denseblock4.denselayer7.norm1.running_var", 832);
	dict.load(ctx.features_denseblock4_denselayer7_conv1_weight, "features.denseblock4.denselayer7.conv1.weight", 128, 832, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_weight, "features.denseblock4.denselayer7.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_bias, "features.denseblock4.denselayer7.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_running_mean, "features.denseblock4.denselayer7.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer7_norm2_running_var, "features.denseblock4.denselayer7.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer7_conv2_weight, "features.denseblock4.denselayer7.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_weight, "features.denseblock4.denselayer8.norm1.weight", 864);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_bias, "features.denseblock4.denselayer8.norm1.bias", 864);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_running_mean, "features.denseblock4.denselayer8.norm1.running_mean", 864);
	dict.load(ctx.features_denseblock4_denselayer8_norm1_running_var, "features.denseblock4.denselayer8.norm1.running_var", 864);
	dict.load(ctx.features_denseblock4_denselayer8_conv1_weight, "features.denseblock4.denselayer8.conv1.weight", 128, 864, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_weight, "features.denseblock4.denselayer8.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_bias, "features.denseblock4.denselayer8.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_running_mean, "features.denseblock4.denselayer8.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer8_norm2_running_var, "features.denseblock4.denselayer8.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer8_conv2_weight, "features.denseblock4.denselayer8.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_weight, "features.denseblock4.denselayer9.norm1.weight", 896);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_bias, "features.denseblock4.denselayer9.norm1.bias", 896);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_running_mean, "features.denseblock4.denselayer9.norm1.running_mean", 896);
	dict.load(ctx.features_denseblock4_denselayer9_norm1_running_var, "features.denseblock4.denselayer9.norm1.running_var", 896);
	dict.load(ctx.features_denseblock4_denselayer9_conv1_weight, "features.denseblock4.denselayer9.conv1.weight", 128, 896, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_weight, "features.denseblock4.denselayer9.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_bias, "features.denseblock4.denselayer9.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_running_mean, "features.denseblock4.denselayer9.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer9_norm2_running_var, "features.denseblock4.denselayer9.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer9_conv2_weight, "features.denseblock4.denselayer9.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_weight, "features.denseblock4.denselayer10.norm1.weight", 928);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_bias, "features.denseblock4.denselayer10.norm1.bias", 928);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_running_mean, "features.denseblock4.denselayer10.norm1.running_mean", 928);
	dict.load(ctx.features_denseblock4_denselayer10_norm1_running_var, "features.denseblock4.denselayer10.norm1.running_var", 928);
	dict.load(ctx.features_denseblock4_denselayer10_conv1_weight, "features.denseblock4.denselayer10.conv1.weight", 128, 928, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_weight, "features.denseblock4.denselayer10.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_bias, "features.denseblock4.denselayer10.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_running_mean, "features.denseblock4.denselayer10.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer10_norm2_running_var, "features.denseblock4.denselayer10.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer10_conv2_weight, "features.denseblock4.denselayer10.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_weight, "features.denseblock4.denselayer11.norm1.weight", 960);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_bias, "features.denseblock4.denselayer11.norm1.bias", 960);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_running_mean, "features.denseblock4.denselayer11.norm1.running_mean", 960);
	dict.load(ctx.features_denseblock4_denselayer11_norm1_running_var, "features.denseblock4.denselayer11.norm1.running_var", 960);
	dict.load(ctx.features_denseblock4_denselayer11_conv1_weight, "features.denseblock4.denselayer11.conv1.weight", 128, 960, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_weight, "features.denseblock4.denselayer11.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_bias, "features.denseblock4.denselayer11.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_running_mean, "features.denseblock4.denselayer11.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer11_norm2_running_var, "features.denseblock4.denselayer11.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer11_conv2_weight, "features.denseblock4.denselayer11.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_weight, "features.denseblock4.denselayer12.norm1.weight", 992);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_bias, "features.denseblock4.denselayer12.norm1.bias", 992);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_running_mean, "features.denseblock4.denselayer12.norm1.running_mean", 992);
	dict.load(ctx.features_denseblock4_denselayer12_norm1_running_var, "features.denseblock4.denselayer12.norm1.running_var", 992);
	dict.load(ctx.features_denseblock4_denselayer12_conv1_weight, "features.denseblock4.denselayer12.conv1.weight", 128, 992, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_weight, "features.denseblock4.denselayer12.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_bias, "features.denseblock4.denselayer12.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_running_mean, "features.denseblock4.denselayer12.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer12_norm2_running_var, "features.denseblock4.denselayer12.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer12_conv2_weight, "features.denseblock4.denselayer12.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_weight, "features.denseblock4.denselayer13.norm1.weight", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_bias, "features.denseblock4.denselayer13.norm1.bias", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_running_mean, "features.denseblock4.denselayer13.norm1.running_mean", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_norm1_running_var, "features.denseblock4.denselayer13.norm1.running_var", 1024);
	dict.load(ctx.features_denseblock4_denselayer13_conv1_weight, "features.denseblock4.denselayer13.conv1.weight", 128, 1024, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_weight, "features.denseblock4.denselayer13.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_bias, "features.denseblock4.denselayer13.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_running_mean, "features.denseblock4.denselayer13.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer13_norm2_running_var, "features.denseblock4.denselayer13.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer13_conv2_weight, "features.denseblock4.denselayer13.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_weight, "features.denseblock4.denselayer14.norm1.weight", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_bias, "features.denseblock4.denselayer14.norm1.bias", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_running_mean, "features.denseblock4.denselayer14.norm1.running_mean", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_norm1_running_var, "features.denseblock4.denselayer14.norm1.running_var", 1056);
	dict.load(ctx.features_denseblock4_denselayer14_conv1_weight, "features.denseblock4.denselayer14.conv1.weight", 128, 1056, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_weight, "features.denseblock4.denselayer14.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_bias, "features.denseblock4.denselayer14.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_running_mean, "features.denseblock4.denselayer14.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer14_norm2_running_var, "features.denseblock4.denselayer14.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer14_conv2_weight, "features.denseblock4.denselayer14.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_weight, "features.denseblock4.denselayer15.norm1.weight", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_bias, "features.denseblock4.denselayer15.norm1.bias", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_running_mean, "features.denseblock4.denselayer15.norm1.running_mean", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_norm1_running_var, "features.denseblock4.denselayer15.norm1.running_var", 1088);
	dict.load(ctx.features_denseblock4_denselayer15_conv1_weight, "features.denseblock4.denselayer15.conv1.weight", 128, 1088, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_weight, "features.denseblock4.denselayer15.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_bias, "features.denseblock4.denselayer15.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_running_mean, "features.denseblock4.denselayer15.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer15_norm2_running_var, "features.denseblock4.denselayer15.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer15_conv2_weight, "features.denseblock4.denselayer15.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_weight, "features.denseblock4.denselayer16.norm1.weight", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_bias, "features.denseblock4.denselayer16.norm1.bias", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_running_mean, "features.denseblock4.denselayer16.norm1.running_mean", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_norm1_running_var, "features.denseblock4.denselayer16.norm1.running_var", 1120);
	dict.load(ctx.features_denseblock4_denselayer16_conv1_weight, "features.denseblock4.denselayer16.conv1.weight", 128, 1120, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_weight, "features.denseblock4.denselayer16.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_bias, "features.denseblock4.denselayer16.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_running_mean, "features.denseblock4.denselayer16.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer16_norm2_running_var, "features.denseblock4.denselayer16.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer16_conv2_weight, "features.denseblock4.denselayer16.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_weight, "features.denseblock4.denselayer17.norm1.weight", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_bias, "features.denseblock4.denselayer17.norm1.bias", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_running_mean, "features.denseblock4.denselayer17.norm1.running_mean", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_norm1_running_var, "features.denseblock4.denselayer17.norm1.running_var", 1152);
	dict.load(ctx.features_denseblock4_denselayer17_conv1_weight, "features.denseblock4.denselayer17.conv1.weight", 128, 1152, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_weight, "features.denseblock4.denselayer17.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_bias, "features.denseblock4.denselayer17.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_running_mean, "features.denseblock4.denselayer17.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer17_norm2_running_var, "features.denseblock4.denselayer17.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer17_conv2_weight, "features.denseblock4.denselayer17.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_weight, "features.denseblock4.denselayer18.norm1.weight", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_bias, "features.denseblock4.denselayer18.norm1.bias", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_running_mean, "features.denseblock4.denselayer18.norm1.running_mean", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_norm1_running_var, "features.denseblock4.denselayer18.norm1.running_var", 1184);
	dict.load(ctx.features_denseblock4_denselayer18_conv1_weight, "features.denseblock4.denselayer18.conv1.weight", 128, 1184, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_weight, "features.denseblock4.denselayer18.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_bias, "features.denseblock4.denselayer18.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_running_mean, "features.denseblock4.denselayer18.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer18_norm2_running_var, "features.denseblock4.denselayer18.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer18_conv2_weight, "features.denseblock4.denselayer18.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_weight, "features.denseblock4.denselayer19.norm1.weight", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_bias, "features.denseblock4.denselayer19.norm1.bias", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_running_mean, "features.denseblock4.denselayer19.norm1.running_mean", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_norm1_running_var, "features.denseblock4.denselayer19.norm1.running_var", 1216);
	dict.load(ctx.features_denseblock4_denselayer19_conv1_weight, "features.denseblock4.denselayer19.conv1.weight", 128, 1216, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_weight, "features.denseblock4.denselayer19.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_bias, "features.denseblock4.denselayer19.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_running_mean, "features.denseblock4.denselayer19.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer19_norm2_running_var, "features.denseblock4.denselayer19.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer19_conv2_weight, "features.denseblock4.denselayer19.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_weight, "features.denseblock4.denselayer20.norm1.weight", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_bias, "features.denseblock4.denselayer20.norm1.bias", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_running_mean, "features.denseblock4.denselayer20.norm1.running_mean", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_norm1_running_var, "features.denseblock4.denselayer20.norm1.running_var", 1248);
	dict.load(ctx.features_denseblock4_denselayer20_conv1_weight, "features.denseblock4.denselayer20.conv1.weight", 128, 1248, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_weight, "features.denseblock4.denselayer20.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_bias, "features.denseblock4.denselayer20.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_running_mean, "features.denseblock4.denselayer20.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer20_norm2_running_var, "features.denseblock4.denselayer20.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer20_conv2_weight, "features.denseblock4.denselayer20.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_weight, "features.denseblock4.denselayer21.norm1.weight", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_bias, "features.denseblock4.denselayer21.norm1.bias", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_running_mean, "features.denseblock4.denselayer21.norm1.running_mean", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_norm1_running_var, "features.denseblock4.denselayer21.norm1.running_var", 1280);
	dict.load(ctx.features_denseblock4_denselayer21_conv1_weight, "features.denseblock4.denselayer21.conv1.weight", 128, 1280, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_weight, "features.denseblock4.denselayer21.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_bias, "features.denseblock4.denselayer21.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_running_mean, "features.denseblock4.denselayer21.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer21_norm2_running_var, "features.denseblock4.denselayer21.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer21_conv2_weight, "features.denseblock4.denselayer21.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_weight, "features.denseblock4.denselayer22.norm1.weight", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_bias, "features.denseblock4.denselayer22.norm1.bias", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_running_mean, "features.denseblock4.denselayer22.norm1.running_mean", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_norm1_running_var, "features.denseblock4.denselayer22.norm1.running_var", 1312);
	dict.load(ctx.features_denseblock4_denselayer22_conv1_weight, "features.denseblock4.denselayer22.conv1.weight", 128, 1312, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_weight, "features.denseblock4.denselayer22.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_bias, "features.denseblock4.denselayer22.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_running_mean, "features.denseblock4.denselayer22.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer22_norm2_running_var, "features.denseblock4.denselayer22.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer22_conv2_weight, "features.denseblock4.denselayer22.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_weight, "features.denseblock4.denselayer23.norm1.weight", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_bias, "features.denseblock4.denselayer23.norm1.bias", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_running_mean, "features.denseblock4.denselayer23.norm1.running_mean", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_norm1_running_var, "features.denseblock4.denselayer23.norm1.running_var", 1344);
	dict.load(ctx.features_denseblock4_denselayer23_conv1_weight, "features.denseblock4.denselayer23.conv1.weight", 128, 1344, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_weight, "features.denseblock4.denselayer23.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_bias, "features.denseblock4.denselayer23.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_running_mean, "features.denseblock4.denselayer23.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer23_norm2_running_var, "features.denseblock4.denselayer23.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer23_conv2_weight, "features.denseblock4.denselayer23.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_weight, "features.denseblock4.denselayer24.norm1.weight", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_bias, "features.denseblock4.denselayer24.norm1.bias", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_running_mean, "features.denseblock4.denselayer24.norm1.running_mean", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_norm1_running_var, "features.denseblock4.denselayer24.norm1.running_var", 1376);
	dict.load(ctx.features_denseblock4_denselayer24_conv1_weight, "features.denseblock4.denselayer24.conv1.weight", 128, 1376, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_weight, "features.denseblock4.denselayer24.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_bias, "features.denseblock4.denselayer24.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_running_mean, "features.denseblock4.denselayer24.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer24_norm2_running_var, "features.denseblock4.denselayer24.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer24_conv2_weight, "features.denseblock4.denselayer24.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_weight, "features.denseblock4.denselayer25.norm1.weight", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_bias, "features.denseblock4.denselayer25.norm1.bias", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_running_mean, "features.denseblock4.denselayer25.norm1.running_mean", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_norm1_running_var, "features.denseblock4.denselayer25.norm1.running_var", 1408);
	dict.load(ctx.features_denseblock4_denselayer25_conv1_weight, "features.denseblock4.denselayer25.conv1.weight", 128, 1408, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_weight, "features.denseblock4.denselayer25.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_bias, "features.denseblock4.denselayer25.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_running_mean, "features.denseblock4.denselayer25.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer25_norm2_running_var, "features.denseblock4.denselayer25.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer25_conv2_weight, "features.denseblock4.denselayer25.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_weight, "features.denseblock4.denselayer26.norm1.weight", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_bias, "features.denseblock4.denselayer26.norm1.bias", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_running_mean, "features.denseblock4.denselayer26.norm1.running_mean", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_norm1_running_var, "features.denseblock4.denselayer26.norm1.running_var", 1440);
	dict.load(ctx.features_denseblock4_denselayer26_conv1_weight, "features.denseblock4.denselayer26.conv1.weight", 128, 1440, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_weight, "features.denseblock4.denselayer26.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_bias, "features.denseblock4.denselayer26.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_running_mean, "features.denseblock4.denselayer26.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer26_norm2_running_var, "features.denseblock4.denselayer26.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer26_conv2_weight, "features.denseblock4.denselayer26.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_weight, "features.denseblock4.denselayer27.norm1.weight", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_bias, "features.denseblock4.denselayer27.norm1.bias", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_running_mean, "features.denseblock4.denselayer27.norm1.running_mean", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_norm1_running_var, "features.denseblock4.denselayer27.norm1.running_var", 1472);
	dict.load(ctx.features_denseblock4_denselayer27_conv1_weight, "features.denseblock4.denselayer27.conv1.weight", 128, 1472, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_weight, "features.denseblock4.denselayer27.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_bias, "features.denseblock4.denselayer27.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_running_mean, "features.denseblock4.denselayer27.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer27_norm2_running_var, "features.denseblock4.denselayer27.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer27_conv2_weight, "features.denseblock4.denselayer27.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_weight, "features.denseblock4.denselayer28.norm1.weight", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_bias, "features.denseblock4.denselayer28.norm1.bias", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_running_mean, "features.denseblock4.denselayer28.norm1.running_mean", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_norm1_running_var, "features.denseblock4.denselayer28.norm1.running_var", 1504);
	dict.load(ctx.features_denseblock4_denselayer28_conv1_weight, "features.denseblock4.denselayer28.conv1.weight", 128, 1504, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_weight, "features.denseblock4.denselayer28.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_bias, "features.denseblock4.denselayer28.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_running_mean, "features.denseblock4.denselayer28.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer28_norm2_running_var, "features.denseblock4.denselayer28.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer28_conv2_weight, "features.denseblock4.denselayer28.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_weight, "features.denseblock4.denselayer29.norm1.weight", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_bias, "features.denseblock4.denselayer29.norm1.bias", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_running_mean, "features.denseblock4.denselayer29.norm1.running_mean", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_norm1_running_var, "features.denseblock4.denselayer29.norm1.running_var", 1536);
	dict.load(ctx.features_denseblock4_denselayer29_conv1_weight, "features.denseblock4.denselayer29.conv1.weight", 128, 1536, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_weight, "features.denseblock4.denselayer29.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_bias, "features.denseblock4.denselayer29.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_running_mean, "features.denseblock4.denselayer29.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer29_norm2_running_var, "features.denseblock4.denselayer29.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer29_conv2_weight, "features.denseblock4.denselayer29.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_weight, "features.denseblock4.denselayer30.norm1.weight", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_bias, "features.denseblock4.denselayer30.norm1.bias", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_running_mean, "features.denseblock4.denselayer30.norm1.running_mean", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_norm1_running_var, "features.denseblock4.denselayer30.norm1.running_var", 1568);
	dict.load(ctx.features_denseblock4_denselayer30_conv1_weight, "features.denseblock4.denselayer30.conv1.weight", 128, 1568, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_weight, "features.denseblock4.denselayer30.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_bias, "features.denseblock4.denselayer30.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_running_mean, "features.denseblock4.denselayer30.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer30_norm2_running_var, "features.denseblock4.denselayer30.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer30_conv2_weight, "features.denseblock4.denselayer30.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_weight, "features.denseblock4.denselayer31.norm1.weight", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_bias, "features.denseblock4.denselayer31.norm1.bias", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_running_mean, "features.denseblock4.denselayer31.norm1.running_mean", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_norm1_running_var, "features.denseblock4.denselayer31.norm1.running_var", 1600);
	dict.load(ctx.features_denseblock4_denselayer31_conv1_weight, "features.denseblock4.denselayer31.conv1.weight", 128, 1600, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_weight, "features.denseblock4.denselayer31.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_bias, "features.denseblock4.denselayer31.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_running_mean, "features.denseblock4.denselayer31.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer31_norm2_running_var, "features.denseblock4.denselayer31.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer31_conv2_weight, "features.denseblock4.denselayer31.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_weight, "features.denseblock4.denselayer32.norm1.weight", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_bias, "features.denseblock4.denselayer32.norm1.bias", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_running_mean, "features.denseblock4.denselayer32.norm1.running_mean", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_norm1_running_var, "features.denseblock4.denselayer32.norm1.running_var", 1632);
	dict.load(ctx.features_denseblock4_denselayer32_conv1_weight, "features.denseblock4.denselayer32.conv1.weight", 128, 1632, 1, 1);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_weight, "features.denseblock4.denselayer32.norm2.weight", 128);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_bias, "features.denseblock4.denselayer32.norm2.bias", 128);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_running_mean, "features.denseblock4.denselayer32.norm2.running_mean", 128);
	dict.load(ctx.features_denseblock4_denselayer32_norm2_running_var, "features.denseblock4.denselayer32.norm2.running_var", 128);
	dict.load(ctx.features_denseblock4_denselayer32_conv2_weight, "features.denseblock4.denselayer32.conv2.weight", 32, 128, 3, 3);
	dict.load(ctx.features_norm5_weight, "features.norm5.weight", 1664);
	dict.load(ctx.features_norm5_bias, "features.norm5.bias", 1664);
	dict.load(ctx.features_norm5_running_mean, "features.norm5.running_mean", 1664);
	dict.load(ctx.features_norm5_running_var, "features.norm5.running_var", 1664);
	dict.load(ctx.classifier_weight, "classifier.weight", 1000, 1664);
	dict.load(ctx.classifier_bias, "classifier.bias", 1000);
	return ctx;
}


t4::tensor2f DenseNetForward(const DenseNet& ctx, t4::tensor4f xinput_1)
{
	t4::tensor4f x1016 = t4::Conv2d<7, 7, 2, 2, 3, 3, 1, 1>(xinput_1, ctx.features_conv0_weight); //features.conv0
	t4::release(xinput_1);
	t4::tensor4f x1017 = t4::BatchNormalizationInplace(x1016, ctx.features_norm0_weight, ctx.features_norm0_bias, ctx.features_norm0_running_mean, ctx.features_norm0_running_var, 1e-05f); //features.norm0
	t4::release(x1016);
	t4::tensor4f x1018 = t4::ReluInplace(x1017); //features.relu0
	t4::release(x1017);
	t4::tensor4f x1019 = t4::MaxPool2d<3, 3, 2, 2, 1, 1>(x1018); //features.pool0
	t4::release(x1018);
	t4::tensor4f x1020 = t4::BatchNormalization(x1019, ctx.features_denseblock1_denselayer1_norm1_weight, ctx.features_denseblock1_denselayer1_norm1_bias, ctx.features_denseblock1_denselayer1_norm1_running_mean, ctx.features_denseblock1_denselayer1_norm1_running_var, 1e-05f); //features.denseblock1.denselayer1.norm1
	t4::tensor4f x1021 = t4::ReluInplace(x1020); //features.denseblock1.denselayer1.relu1
	t4::release(x1020);
	t4::tensor4f x1022 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1021, ctx.features_denseblock1_denselayer1_conv1_weight); //features.denseblock1.denselayer1.conv1
	t4::release(x1021);
	t4::tensor4f x1023 = t4::BatchNormalizationInplace(x1022, ctx.features_denseblock1_denselayer1_norm2_weight, ctx.features_denseblock1_denselayer1_norm2_bias, ctx.features_denseblock1_denselayer1_norm2_running_mean, ctx.features_denseblock1_denselayer1_norm2_running_var, 1e-05f); //features.denseblock1.denselayer1.norm2
	t4::release(x1022);
	t4::tensor4f x1024 = t4::ReluInplace(x1023); //features.denseblock1.denselayer1.relu2
	t4::release(x1023);
	t4::tensor4f x1025 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1024, ctx.features_denseblock1_denselayer1_conv2_weight); //features.denseblock1.denselayer1.conv2
	t4::release(x1024);
	t4::tensor4f x1026 = t4::Concat<1>(x1019, x1025); //features.denseblock1.denselayer1
	t4::release(x1019, x1025);
	t4::tensor4f x1027 = t4::BatchNormalization(x1026, ctx.features_denseblock1_denselayer2_norm1_weight, ctx.features_denseblock1_denselayer2_norm1_bias, ctx.features_denseblock1_denselayer2_norm1_running_mean, ctx.features_denseblock1_denselayer2_norm1_running_var, 1e-05f); //features.denseblock1.denselayer2.norm1
	t4::tensor4f x1028 = t4::ReluInplace(x1027); //features.denseblock1.denselayer2.relu1
	t4::release(x1027);
	t4::tensor4f x1029 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1028, ctx.features_denseblock1_denselayer2_conv1_weight); //features.denseblock1.denselayer2.conv1
	t4::release(x1028);
	t4::tensor4f x1030 = t4::BatchNormalizationInplace(x1029, ctx.features_denseblock1_denselayer2_norm2_weight, ctx.features_denseblock1_denselayer2_norm2_bias, ctx.features_denseblock1_denselayer2_norm2_running_mean, ctx.features_denseblock1_denselayer2_norm2_running_var, 1e-05f); //features.denseblock1.denselayer2.norm2
	t4::release(x1029);
	t4::tensor4f x1031 = t4::ReluInplace(x1030); //features.denseblock1.denselayer2.relu2
	t4::release(x1030);
	t4::tensor4f x1032 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1031, ctx.features_denseblock1_denselayer2_conv2_weight); //features.denseblock1.denselayer2.conv2
	t4::release(x1031);
	t4::tensor4f x1033 = t4::Concat<1>(x1026, x1032); //features.denseblock1.denselayer2
	t4::release(x1026, x1032);
	t4::tensor4f x1034 = t4::BatchNormalization(x1033, ctx.features_denseblock1_denselayer3_norm1_weight, ctx.features_denseblock1_denselayer3_norm1_bias, ctx.features_denseblock1_denselayer3_norm1_running_mean, ctx.features_denseblock1_denselayer3_norm1_running_var, 1e-05f); //features.denseblock1.denselayer3.norm1
	t4::tensor4f x1035 = t4::ReluInplace(x1034); //features.denseblock1.denselayer3.relu1
	t4::release(x1034);
	t4::tensor4f x1036 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1035, ctx.features_denseblock1_denselayer3_conv1_weight); //features.denseblock1.denselayer3.conv1
	t4::release(x1035);
	t4::tensor4f x1037 = t4::BatchNormalizationInplace(x1036, ctx.features_denseblock1_denselayer3_norm2_weight, ctx.features_denseblock1_denselayer3_norm2_bias, ctx.features_denseblock1_denselayer3_norm2_running_mean, ctx.features_denseblock1_denselayer3_norm2_running_var, 1e-05f); //features.denseblock1.denselayer3.norm2
	t4::release(x1036);
	t4::tensor4f x1038 = t4::ReluInplace(x1037); //features.denseblock1.denselayer3.relu2
	t4::release(x1037);
	t4::tensor4f x1039 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1038, ctx.features_denseblock1_denselayer3_conv2_weight); //features.denseblock1.denselayer3.conv2
	t4::release(x1038);
	t4::tensor4f x1040 = t4::Concat<1>(x1033, x1039); //features.denseblock1.denselayer3
	t4::release(x1033, x1039);
	t4::tensor4f x1041 = t4::BatchNormalization(x1040, ctx.features_denseblock1_denselayer4_norm1_weight, ctx.features_denseblock1_denselayer4_norm1_bias, ctx.features_denseblock1_denselayer4_norm1_running_mean, ctx.features_denseblock1_denselayer4_norm1_running_var, 1e-05f); //features.denseblock1.denselayer4.norm1
	t4::tensor4f x1042 = t4::ReluInplace(x1041); //features.denseblock1.denselayer4.relu1
	t4::release(x1041);
	t4::tensor4f x1043 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1042, ctx.features_denseblock1_denselayer4_conv1_weight); //features.denseblock1.denselayer4.conv1
	t4::release(x1042);
	t4::tensor4f x1044 = t4::BatchNormalizationInplace(x1043, ctx.features_denseblock1_denselayer4_norm2_weight, ctx.features_denseblock1_denselayer4_norm2_bias, ctx.features_denseblock1_denselayer4_norm2_running_mean, ctx.features_denseblock1_denselayer4_norm2_running_var, 1e-05f); //features.denseblock1.denselayer4.norm2
	t4::release(x1043);
	t4::tensor4f x1045 = t4::ReluInplace(x1044); //features.denseblock1.denselayer4.relu2
	t4::release(x1044);
	t4::tensor4f x1046 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1045, ctx.features_denseblock1_denselayer4_conv2_weight); //features.denseblock1.denselayer4.conv2
	t4::release(x1045);
	t4::tensor4f x1047 = t4::Concat<1>(x1040, x1046); //features.denseblock1.denselayer4
	t4::release(x1040, x1046);
	t4::tensor4f x1048 = t4::BatchNormalization(x1047, ctx.features_denseblock1_denselayer5_norm1_weight, ctx.features_denseblock1_denselayer5_norm1_bias, ctx.features_denseblock1_denselayer5_norm1_running_mean, ctx.features_denseblock1_denselayer5_norm1_running_var, 1e-05f); //features.denseblock1.denselayer5.norm1
	t4::tensor4f x1049 = t4::ReluInplace(x1048); //features.denseblock1.denselayer5.relu1
	t4::release(x1048);
	t4::tensor4f x1050 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1049, ctx.features_denseblock1_denselayer5_conv1_weight); //features.denseblock1.denselayer5.conv1
	t4::release(x1049);
	t4::tensor4f x1051 = t4::BatchNormalizationInplace(x1050, ctx.features_denseblock1_denselayer5_norm2_weight, ctx.features_denseblock1_denselayer5_norm2_bias, ctx.features_denseblock1_denselayer5_norm2_running_mean, ctx.features_denseblock1_denselayer5_norm2_running_var, 1e-05f); //features.denseblock1.denselayer5.norm2
	t4::release(x1050);
	t4::tensor4f x1052 = t4::ReluInplace(x1051); //features.denseblock1.denselayer5.relu2
	t4::release(x1051);
	t4::tensor4f x1053 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1052, ctx.features_denseblock1_denselayer5_conv2_weight); //features.denseblock1.denselayer5.conv2
	t4::release(x1052);
	t4::tensor4f x1054 = t4::Concat<1>(x1047, x1053); //features.denseblock1.denselayer5
	t4::release(x1047, x1053);
	t4::tensor4f x1055 = t4::BatchNormalization(x1054, ctx.features_denseblock1_denselayer6_norm1_weight, ctx.features_denseblock1_denselayer6_norm1_bias, ctx.features_denseblock1_denselayer6_norm1_running_mean, ctx.features_denseblock1_denselayer6_norm1_running_var, 1e-05f); //features.denseblock1.denselayer6.norm1
	t4::tensor4f x1056 = t4::ReluInplace(x1055); //features.denseblock1.denselayer6.relu1
	t4::release(x1055);
	t4::tensor4f x1057 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1056, ctx.features_denseblock1_denselayer6_conv1_weight); //features.denseblock1.denselayer6.conv1
	t4::release(x1056);
	t4::tensor4f x1058 = t4::BatchNormalizationInplace(x1057, ctx.features_denseblock1_denselayer6_norm2_weight, ctx.features_denseblock1_denselayer6_norm2_bias, ctx.features_denseblock1_denselayer6_norm2_running_mean, ctx.features_denseblock1_denselayer6_norm2_running_var, 1e-05f); //features.denseblock1.denselayer6.norm2
	t4::release(x1057);
	t4::tensor4f x1059 = t4::ReluInplace(x1058); //features.denseblock1.denselayer6.relu2
	t4::release(x1058);
	t4::tensor4f x1060 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1059, ctx.features_denseblock1_denselayer6_conv2_weight); //features.denseblock1.denselayer6.conv2
	t4::release(x1059);
	t4::tensor4f x1061 = t4::Concat<1>(x1054, x1060); //features.denseblock1.denselayer6
	t4::release(x1054, x1060);
	t4::tensor4f x1062 = t4::BatchNormalizationInplace(x1061, ctx.features_transition1_norm_weight, ctx.features_transition1_norm_bias, ctx.features_transition1_norm_running_mean, ctx.features_transition1_norm_running_var, 1e-05f); //features.transition1.norm
	t4::release(x1061);
	t4::tensor4f x1063 = t4::ReluInplace(x1062); //features.transition1.relu
	t4::release(x1062);
	t4::tensor4f x1064 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1063, ctx.features_transition1_conv_weight); //features.transition1.conv
	t4::release(x1063);
	t4::tensor4f x1065 = t4::Pad<t4::constant>(x1064, 0, 0, 0, 0, 0, 0, 0, 0); //features.transition1.pool
	t4::release(x1064);
	t4::tensor4f x1066 = t4::AveragePool2d<2, 2, 2, 2, 0, 0>(x1065); //features.transition1.pool
	t4::release(x1065);
	t4::tensor4f x1067 = t4::BatchNormalization(x1066, ctx.features_denseblock2_denselayer1_norm1_weight, ctx.features_denseblock2_denselayer1_norm1_bias, ctx.features_denseblock2_denselayer1_norm1_running_mean, ctx.features_denseblock2_denselayer1_norm1_running_var, 1e-05f); //features.denseblock2.denselayer1.norm1
	t4::tensor4f x1068 = t4::ReluInplace(x1067); //features.denseblock2.denselayer1.relu1
	t4::release(x1067);
	t4::tensor4f x1069 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1068, ctx.features_denseblock2_denselayer1_conv1_weight); //features.denseblock2.denselayer1.conv1
	t4::release(x1068);
	t4::tensor4f x1070 = t4::BatchNormalizationInplace(x1069, ctx.features_denseblock2_denselayer1_norm2_weight, ctx.features_denseblock2_denselayer1_norm2_bias, ctx.features_denseblock2_denselayer1_norm2_running_mean, ctx.features_denseblock2_denselayer1_norm2_running_var, 1e-05f); //features.denseblock2.denselayer1.norm2
	t4::release(x1069);
	t4::tensor4f x1071 = t4::ReluInplace(x1070); //features.denseblock2.denselayer1.relu2
	t4::release(x1070);
	t4::tensor4f x1072 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1071, ctx.features_denseblock2_denselayer1_conv2_weight); //features.denseblock2.denselayer1.conv2
	t4::release(x1071);
	t4::tensor4f x1073 = t4::Concat<1>(x1066, x1072); //features.denseblock2.denselayer1
	t4::release(x1066, x1072);
	t4::tensor4f x1074 = t4::BatchNormalization(x1073, ctx.features_denseblock2_denselayer2_norm1_weight, ctx.features_denseblock2_denselayer2_norm1_bias, ctx.features_denseblock2_denselayer2_norm1_running_mean, ctx.features_denseblock2_denselayer2_norm1_running_var, 1e-05f); //features.denseblock2.denselayer2.norm1
	t4::tensor4f x1075 = t4::ReluInplace(x1074); //features.denseblock2.denselayer2.relu1
	t4::release(x1074);
	t4::tensor4f x1076 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1075, ctx.features_denseblock2_denselayer2_conv1_weight); //features.denseblock2.denselayer2.conv1
	t4::release(x1075);
	t4::tensor4f x1077 = t4::BatchNormalizationInplace(x1076, ctx.features_denseblock2_denselayer2_norm2_weight, ctx.features_denseblock2_denselayer2_norm2_bias, ctx.features_denseblock2_denselayer2_norm2_running_mean, ctx.features_denseblock2_denselayer2_norm2_running_var, 1e-05f); //features.denseblock2.denselayer2.norm2
	t4::release(x1076);
	t4::tensor4f x1078 = t4::ReluInplace(x1077); //features.denseblock2.denselayer2.relu2
	t4::release(x1077);
	t4::tensor4f x1079 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1078, ctx.features_denseblock2_denselayer2_conv2_weight); //features.denseblock2.denselayer2.conv2
	t4::release(x1078);
	t4::tensor4f x1080 = t4::Concat<1>(x1073, x1079); //features.denseblock2.denselayer2
	t4::release(x1073, x1079);
	t4::tensor4f x1081 = t4::BatchNormalization(x1080, ctx.features_denseblock2_denselayer3_norm1_weight, ctx.features_denseblock2_denselayer3_norm1_bias, ctx.features_denseblock2_denselayer3_norm1_running_mean, ctx.features_denseblock2_denselayer3_norm1_running_var, 1e-05f); //features.denseblock2.denselayer3.norm1
	t4::tensor4f x1082 = t4::ReluInplace(x1081); //features.denseblock2.denselayer3.relu1
	t4::release(x1081);
	t4::tensor4f x1083 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1082, ctx.features_denseblock2_denselayer3_conv1_weight); //features.denseblock2.denselayer3.conv1
	t4::release(x1082);
	t4::tensor4f x1084 = t4::BatchNormalizationInplace(x1083, ctx.features_denseblock2_denselayer3_norm2_weight, ctx.features_denseblock2_denselayer3_norm2_bias, ctx.features_denseblock2_denselayer3_norm2_running_mean, ctx.features_denseblock2_denselayer3_norm2_running_var, 1e-05f); //features.denseblock2.denselayer3.norm2
	t4::release(x1083);
	t4::tensor4f x1085 = t4::ReluInplace(x1084); //features.denseblock2.denselayer3.relu2
	t4::release(x1084);
	t4::tensor4f x1086 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1085, ctx.features_denseblock2_denselayer3_conv2_weight); //features.denseblock2.denselayer3.conv2
	t4::release(x1085);
	t4::tensor4f x1087 = t4::Concat<1>(x1080, x1086); //features.denseblock2.denselayer3
	t4::release(x1080, x1086);
	t4::tensor4f x1088 = t4::BatchNormalization(x1087, ctx.features_denseblock2_denselayer4_norm1_weight, ctx.features_denseblock2_denselayer4_norm1_bias, ctx.features_denseblock2_denselayer4_norm1_running_mean, ctx.features_denseblock2_denselayer4_norm1_running_var, 1e-05f); //features.denseblock2.denselayer4.norm1
	t4::tensor4f x1089 = t4::ReluInplace(x1088); //features.denseblock2.denselayer4.relu1
	t4::release(x1088);
	t4::tensor4f x1090 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1089, ctx.features_denseblock2_denselayer4_conv1_weight); //features.denseblock2.denselayer4.conv1
	t4::release(x1089);
	t4::tensor4f x1091 = t4::BatchNormalizationInplace(x1090, ctx.features_denseblock2_denselayer4_norm2_weight, ctx.features_denseblock2_denselayer4_norm2_bias, ctx.features_denseblock2_denselayer4_norm2_running_mean, ctx.features_denseblock2_denselayer4_norm2_running_var, 1e-05f); //features.denseblock2.denselayer4.norm2
	t4::release(x1090);
	t4::tensor4f x1092 = t4::ReluInplace(x1091); //features.denseblock2.denselayer4.relu2
	t4::release(x1091);
	t4::tensor4f x1093 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1092, ctx.features_denseblock2_denselayer4_conv2_weight); //features.denseblock2.denselayer4.conv2
	t4::release(x1092);
	t4::tensor4f x1094 = t4::Concat<1>(x1087, x1093); //features.denseblock2.denselayer4
	t4::release(x1087, x1093);
	t4::tensor4f x1095 = t4::BatchNormalization(x1094, ctx.features_denseblock2_denselayer5_norm1_weight, ctx.features_denseblock2_denselayer5_norm1_bias, ctx.features_denseblock2_denselayer5_norm1_running_mean, ctx.features_denseblock2_denselayer5_norm1_running_var, 1e-05f); //features.denseblock2.denselayer5.norm1
	t4::tensor4f x1096 = t4::ReluInplace(x1095); //features.denseblock2.denselayer5.relu1
	t4::release(x1095);
	t4::tensor4f x1097 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1096, ctx.features_denseblock2_denselayer5_conv1_weight); //features.denseblock2.denselayer5.conv1
	t4::release(x1096);
	t4::tensor4f x1098 = t4::BatchNormalizationInplace(x1097, ctx.features_denseblock2_denselayer5_norm2_weight, ctx.features_denseblock2_denselayer5_norm2_bias, ctx.features_denseblock2_denselayer5_norm2_running_mean, ctx.features_denseblock2_denselayer5_norm2_running_var, 1e-05f); //features.denseblock2.denselayer5.norm2
	t4::release(x1097);
	t4::tensor4f x1099 = t4::ReluInplace(x1098); //features.denseblock2.denselayer5.relu2
	t4::release(x1098);
	t4::tensor4f x1100 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1099, ctx.features_denseblock2_denselayer5_conv2_weight); //features.denseblock2.denselayer5.conv2
	t4::release(x1099);
	t4::tensor4f x1101 = t4::Concat<1>(x1094, x1100); //features.denseblock2.denselayer5
	t4::release(x1094, x1100);
	t4::tensor4f x1102 = t4::BatchNormalization(x1101, ctx.features_denseblock2_denselayer6_norm1_weight, ctx.features_denseblock2_denselayer6_norm1_bias, ctx.features_denseblock2_denselayer6_norm1_running_mean, ctx.features_denseblock2_denselayer6_norm1_running_var, 1e-05f); //features.denseblock2.denselayer6.norm1
	t4::tensor4f x1103 = t4::ReluInplace(x1102); //features.denseblock2.denselayer6.relu1
	t4::release(x1102);
	t4::tensor4f x1104 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1103, ctx.features_denseblock2_denselayer6_conv1_weight); //features.denseblock2.denselayer6.conv1
	t4::release(x1103);
	t4::tensor4f x1105 = t4::BatchNormalizationInplace(x1104, ctx.features_denseblock2_denselayer6_norm2_weight, ctx.features_denseblock2_denselayer6_norm2_bias, ctx.features_denseblock2_denselayer6_norm2_running_mean, ctx.features_denseblock2_denselayer6_norm2_running_var, 1e-05f); //features.denseblock2.denselayer6.norm2
	t4::release(x1104);
	t4::tensor4f x1106 = t4::ReluInplace(x1105); //features.denseblock2.denselayer6.relu2
	t4::release(x1105);
	t4::tensor4f x1107 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1106, ctx.features_denseblock2_denselayer6_conv2_weight); //features.denseblock2.denselayer6.conv2
	t4::release(x1106);
	t4::tensor4f x1108 = t4::Concat<1>(x1101, x1107); //features.denseblock2.denselayer6
	t4::release(x1101, x1107);
	t4::tensor4f x1109 = t4::BatchNormalization(x1108, ctx.features_denseblock2_denselayer7_norm1_weight, ctx.features_denseblock2_denselayer7_norm1_bias, ctx.features_denseblock2_denselayer7_norm1_running_mean, ctx.features_denseblock2_denselayer7_norm1_running_var, 1e-05f); //features.denseblock2.denselayer7.norm1
	t4::tensor4f x1110 = t4::ReluInplace(x1109); //features.denseblock2.denselayer7.relu1
	t4::release(x1109);
	t4::tensor4f x1111 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1110, ctx.features_denseblock2_denselayer7_conv1_weight); //features.denseblock2.denselayer7.conv1
	t4::release(x1110);
	t4::tensor4f x1112 = t4::BatchNormalizationInplace(x1111, ctx.features_denseblock2_denselayer7_norm2_weight, ctx.features_denseblock2_denselayer7_norm2_bias, ctx.features_denseblock2_denselayer7_norm2_running_mean, ctx.features_denseblock2_denselayer7_norm2_running_var, 1e-05f); //features.denseblock2.denselayer7.norm2
	t4::release(x1111);
	t4::tensor4f x1113 = t4::ReluInplace(x1112); //features.denseblock2.denselayer7.relu2
	t4::release(x1112);
	t4::tensor4f x1114 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1113, ctx.features_denseblock2_denselayer7_conv2_weight); //features.denseblock2.denselayer7.conv2
	t4::release(x1113);
	t4::tensor4f x1115 = t4::Concat<1>(x1108, x1114); //features.denseblock2.denselayer7
	t4::release(x1108, x1114);
	t4::tensor4f x1116 = t4::BatchNormalization(x1115, ctx.features_denseblock2_denselayer8_norm1_weight, ctx.features_denseblock2_denselayer8_norm1_bias, ctx.features_denseblock2_denselayer8_norm1_running_mean, ctx.features_denseblock2_denselayer8_norm1_running_var, 1e-05f); //features.denseblock2.denselayer8.norm1
	t4::tensor4f x1117 = t4::ReluInplace(x1116); //features.denseblock2.denselayer8.relu1
	t4::release(x1116);
	t4::tensor4f x1118 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1117, ctx.features_denseblock2_denselayer8_conv1_weight); //features.denseblock2.denselayer8.conv1
	t4::release(x1117);
	t4::tensor4f x1119 = t4::BatchNormalizationInplace(x1118, ctx.features_denseblock2_denselayer8_norm2_weight, ctx.features_denseblock2_denselayer8_norm2_bias, ctx.features_denseblock2_denselayer8_norm2_running_mean, ctx.features_denseblock2_denselayer8_norm2_running_var, 1e-05f); //features.denseblock2.denselayer8.norm2
	t4::release(x1118);
	t4::tensor4f x1120 = t4::ReluInplace(x1119); //features.denseblock2.denselayer8.relu2
	t4::release(x1119);
	t4::tensor4f x1121 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1120, ctx.features_denseblock2_denselayer8_conv2_weight); //features.denseblock2.denselayer8.conv2
	t4::release(x1120);
	t4::tensor4f x1122 = t4::Concat<1>(x1115, x1121); //features.denseblock2.denselayer8
	t4::release(x1115, x1121);
	t4::tensor4f x1123 = t4::BatchNormalization(x1122, ctx.features_denseblock2_denselayer9_norm1_weight, ctx.features_denseblock2_denselayer9_norm1_bias, ctx.features_denseblock2_denselayer9_norm1_running_mean, ctx.features_denseblock2_denselayer9_norm1_running_var, 1e-05f); //features.denseblock2.denselayer9.norm1
	t4::tensor4f x1124 = t4::ReluInplace(x1123); //features.denseblock2.denselayer9.relu1
	t4::release(x1123);
	t4::tensor4f x1125 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1124, ctx.features_denseblock2_denselayer9_conv1_weight); //features.denseblock2.denselayer9.conv1
	t4::release(x1124);
	t4::tensor4f x1126 = t4::BatchNormalizationInplace(x1125, ctx.features_denseblock2_denselayer9_norm2_weight, ctx.features_denseblock2_denselayer9_norm2_bias, ctx.features_denseblock2_denselayer9_norm2_running_mean, ctx.features_denseblock2_denselayer9_norm2_running_var, 1e-05f); //features.denseblock2.denselayer9.norm2
	t4::release(x1125);
	t4::tensor4f x1127 = t4::ReluInplace(x1126); //features.denseblock2.denselayer9.relu2
	t4::release(x1126);
	t4::tensor4f x1128 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1127, ctx.features_denseblock2_denselayer9_conv2_weight); //features.denseblock2.denselayer9.conv2
	t4::release(x1127);
	t4::tensor4f x1129 = t4::Concat<1>(x1122, x1128); //features.denseblock2.denselayer9
	t4::release(x1122, x1128);
	t4::tensor4f x1130 = t4::BatchNormalization(x1129, ctx.features_denseblock2_denselayer10_norm1_weight, ctx.features_denseblock2_denselayer10_norm1_bias, ctx.features_denseblock2_denselayer10_norm1_running_mean, ctx.features_denseblock2_denselayer10_norm1_running_var, 1e-05f); //features.denseblock2.denselayer10.norm1
	t4::tensor4f x1131 = t4::ReluInplace(x1130); //features.denseblock2.denselayer10.relu1
	t4::release(x1130);
	t4::tensor4f x1132 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1131, ctx.features_denseblock2_denselayer10_conv1_weight); //features.denseblock2.denselayer10.conv1
	t4::release(x1131);
	t4::tensor4f x1133 = t4::BatchNormalizationInplace(x1132, ctx.features_denseblock2_denselayer10_norm2_weight, ctx.features_denseblock2_denselayer10_norm2_bias, ctx.features_denseblock2_denselayer10_norm2_running_mean, ctx.features_denseblock2_denselayer10_norm2_running_var, 1e-05f); //features.denseblock2.denselayer10.norm2
	t4::release(x1132);
	t4::tensor4f x1134 = t4::ReluInplace(x1133); //features.denseblock2.denselayer10.relu2
	t4::release(x1133);
	t4::tensor4f x1135 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1134, ctx.features_denseblock2_denselayer10_conv2_weight); //features.denseblock2.denselayer10.conv2
	t4::release(x1134);
	t4::tensor4f x1136 = t4::Concat<1>(x1129, x1135); //features.denseblock2.denselayer10
	t4::release(x1129, x1135);
	t4::tensor4f x1137 = t4::BatchNormalization(x1136, ctx.features_denseblock2_denselayer11_norm1_weight, ctx.features_denseblock2_denselayer11_norm1_bias, ctx.features_denseblock2_denselayer11_norm1_running_mean, ctx.features_denseblock2_denselayer11_norm1_running_var, 1e-05f); //features.denseblock2.denselayer11.norm1
	t4::tensor4f x1138 = t4::ReluInplace(x1137); //features.denseblock2.denselayer11.relu1
	t4::release(x1137);
	t4::tensor4f x1139 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1138, ctx.features_denseblock2_denselayer11_conv1_weight); //features.denseblock2.denselayer11.conv1
	t4::release(x1138);
	t4::tensor4f x1140 = t4::BatchNormalizationInplace(x1139, ctx.features_denseblock2_denselayer11_norm2_weight, ctx.features_denseblock2_denselayer11_norm2_bias, ctx.features_denseblock2_denselayer11_norm2_running_mean, ctx.features_denseblock2_denselayer11_norm2_running_var, 1e-05f); //features.denseblock2.denselayer11.norm2
	t4::release(x1139);
	t4::tensor4f x1141 = t4::ReluInplace(x1140); //features.denseblock2.denselayer11.relu2
	t4::release(x1140);
	t4::tensor4f x1142 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1141, ctx.features_denseblock2_denselayer11_conv2_weight); //features.denseblock2.denselayer11.conv2
	t4::release(x1141);
	t4::tensor4f x1143 = t4::Concat<1>(x1136, x1142); //features.denseblock2.denselayer11
	t4::release(x1136, x1142);
	t4::tensor4f x1144 = t4::BatchNormalization(x1143, ctx.features_denseblock2_denselayer12_norm1_weight, ctx.features_denseblock2_denselayer12_norm1_bias, ctx.features_denseblock2_denselayer12_norm1_running_mean, ctx.features_denseblock2_denselayer12_norm1_running_var, 1e-05f); //features.denseblock2.denselayer12.norm1
	t4::tensor4f x1145 = t4::ReluInplace(x1144); //features.denseblock2.denselayer12.relu1
	t4::release(x1144);
	t4::tensor4f x1146 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1145, ctx.features_denseblock2_denselayer12_conv1_weight); //features.denseblock2.denselayer12.conv1
	t4::release(x1145);
	t4::tensor4f x1147 = t4::BatchNormalizationInplace(x1146, ctx.features_denseblock2_denselayer12_norm2_weight, ctx.features_denseblock2_denselayer12_norm2_bias, ctx.features_denseblock2_denselayer12_norm2_running_mean, ctx.features_denseblock2_denselayer12_norm2_running_var, 1e-05f); //features.denseblock2.denselayer12.norm2
	t4::release(x1146);
	t4::tensor4f x1148 = t4::ReluInplace(x1147); //features.denseblock2.denselayer12.relu2
	t4::release(x1147);
	t4::tensor4f x1149 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1148, ctx.features_denseblock2_denselayer12_conv2_weight); //features.denseblock2.denselayer12.conv2
	t4::release(x1148);
	t4::tensor4f x1150 = t4::Concat<1>(x1143, x1149); //features.denseblock2.denselayer12
	t4::release(x1143, x1149);
	t4::tensor4f x1151 = t4::BatchNormalizationInplace(x1150, ctx.features_transition2_norm_weight, ctx.features_transition2_norm_bias, ctx.features_transition2_norm_running_mean, ctx.features_transition2_norm_running_var, 1e-05f); //features.transition2.norm
	t4::release(x1150);
	t4::tensor4f x1152 = t4::ReluInplace(x1151); //features.transition2.relu
	t4::release(x1151);
	t4::tensor4f x1153 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1152, ctx.features_transition2_conv_weight); //features.transition2.conv
	t4::release(x1152);
	t4::tensor4f x1154 = t4::Pad<t4::constant>(x1153, 0, 0, 0, 0, 0, 0, 0, 0); //features.transition2.pool
	t4::release(x1153);
	t4::tensor4f x1155 = t4::AveragePool2d<2, 2, 2, 2, 0, 0>(x1154); //features.transition2.pool
	t4::release(x1154);
	t4::tensor4f x1156 = t4::BatchNormalization(x1155, ctx.features_denseblock3_denselayer1_norm1_weight, ctx.features_denseblock3_denselayer1_norm1_bias, ctx.features_denseblock3_denselayer1_norm1_running_mean, ctx.features_denseblock3_denselayer1_norm1_running_var, 1e-05f); //features.denseblock3.denselayer1.norm1
	t4::tensor4f x1157 = t4::ReluInplace(x1156); //features.denseblock3.denselayer1.relu1
	t4::release(x1156);
	t4::tensor4f x1158 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1157, ctx.features_denseblock3_denselayer1_conv1_weight); //features.denseblock3.denselayer1.conv1
	t4::release(x1157);
	t4::tensor4f x1159 = t4::BatchNormalizationInplace(x1158, ctx.features_denseblock3_denselayer1_norm2_weight, ctx.features_denseblock3_denselayer1_norm2_bias, ctx.features_denseblock3_denselayer1_norm2_running_mean, ctx.features_denseblock3_denselayer1_norm2_running_var, 1e-05f); //features.denseblock3.denselayer1.norm2
	t4::release(x1158);
	t4::tensor4f x1160 = t4::ReluInplace(x1159); //features.denseblock3.denselayer1.relu2
	t4::release(x1159);
	t4::tensor4f x1161 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1160, ctx.features_denseblock3_denselayer1_conv2_weight); //features.denseblock3.denselayer1.conv2
	t4::release(x1160);
	t4::tensor4f x1162 = t4::Concat<1>(x1155, x1161); //features.denseblock3.denselayer1
	t4::release(x1155, x1161);
	t4::tensor4f x1163 = t4::BatchNormalization(x1162, ctx.features_denseblock3_denselayer2_norm1_weight, ctx.features_denseblock3_denselayer2_norm1_bias, ctx.features_denseblock3_denselayer2_norm1_running_mean, ctx.features_denseblock3_denselayer2_norm1_running_var, 1e-05f); //features.denseblock3.denselayer2.norm1
	t4::tensor4f x1164 = t4::ReluInplace(x1163); //features.denseblock3.denselayer2.relu1
	t4::release(x1163);
	t4::tensor4f x1165 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1164, ctx.features_denseblock3_denselayer2_conv1_weight); //features.denseblock3.denselayer2.conv1
	t4::release(x1164);
	t4::tensor4f x1166 = t4::BatchNormalizationInplace(x1165, ctx.features_denseblock3_denselayer2_norm2_weight, ctx.features_denseblock3_denselayer2_norm2_bias, ctx.features_denseblock3_denselayer2_norm2_running_mean, ctx.features_denseblock3_denselayer2_norm2_running_var, 1e-05f); //features.denseblock3.denselayer2.norm2
	t4::release(x1165);
	t4::tensor4f x1167 = t4::ReluInplace(x1166); //features.denseblock3.denselayer2.relu2
	t4::release(x1166);
	t4::tensor4f x1168 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1167, ctx.features_denseblock3_denselayer2_conv2_weight); //features.denseblock3.denselayer2.conv2
	t4::release(x1167);
	t4::tensor4f x1169 = t4::Concat<1>(x1162, x1168); //features.denseblock3.denselayer2
	t4::release(x1162, x1168);
	t4::tensor4f x1170 = t4::BatchNormalization(x1169, ctx.features_denseblock3_denselayer3_norm1_weight, ctx.features_denseblock3_denselayer3_norm1_bias, ctx.features_denseblock3_denselayer3_norm1_running_mean, ctx.features_denseblock3_denselayer3_norm1_running_var, 1e-05f); //features.denseblock3.denselayer3.norm1
	t4::tensor4f x1171 = t4::ReluInplace(x1170); //features.denseblock3.denselayer3.relu1
	t4::release(x1170);
	t4::tensor4f x1172 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1171, ctx.features_denseblock3_denselayer3_conv1_weight); //features.denseblock3.denselayer3.conv1
	t4::release(x1171);
	t4::tensor4f x1173 = t4::BatchNormalizationInplace(x1172, ctx.features_denseblock3_denselayer3_norm2_weight, ctx.features_denseblock3_denselayer3_norm2_bias, ctx.features_denseblock3_denselayer3_norm2_running_mean, ctx.features_denseblock3_denselayer3_norm2_running_var, 1e-05f); //features.denseblock3.denselayer3.norm2
	t4::release(x1172);
	t4::tensor4f x1174 = t4::ReluInplace(x1173); //features.denseblock3.denselayer3.relu2
	t4::release(x1173);
	t4::tensor4f x1175 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1174, ctx.features_denseblock3_denselayer3_conv2_weight); //features.denseblock3.denselayer3.conv2
	t4::release(x1174);
	t4::tensor4f x1176 = t4::Concat<1>(x1169, x1175); //features.denseblock3.denselayer3
	t4::release(x1169, x1175);
	t4::tensor4f x1177 = t4::BatchNormalization(x1176, ctx.features_denseblock3_denselayer4_norm1_weight, ctx.features_denseblock3_denselayer4_norm1_bias, ctx.features_denseblock3_denselayer4_norm1_running_mean, ctx.features_denseblock3_denselayer4_norm1_running_var, 1e-05f); //features.denseblock3.denselayer4.norm1
	t4::tensor4f x1178 = t4::ReluInplace(x1177); //features.denseblock3.denselayer4.relu1
	t4::release(x1177);
	t4::tensor4f x1179 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1178, ctx.features_denseblock3_denselayer4_conv1_weight); //features.denseblock3.denselayer4.conv1
	t4::release(x1178);
	t4::tensor4f x1180 = t4::BatchNormalizationInplace(x1179, ctx.features_denseblock3_denselayer4_norm2_weight, ctx.features_denseblock3_denselayer4_norm2_bias, ctx.features_denseblock3_denselayer4_norm2_running_mean, ctx.features_denseblock3_denselayer4_norm2_running_var, 1e-05f); //features.denseblock3.denselayer4.norm2
	t4::release(x1179);
	t4::tensor4f x1181 = t4::ReluInplace(x1180); //features.denseblock3.denselayer4.relu2
	t4::release(x1180);
	t4::tensor4f x1182 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1181, ctx.features_denseblock3_denselayer4_conv2_weight); //features.denseblock3.denselayer4.conv2
	t4::release(x1181);
	t4::tensor4f x1183 = t4::Concat<1>(x1176, x1182); //features.denseblock3.denselayer4
	t4::release(x1176, x1182);
	t4::tensor4f x1184 = t4::BatchNormalization(x1183, ctx.features_denseblock3_denselayer5_norm1_weight, ctx.features_denseblock3_denselayer5_norm1_bias, ctx.features_denseblock3_denselayer5_norm1_running_mean, ctx.features_denseblock3_denselayer5_norm1_running_var, 1e-05f); //features.denseblock3.denselayer5.norm1
	t4::tensor4f x1185 = t4::ReluInplace(x1184); //features.denseblock3.denselayer5.relu1
	t4::release(x1184);
	t4::tensor4f x1186 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1185, ctx.features_denseblock3_denselayer5_conv1_weight); //features.denseblock3.denselayer5.conv1
	t4::release(x1185);
	t4::tensor4f x1187 = t4::BatchNormalizationInplace(x1186, ctx.features_denseblock3_denselayer5_norm2_weight, ctx.features_denseblock3_denselayer5_norm2_bias, ctx.features_denseblock3_denselayer5_norm2_running_mean, ctx.features_denseblock3_denselayer5_norm2_running_var, 1e-05f); //features.denseblock3.denselayer5.norm2
	t4::release(x1186);
	t4::tensor4f x1188 = t4::ReluInplace(x1187); //features.denseblock3.denselayer5.relu2
	t4::release(x1187);
	t4::tensor4f x1189 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1188, ctx.features_denseblock3_denselayer5_conv2_weight); //features.denseblock3.denselayer5.conv2
	t4::release(x1188);
	t4::tensor4f x1190 = t4::Concat<1>(x1183, x1189); //features.denseblock3.denselayer5
	t4::release(x1183, x1189);
	t4::tensor4f x1191 = t4::BatchNormalization(x1190, ctx.features_denseblock3_denselayer6_norm1_weight, ctx.features_denseblock3_denselayer6_norm1_bias, ctx.features_denseblock3_denselayer6_norm1_running_mean, ctx.features_denseblock3_denselayer6_norm1_running_var, 1e-05f); //features.denseblock3.denselayer6.norm1
	t4::tensor4f x1192 = t4::ReluInplace(x1191); //features.denseblock3.denselayer6.relu1
	t4::release(x1191);
	t4::tensor4f x1193 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1192, ctx.features_denseblock3_denselayer6_conv1_weight); //features.denseblock3.denselayer6.conv1
	t4::release(x1192);
	t4::tensor4f x1194 = t4::BatchNormalizationInplace(x1193, ctx.features_denseblock3_denselayer6_norm2_weight, ctx.features_denseblock3_denselayer6_norm2_bias, ctx.features_denseblock3_denselayer6_norm2_running_mean, ctx.features_denseblock3_denselayer6_norm2_running_var, 1e-05f); //features.denseblock3.denselayer6.norm2
	t4::release(x1193);
	t4::tensor4f x1195 = t4::ReluInplace(x1194); //features.denseblock3.denselayer6.relu2
	t4::release(x1194);
	t4::tensor4f x1196 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1195, ctx.features_denseblock3_denselayer6_conv2_weight); //features.denseblock3.denselayer6.conv2
	t4::release(x1195);
	t4::tensor4f x1197 = t4::Concat<1>(x1190, x1196); //features.denseblock3.denselayer6
	t4::release(x1190, x1196);
	t4::tensor4f x1198 = t4::BatchNormalization(x1197, ctx.features_denseblock3_denselayer7_norm1_weight, ctx.features_denseblock3_denselayer7_norm1_bias, ctx.features_denseblock3_denselayer7_norm1_running_mean, ctx.features_denseblock3_denselayer7_norm1_running_var, 1e-05f); //features.denseblock3.denselayer7.norm1
	t4::tensor4f x1199 = t4::ReluInplace(x1198); //features.denseblock3.denselayer7.relu1
	t4::release(x1198);
	t4::tensor4f x1200 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1199, ctx.features_denseblock3_denselayer7_conv1_weight); //features.denseblock3.denselayer7.conv1
	t4::release(x1199);
	t4::tensor4f x1201 = t4::BatchNormalizationInplace(x1200, ctx.features_denseblock3_denselayer7_norm2_weight, ctx.features_denseblock3_denselayer7_norm2_bias, ctx.features_denseblock3_denselayer7_norm2_running_mean, ctx.features_denseblock3_denselayer7_norm2_running_var, 1e-05f); //features.denseblock3.denselayer7.norm2
	t4::release(x1200);
	t4::tensor4f x1202 = t4::ReluInplace(x1201); //features.denseblock3.denselayer7.relu2
	t4::release(x1201);
	t4::tensor4f x1203 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1202, ctx.features_denseblock3_denselayer7_conv2_weight); //features.denseblock3.denselayer7.conv2
	t4::release(x1202);
	t4::tensor4f x1204 = t4::Concat<1>(x1197, x1203); //features.denseblock3.denselayer7
	t4::release(x1197, x1203);
	t4::tensor4f x1205 = t4::BatchNormalization(x1204, ctx.features_denseblock3_denselayer8_norm1_weight, ctx.features_denseblock3_denselayer8_norm1_bias, ctx.features_denseblock3_denselayer8_norm1_running_mean, ctx.features_denseblock3_denselayer8_norm1_running_var, 1e-05f); //features.denseblock3.denselayer8.norm1
	t4::tensor4f x1206 = t4::ReluInplace(x1205); //features.denseblock3.denselayer8.relu1
	t4::release(x1205);
	t4::tensor4f x1207 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1206, ctx.features_denseblock3_denselayer8_conv1_weight); //features.denseblock3.denselayer8.conv1
	t4::release(x1206);
	t4::tensor4f x1208 = t4::BatchNormalizationInplace(x1207, ctx.features_denseblock3_denselayer8_norm2_weight, ctx.features_denseblock3_denselayer8_norm2_bias, ctx.features_denseblock3_denselayer8_norm2_running_mean, ctx.features_denseblock3_denselayer8_norm2_running_var, 1e-05f); //features.denseblock3.denselayer8.norm2
	t4::release(x1207);
	t4::tensor4f x1209 = t4::ReluInplace(x1208); //features.denseblock3.denselayer8.relu2
	t4::release(x1208);
	t4::tensor4f x1210 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1209, ctx.features_denseblock3_denselayer8_conv2_weight); //features.denseblock3.denselayer8.conv2
	t4::release(x1209);
	t4::tensor4f x1211 = t4::Concat<1>(x1204, x1210); //features.denseblock3.denselayer8
	t4::release(x1204, x1210);
	t4::tensor4f x1212 = t4::BatchNormalization(x1211, ctx.features_denseblock3_denselayer9_norm1_weight, ctx.features_denseblock3_denselayer9_norm1_bias, ctx.features_denseblock3_denselayer9_norm1_running_mean, ctx.features_denseblock3_denselayer9_norm1_running_var, 1e-05f); //features.denseblock3.denselayer9.norm1
	t4::tensor4f x1213 = t4::ReluInplace(x1212); //features.denseblock3.denselayer9.relu1
	t4::release(x1212);
	t4::tensor4f x1214 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1213, ctx.features_denseblock3_denselayer9_conv1_weight); //features.denseblock3.denselayer9.conv1
	t4::release(x1213);
	t4::tensor4f x1215 = t4::BatchNormalizationInplace(x1214, ctx.features_denseblock3_denselayer9_norm2_weight, ctx.features_denseblock3_denselayer9_norm2_bias, ctx.features_denseblock3_denselayer9_norm2_running_mean, ctx.features_denseblock3_denselayer9_norm2_running_var, 1e-05f); //features.denseblock3.denselayer9.norm2
	t4::release(x1214);
	t4::tensor4f x1216 = t4::ReluInplace(x1215); //features.denseblock3.denselayer9.relu2
	t4::release(x1215);
	t4::tensor4f x1217 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1216, ctx.features_denseblock3_denselayer9_conv2_weight); //features.denseblock3.denselayer9.conv2
	t4::release(x1216);
	t4::tensor4f x1218 = t4::Concat<1>(x1211, x1217); //features.denseblock3.denselayer9
	t4::release(x1211, x1217);
	t4::tensor4f x1219 = t4::BatchNormalization(x1218, ctx.features_denseblock3_denselayer10_norm1_weight, ctx.features_denseblock3_denselayer10_norm1_bias, ctx.features_denseblock3_denselayer10_norm1_running_mean, ctx.features_denseblock3_denselayer10_norm1_running_var, 1e-05f); //features.denseblock3.denselayer10.norm1
	t4::tensor4f x1220 = t4::ReluInplace(x1219); //features.denseblock3.denselayer10.relu1
	t4::release(x1219);
	t4::tensor4f x1221 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1220, ctx.features_denseblock3_denselayer10_conv1_weight); //features.denseblock3.denselayer10.conv1
	t4::release(x1220);
	t4::tensor4f x1222 = t4::BatchNormalizationInplace(x1221, ctx.features_denseblock3_denselayer10_norm2_weight, ctx.features_denseblock3_denselayer10_norm2_bias, ctx.features_denseblock3_denselayer10_norm2_running_mean, ctx.features_denseblock3_denselayer10_norm2_running_var, 1e-05f); //features.denseblock3.denselayer10.norm2
	t4::release(x1221);
	t4::tensor4f x1223 = t4::ReluInplace(x1222); //features.denseblock3.denselayer10.relu2
	t4::release(x1222);
	t4::tensor4f x1224 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1223, ctx.features_denseblock3_denselayer10_conv2_weight); //features.denseblock3.denselayer10.conv2
	t4::release(x1223);
	t4::tensor4f x1225 = t4::Concat<1>(x1218, x1224); //features.denseblock3.denselayer10
	t4::release(x1218, x1224);
	t4::tensor4f x1226 = t4::BatchNormalization(x1225, ctx.features_denseblock3_denselayer11_norm1_weight, ctx.features_denseblock3_denselayer11_norm1_bias, ctx.features_denseblock3_denselayer11_norm1_running_mean, ctx.features_denseblock3_denselayer11_norm1_running_var, 1e-05f); //features.denseblock3.denselayer11.norm1
	t4::tensor4f x1227 = t4::ReluInplace(x1226); //features.denseblock3.denselayer11.relu1
	t4::release(x1226);
	t4::tensor4f x1228 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1227, ctx.features_denseblock3_denselayer11_conv1_weight); //features.denseblock3.denselayer11.conv1
	t4::release(x1227);
	t4::tensor4f x1229 = t4::BatchNormalizationInplace(x1228, ctx.features_denseblock3_denselayer11_norm2_weight, ctx.features_denseblock3_denselayer11_norm2_bias, ctx.features_denseblock3_denselayer11_norm2_running_mean, ctx.features_denseblock3_denselayer11_norm2_running_var, 1e-05f); //features.denseblock3.denselayer11.norm2
	t4::release(x1228);
	t4::tensor4f x1230 = t4::ReluInplace(x1229); //features.denseblock3.denselayer11.relu2
	t4::release(x1229);
	t4::tensor4f x1231 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1230, ctx.features_denseblock3_denselayer11_conv2_weight); //features.denseblock3.denselayer11.conv2
	t4::release(x1230);
	t4::tensor4f x1232 = t4::Concat<1>(x1225, x1231); //features.denseblock3.denselayer11
	t4::release(x1225, x1231);
	t4::tensor4f x1233 = t4::BatchNormalization(x1232, ctx.features_denseblock3_denselayer12_norm1_weight, ctx.features_denseblock3_denselayer12_norm1_bias, ctx.features_denseblock3_denselayer12_norm1_running_mean, ctx.features_denseblock3_denselayer12_norm1_running_var, 1e-05f); //features.denseblock3.denselayer12.norm1
	t4::tensor4f x1234 = t4::ReluInplace(x1233); //features.denseblock3.denselayer12.relu1
	t4::release(x1233);
	t4::tensor4f x1235 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1234, ctx.features_denseblock3_denselayer12_conv1_weight); //features.denseblock3.denselayer12.conv1
	t4::release(x1234);
	t4::tensor4f x1236 = t4::BatchNormalizationInplace(x1235, ctx.features_denseblock3_denselayer12_norm2_weight, ctx.features_denseblock3_denselayer12_norm2_bias, ctx.features_denseblock3_denselayer12_norm2_running_mean, ctx.features_denseblock3_denselayer12_norm2_running_var, 1e-05f); //features.denseblock3.denselayer12.norm2
	t4::release(x1235);
	t4::tensor4f x1237 = t4::ReluInplace(x1236); //features.denseblock3.denselayer12.relu2
	t4::release(x1236);
	t4::tensor4f x1238 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1237, ctx.features_denseblock3_denselayer12_conv2_weight); //features.denseblock3.denselayer12.conv2
	t4::release(x1237);
	t4::tensor4f x1239 = t4::Concat<1>(x1232, x1238); //features.denseblock3.denselayer12
	t4::release(x1232, x1238);
	t4::tensor4f x1240 = t4::BatchNormalization(x1239, ctx.features_denseblock3_denselayer13_norm1_weight, ctx.features_denseblock3_denselayer13_norm1_bias, ctx.features_denseblock3_denselayer13_norm1_running_mean, ctx.features_denseblock3_denselayer13_norm1_running_var, 1e-05f); //features.denseblock3.denselayer13.norm1
	t4::tensor4f x1241 = t4::ReluInplace(x1240); //features.denseblock3.denselayer13.relu1
	t4::release(x1240);
	t4::tensor4f x1242 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1241, ctx.features_denseblock3_denselayer13_conv1_weight); //features.denseblock3.denselayer13.conv1
	t4::release(x1241);
	t4::tensor4f x1243 = t4::BatchNormalizationInplace(x1242, ctx.features_denseblock3_denselayer13_norm2_weight, ctx.features_denseblock3_denselayer13_norm2_bias, ctx.features_denseblock3_denselayer13_norm2_running_mean, ctx.features_denseblock3_denselayer13_norm2_running_var, 1e-05f); //features.denseblock3.denselayer13.norm2
	t4::release(x1242);
	t4::tensor4f x1244 = t4::ReluInplace(x1243); //features.denseblock3.denselayer13.relu2
	t4::release(x1243);
	t4::tensor4f x1245 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1244, ctx.features_denseblock3_denselayer13_conv2_weight); //features.denseblock3.denselayer13.conv2
	t4::release(x1244);
	t4::tensor4f x1246 = t4::Concat<1>(x1239, x1245); //features.denseblock3.denselayer13
	t4::release(x1239, x1245);
	t4::tensor4f x1247 = t4::BatchNormalization(x1246, ctx.features_denseblock3_denselayer14_norm1_weight, ctx.features_denseblock3_denselayer14_norm1_bias, ctx.features_denseblock3_denselayer14_norm1_running_mean, ctx.features_denseblock3_denselayer14_norm1_running_var, 1e-05f); //features.denseblock3.denselayer14.norm1
	t4::tensor4f x1248 = t4::ReluInplace(x1247); //features.denseblock3.denselayer14.relu1
	t4::release(x1247);
	t4::tensor4f x1249 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1248, ctx.features_denseblock3_denselayer14_conv1_weight); //features.denseblock3.denselayer14.conv1
	t4::release(x1248);
	t4::tensor4f x1250 = t4::BatchNormalizationInplace(x1249, ctx.features_denseblock3_denselayer14_norm2_weight, ctx.features_denseblock3_denselayer14_norm2_bias, ctx.features_denseblock3_denselayer14_norm2_running_mean, ctx.features_denseblock3_denselayer14_norm2_running_var, 1e-05f); //features.denseblock3.denselayer14.norm2
	t4::release(x1249);
	t4::tensor4f x1251 = t4::ReluInplace(x1250); //features.denseblock3.denselayer14.relu2
	t4::release(x1250);
	t4::tensor4f x1252 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1251, ctx.features_denseblock3_denselayer14_conv2_weight); //features.denseblock3.denselayer14.conv2
	t4::release(x1251);
	t4::tensor4f x1253 = t4::Concat<1>(x1246, x1252); //features.denseblock3.denselayer14
	t4::release(x1246, x1252);
	t4::tensor4f x1254 = t4::BatchNormalization(x1253, ctx.features_denseblock3_denselayer15_norm1_weight, ctx.features_denseblock3_denselayer15_norm1_bias, ctx.features_denseblock3_denselayer15_norm1_running_mean, ctx.features_denseblock3_denselayer15_norm1_running_var, 1e-05f); //features.denseblock3.denselayer15.norm1
	t4::tensor4f x1255 = t4::ReluInplace(x1254); //features.denseblock3.denselayer15.relu1
	t4::release(x1254);
	t4::tensor4f x1256 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1255, ctx.features_denseblock3_denselayer15_conv1_weight); //features.denseblock3.denselayer15.conv1
	t4::release(x1255);
	t4::tensor4f x1257 = t4::BatchNormalizationInplace(x1256, ctx.features_denseblock3_denselayer15_norm2_weight, ctx.features_denseblock3_denselayer15_norm2_bias, ctx.features_denseblock3_denselayer15_norm2_running_mean, ctx.features_denseblock3_denselayer15_norm2_running_var, 1e-05f); //features.denseblock3.denselayer15.norm2
	t4::release(x1256);
	t4::tensor4f x1258 = t4::ReluInplace(x1257); //features.denseblock3.denselayer15.relu2
	t4::release(x1257);
	t4::tensor4f x1259 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1258, ctx.features_denseblock3_denselayer15_conv2_weight); //features.denseblock3.denselayer15.conv2
	t4::release(x1258);
	t4::tensor4f x1260 = t4::Concat<1>(x1253, x1259); //features.denseblock3.denselayer15
	t4::release(x1253, x1259);
	t4::tensor4f x1261 = t4::BatchNormalization(x1260, ctx.features_denseblock3_denselayer16_norm1_weight, ctx.features_denseblock3_denselayer16_norm1_bias, ctx.features_denseblock3_denselayer16_norm1_running_mean, ctx.features_denseblock3_denselayer16_norm1_running_var, 1e-05f); //features.denseblock3.denselayer16.norm1
	t4::tensor4f x1262 = t4::ReluInplace(x1261); //features.denseblock3.denselayer16.relu1
	t4::release(x1261);
	t4::tensor4f x1263 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1262, ctx.features_denseblock3_denselayer16_conv1_weight); //features.denseblock3.denselayer16.conv1
	t4::release(x1262);
	t4::tensor4f x1264 = t4::BatchNormalizationInplace(x1263, ctx.features_denseblock3_denselayer16_norm2_weight, ctx.features_denseblock3_denselayer16_norm2_bias, ctx.features_denseblock3_denselayer16_norm2_running_mean, ctx.features_denseblock3_denselayer16_norm2_running_var, 1e-05f); //features.denseblock3.denselayer16.norm2
	t4::release(x1263);
	t4::tensor4f x1265 = t4::ReluInplace(x1264); //features.denseblock3.denselayer16.relu2
	t4::release(x1264);
	t4::tensor4f x1266 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1265, ctx.features_denseblock3_denselayer16_conv2_weight); //features.denseblock3.denselayer16.conv2
	t4::release(x1265);
	t4::tensor4f x1267 = t4::Concat<1>(x1260, x1266); //features.denseblock3.denselayer16
	t4::release(x1260, x1266);
	t4::tensor4f x1268 = t4::BatchNormalization(x1267, ctx.features_denseblock3_denselayer17_norm1_weight, ctx.features_denseblock3_denselayer17_norm1_bias, ctx.features_denseblock3_denselayer17_norm1_running_mean, ctx.features_denseblock3_denselayer17_norm1_running_var, 1e-05f); //features.denseblock3.denselayer17.norm1
	t4::tensor4f x1269 = t4::ReluInplace(x1268); //features.denseblock3.denselayer17.relu1
	t4::release(x1268);
	t4::tensor4f x1270 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1269, ctx.features_denseblock3_denselayer17_conv1_weight); //features.denseblock3.denselayer17.conv1
	t4::release(x1269);
	t4::tensor4f x1271 = t4::BatchNormalizationInplace(x1270, ctx.features_denseblock3_denselayer17_norm2_weight, ctx.features_denseblock3_denselayer17_norm2_bias, ctx.features_denseblock3_denselayer17_norm2_running_mean, ctx.features_denseblock3_denselayer17_norm2_running_var, 1e-05f); //features.denseblock3.denselayer17.norm2
	t4::release(x1270);
	t4::tensor4f x1272 = t4::ReluInplace(x1271); //features.denseblock3.denselayer17.relu2
	t4::release(x1271);
	t4::tensor4f x1273 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1272, ctx.features_denseblock3_denselayer17_conv2_weight); //features.denseblock3.denselayer17.conv2
	t4::release(x1272);
	t4::tensor4f x1274 = t4::Concat<1>(x1267, x1273); //features.denseblock3.denselayer17
	t4::release(x1267, x1273);
	t4::tensor4f x1275 = t4::BatchNormalization(x1274, ctx.features_denseblock3_denselayer18_norm1_weight, ctx.features_denseblock3_denselayer18_norm1_bias, ctx.features_denseblock3_denselayer18_norm1_running_mean, ctx.features_denseblock3_denselayer18_norm1_running_var, 1e-05f); //features.denseblock3.denselayer18.norm1
	t4::tensor4f x1276 = t4::ReluInplace(x1275); //features.denseblock3.denselayer18.relu1
	t4::release(x1275);
	t4::tensor4f x1277 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1276, ctx.features_denseblock3_denselayer18_conv1_weight); //features.denseblock3.denselayer18.conv1
	t4::release(x1276);
	t4::tensor4f x1278 = t4::BatchNormalizationInplace(x1277, ctx.features_denseblock3_denselayer18_norm2_weight, ctx.features_denseblock3_denselayer18_norm2_bias, ctx.features_denseblock3_denselayer18_norm2_running_mean, ctx.features_denseblock3_denselayer18_norm2_running_var, 1e-05f); //features.denseblock3.denselayer18.norm2
	t4::release(x1277);
	t4::tensor4f x1279 = t4::ReluInplace(x1278); //features.denseblock3.denselayer18.relu2
	t4::release(x1278);
	t4::tensor4f x1280 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1279, ctx.features_denseblock3_denselayer18_conv2_weight); //features.denseblock3.denselayer18.conv2
	t4::release(x1279);
	t4::tensor4f x1281 = t4::Concat<1>(x1274, x1280); //features.denseblock3.denselayer18
	t4::release(x1274, x1280);
	t4::tensor4f x1282 = t4::BatchNormalization(x1281, ctx.features_denseblock3_denselayer19_norm1_weight, ctx.features_denseblock3_denselayer19_norm1_bias, ctx.features_denseblock3_denselayer19_norm1_running_mean, ctx.features_denseblock3_denselayer19_norm1_running_var, 1e-05f); //features.denseblock3.denselayer19.norm1
	t4::tensor4f x1283 = t4::ReluInplace(x1282); //features.denseblock3.denselayer19.relu1
	t4::release(x1282);
	t4::tensor4f x1284 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1283, ctx.features_denseblock3_denselayer19_conv1_weight); //features.denseblock3.denselayer19.conv1
	t4::release(x1283);
	t4::tensor4f x1285 = t4::BatchNormalizationInplace(x1284, ctx.features_denseblock3_denselayer19_norm2_weight, ctx.features_denseblock3_denselayer19_norm2_bias, ctx.features_denseblock3_denselayer19_norm2_running_mean, ctx.features_denseblock3_denselayer19_norm2_running_var, 1e-05f); //features.denseblock3.denselayer19.norm2
	t4::release(x1284);
	t4::tensor4f x1286 = t4::ReluInplace(x1285); //features.denseblock3.denselayer19.relu2
	t4::release(x1285);
	t4::tensor4f x1287 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1286, ctx.features_denseblock3_denselayer19_conv2_weight); //features.denseblock3.denselayer19.conv2
	t4::release(x1286);
	t4::tensor4f x1288 = t4::Concat<1>(x1281, x1287); //features.denseblock3.denselayer19
	t4::release(x1281, x1287);
	t4::tensor4f x1289 = t4::BatchNormalization(x1288, ctx.features_denseblock3_denselayer20_norm1_weight, ctx.features_denseblock3_denselayer20_norm1_bias, ctx.features_denseblock3_denselayer20_norm1_running_mean, ctx.features_denseblock3_denselayer20_norm1_running_var, 1e-05f); //features.denseblock3.denselayer20.norm1
	t4::tensor4f x1290 = t4::ReluInplace(x1289); //features.denseblock3.denselayer20.relu1
	t4::release(x1289);
	t4::tensor4f x1291 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1290, ctx.features_denseblock3_denselayer20_conv1_weight); //features.denseblock3.denselayer20.conv1
	t4::release(x1290);
	t4::tensor4f x1292 = t4::BatchNormalizationInplace(x1291, ctx.features_denseblock3_denselayer20_norm2_weight, ctx.features_denseblock3_denselayer20_norm2_bias, ctx.features_denseblock3_denselayer20_norm2_running_mean, ctx.features_denseblock3_denselayer20_norm2_running_var, 1e-05f); //features.denseblock3.denselayer20.norm2
	t4::release(x1291);
	t4::tensor4f x1293 = t4::ReluInplace(x1292); //features.denseblock3.denselayer20.relu2
	t4::release(x1292);
	t4::tensor4f x1294 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1293, ctx.features_denseblock3_denselayer20_conv2_weight); //features.denseblock3.denselayer20.conv2
	t4::release(x1293);
	t4::tensor4f x1295 = t4::Concat<1>(x1288, x1294); //features.denseblock3.denselayer20
	t4::release(x1288, x1294);
	t4::tensor4f x1296 = t4::BatchNormalization(x1295, ctx.features_denseblock3_denselayer21_norm1_weight, ctx.features_denseblock3_denselayer21_norm1_bias, ctx.features_denseblock3_denselayer21_norm1_running_mean, ctx.features_denseblock3_denselayer21_norm1_running_var, 1e-05f); //features.denseblock3.denselayer21.norm1
	t4::tensor4f x1297 = t4::ReluInplace(x1296); //features.denseblock3.denselayer21.relu1
	t4::release(x1296);
	t4::tensor4f x1298 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1297, ctx.features_denseblock3_denselayer21_conv1_weight); //features.denseblock3.denselayer21.conv1
	t4::release(x1297);
	t4::tensor4f x1299 = t4::BatchNormalizationInplace(x1298, ctx.features_denseblock3_denselayer21_norm2_weight, ctx.features_denseblock3_denselayer21_norm2_bias, ctx.features_denseblock3_denselayer21_norm2_running_mean, ctx.features_denseblock3_denselayer21_norm2_running_var, 1e-05f); //features.denseblock3.denselayer21.norm2
	t4::release(x1298);
	t4::tensor4f x1300 = t4::ReluInplace(x1299); //features.denseblock3.denselayer21.relu2
	t4::release(x1299);
	t4::tensor4f x1301 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1300, ctx.features_denseblock3_denselayer21_conv2_weight); //features.denseblock3.denselayer21.conv2
	t4::release(x1300);
	t4::tensor4f x1302 = t4::Concat<1>(x1295, x1301); //features.denseblock3.denselayer21
	t4::release(x1295, x1301);
	t4::tensor4f x1303 = t4::BatchNormalization(x1302, ctx.features_denseblock3_denselayer22_norm1_weight, ctx.features_denseblock3_denselayer22_norm1_bias, ctx.features_denseblock3_denselayer22_norm1_running_mean, ctx.features_denseblock3_denselayer22_norm1_running_var, 1e-05f); //features.denseblock3.denselayer22.norm1
	t4::tensor4f x1304 = t4::ReluInplace(x1303); //features.denseblock3.denselayer22.relu1
	t4::release(x1303);
	t4::tensor4f x1305 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1304, ctx.features_denseblock3_denselayer22_conv1_weight); //features.denseblock3.denselayer22.conv1
	t4::release(x1304);
	t4::tensor4f x1306 = t4::BatchNormalizationInplace(x1305, ctx.features_denseblock3_denselayer22_norm2_weight, ctx.features_denseblock3_denselayer22_norm2_bias, ctx.features_denseblock3_denselayer22_norm2_running_mean, ctx.features_denseblock3_denselayer22_norm2_running_var, 1e-05f); //features.denseblock3.denselayer22.norm2
	t4::release(x1305);
	t4::tensor4f x1307 = t4::ReluInplace(x1306); //features.denseblock3.denselayer22.relu2
	t4::release(x1306);
	t4::tensor4f x1308 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1307, ctx.features_denseblock3_denselayer22_conv2_weight); //features.denseblock3.denselayer22.conv2
	t4::release(x1307);
	t4::tensor4f x1309 = t4::Concat<1>(x1302, x1308); //features.denseblock3.denselayer22
	t4::release(x1302, x1308);
	t4::tensor4f x1310 = t4::BatchNormalization(x1309, ctx.features_denseblock3_denselayer23_norm1_weight, ctx.features_denseblock3_denselayer23_norm1_bias, ctx.features_denseblock3_denselayer23_norm1_running_mean, ctx.features_denseblock3_denselayer23_norm1_running_var, 1e-05f); //features.denseblock3.denselayer23.norm1
	t4::tensor4f x1311 = t4::ReluInplace(x1310); //features.denseblock3.denselayer23.relu1
	t4::release(x1310);
	t4::tensor4f x1312 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1311, ctx.features_denseblock3_denselayer23_conv1_weight); //features.denseblock3.denselayer23.conv1
	t4::release(x1311);
	t4::tensor4f x1313 = t4::BatchNormalizationInplace(x1312, ctx.features_denseblock3_denselayer23_norm2_weight, ctx.features_denseblock3_denselayer23_norm2_bias, ctx.features_denseblock3_denselayer23_norm2_running_mean, ctx.features_denseblock3_denselayer23_norm2_running_var, 1e-05f); //features.denseblock3.denselayer23.norm2
	t4::release(x1312);
	t4::tensor4f x1314 = t4::ReluInplace(x1313); //features.denseblock3.denselayer23.relu2
	t4::release(x1313);
	t4::tensor4f x1315 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1314, ctx.features_denseblock3_denselayer23_conv2_weight); //features.denseblock3.denselayer23.conv2
	t4::release(x1314);
	t4::tensor4f x1316 = t4::Concat<1>(x1309, x1315); //features.denseblock3.denselayer23
	t4::release(x1309, x1315);
	t4::tensor4f x1317 = t4::BatchNormalization(x1316, ctx.features_denseblock3_denselayer24_norm1_weight, ctx.features_denseblock3_denselayer24_norm1_bias, ctx.features_denseblock3_denselayer24_norm1_running_mean, ctx.features_denseblock3_denselayer24_norm1_running_var, 1e-05f); //features.denseblock3.denselayer24.norm1
	t4::tensor4f x1318 = t4::ReluInplace(x1317); //features.denseblock3.denselayer24.relu1
	t4::release(x1317);
	t4::tensor4f x1319 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1318, ctx.features_denseblock3_denselayer24_conv1_weight); //features.denseblock3.denselayer24.conv1
	t4::release(x1318);
	t4::tensor4f x1320 = t4::BatchNormalizationInplace(x1319, ctx.features_denseblock3_denselayer24_norm2_weight, ctx.features_denseblock3_denselayer24_norm2_bias, ctx.features_denseblock3_denselayer24_norm2_running_mean, ctx.features_denseblock3_denselayer24_norm2_running_var, 1e-05f); //features.denseblock3.denselayer24.norm2
	t4::release(x1319);
	t4::tensor4f x1321 = t4::ReluInplace(x1320); //features.denseblock3.denselayer24.relu2
	t4::release(x1320);
	t4::tensor4f x1322 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1321, ctx.features_denseblock3_denselayer24_conv2_weight); //features.denseblock3.denselayer24.conv2
	t4::release(x1321);
	t4::tensor4f x1323 = t4::Concat<1>(x1316, x1322); //features.denseblock3.denselayer24
	t4::release(x1316, x1322);
	t4::tensor4f x1324 = t4::BatchNormalization(x1323, ctx.features_denseblock3_denselayer25_norm1_weight, ctx.features_denseblock3_denselayer25_norm1_bias, ctx.features_denseblock3_denselayer25_norm1_running_mean, ctx.features_denseblock3_denselayer25_norm1_running_var, 1e-05f); //features.denseblock3.denselayer25.norm1
	t4::tensor4f x1325 = t4::ReluInplace(x1324); //features.denseblock3.denselayer25.relu1
	t4::release(x1324);
	t4::tensor4f x1326 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1325, ctx.features_denseblock3_denselayer25_conv1_weight); //features.denseblock3.denselayer25.conv1
	t4::release(x1325);
	t4::tensor4f x1327 = t4::BatchNormalizationInplace(x1326, ctx.features_denseblock3_denselayer25_norm2_weight, ctx.features_denseblock3_denselayer25_norm2_bias, ctx.features_denseblock3_denselayer25_norm2_running_mean, ctx.features_denseblock3_denselayer25_norm2_running_var, 1e-05f); //features.denseblock3.denselayer25.norm2
	t4::release(x1326);
	t4::tensor4f x1328 = t4::ReluInplace(x1327); //features.denseblock3.denselayer25.relu2
	t4::release(x1327);
	t4::tensor4f x1329 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1328, ctx.features_denseblock3_denselayer25_conv2_weight); //features.denseblock3.denselayer25.conv2
	t4::release(x1328);
	t4::tensor4f x1330 = t4::Concat<1>(x1323, x1329); //features.denseblock3.denselayer25
	t4::release(x1323, x1329);
	t4::tensor4f x1331 = t4::BatchNormalization(x1330, ctx.features_denseblock3_denselayer26_norm1_weight, ctx.features_denseblock3_denselayer26_norm1_bias, ctx.features_denseblock3_denselayer26_norm1_running_mean, ctx.features_denseblock3_denselayer26_norm1_running_var, 1e-05f); //features.denseblock3.denselayer26.norm1
	t4::tensor4f x1332 = t4::ReluInplace(x1331); //features.denseblock3.denselayer26.relu1
	t4::release(x1331);
	t4::tensor4f x1333 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1332, ctx.features_denseblock3_denselayer26_conv1_weight); //features.denseblock3.denselayer26.conv1
	t4::release(x1332);
	t4::tensor4f x1334 = t4::BatchNormalizationInplace(x1333, ctx.features_denseblock3_denselayer26_norm2_weight, ctx.features_denseblock3_denselayer26_norm2_bias, ctx.features_denseblock3_denselayer26_norm2_running_mean, ctx.features_denseblock3_denselayer26_norm2_running_var, 1e-05f); //features.denseblock3.denselayer26.norm2
	t4::release(x1333);
	t4::tensor4f x1335 = t4::ReluInplace(x1334); //features.denseblock3.denselayer26.relu2
	t4::release(x1334);
	t4::tensor4f x1336 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1335, ctx.features_denseblock3_denselayer26_conv2_weight); //features.denseblock3.denselayer26.conv2
	t4::release(x1335);
	t4::tensor4f x1337 = t4::Concat<1>(x1330, x1336); //features.denseblock3.denselayer26
	t4::release(x1330, x1336);
	t4::tensor4f x1338 = t4::BatchNormalization(x1337, ctx.features_denseblock3_denselayer27_norm1_weight, ctx.features_denseblock3_denselayer27_norm1_bias, ctx.features_denseblock3_denselayer27_norm1_running_mean, ctx.features_denseblock3_denselayer27_norm1_running_var, 1e-05f); //features.denseblock3.denselayer27.norm1
	t4::tensor4f x1339 = t4::ReluInplace(x1338); //features.denseblock3.denselayer27.relu1
	t4::release(x1338);
	t4::tensor4f x1340 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1339, ctx.features_denseblock3_denselayer27_conv1_weight); //features.denseblock3.denselayer27.conv1
	t4::release(x1339);
	t4::tensor4f x1341 = t4::BatchNormalizationInplace(x1340, ctx.features_denseblock3_denselayer27_norm2_weight, ctx.features_denseblock3_denselayer27_norm2_bias, ctx.features_denseblock3_denselayer27_norm2_running_mean, ctx.features_denseblock3_denselayer27_norm2_running_var, 1e-05f); //features.denseblock3.denselayer27.norm2
	t4::release(x1340);
	t4::tensor4f x1342 = t4::ReluInplace(x1341); //features.denseblock3.denselayer27.relu2
	t4::release(x1341);
	t4::tensor4f x1343 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1342, ctx.features_denseblock3_denselayer27_conv2_weight); //features.denseblock3.denselayer27.conv2
	t4::release(x1342);
	t4::tensor4f x1344 = t4::Concat<1>(x1337, x1343); //features.denseblock3.denselayer27
	t4::release(x1337, x1343);
	t4::tensor4f x1345 = t4::BatchNormalization(x1344, ctx.features_denseblock3_denselayer28_norm1_weight, ctx.features_denseblock3_denselayer28_norm1_bias, ctx.features_denseblock3_denselayer28_norm1_running_mean, ctx.features_denseblock3_denselayer28_norm1_running_var, 1e-05f); //features.denseblock3.denselayer28.norm1
	t4::tensor4f x1346 = t4::ReluInplace(x1345); //features.denseblock3.denselayer28.relu1
	t4::release(x1345);
	t4::tensor4f x1347 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1346, ctx.features_denseblock3_denselayer28_conv1_weight); //features.denseblock3.denselayer28.conv1
	t4::release(x1346);
	t4::tensor4f x1348 = t4::BatchNormalizationInplace(x1347, ctx.features_denseblock3_denselayer28_norm2_weight, ctx.features_denseblock3_denselayer28_norm2_bias, ctx.features_denseblock3_denselayer28_norm2_running_mean, ctx.features_denseblock3_denselayer28_norm2_running_var, 1e-05f); //features.denseblock3.denselayer28.norm2
	t4::release(x1347);
	t4::tensor4f x1349 = t4::ReluInplace(x1348); //features.denseblock3.denselayer28.relu2
	t4::release(x1348);
	t4::tensor4f x1350 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1349, ctx.features_denseblock3_denselayer28_conv2_weight); //features.denseblock3.denselayer28.conv2
	t4::release(x1349);
	t4::tensor4f x1351 = t4::Concat<1>(x1344, x1350); //features.denseblock3.denselayer28
	t4::release(x1344, x1350);
	t4::tensor4f x1352 = t4::BatchNormalization(x1351, ctx.features_denseblock3_denselayer29_norm1_weight, ctx.features_denseblock3_denselayer29_norm1_bias, ctx.features_denseblock3_denselayer29_norm1_running_mean, ctx.features_denseblock3_denselayer29_norm1_running_var, 1e-05f); //features.denseblock3.denselayer29.norm1
	t4::tensor4f x1353 = t4::ReluInplace(x1352); //features.denseblock3.denselayer29.relu1
	t4::release(x1352);
	t4::tensor4f x1354 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1353, ctx.features_denseblock3_denselayer29_conv1_weight); //features.denseblock3.denselayer29.conv1
	t4::release(x1353);
	t4::tensor4f x1355 = t4::BatchNormalizationInplace(x1354, ctx.features_denseblock3_denselayer29_norm2_weight, ctx.features_denseblock3_denselayer29_norm2_bias, ctx.features_denseblock3_denselayer29_norm2_running_mean, ctx.features_denseblock3_denselayer29_norm2_running_var, 1e-05f); //features.denseblock3.denselayer29.norm2
	t4::release(x1354);
	t4::tensor4f x1356 = t4::ReluInplace(x1355); //features.denseblock3.denselayer29.relu2
	t4::release(x1355);
	t4::tensor4f x1357 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1356, ctx.features_denseblock3_denselayer29_conv2_weight); //features.denseblock3.denselayer29.conv2
	t4::release(x1356);
	t4::tensor4f x1358 = t4::Concat<1>(x1351, x1357); //features.denseblock3.denselayer29
	t4::release(x1351, x1357);
	t4::tensor4f x1359 = t4::BatchNormalization(x1358, ctx.features_denseblock3_denselayer30_norm1_weight, ctx.features_denseblock3_denselayer30_norm1_bias, ctx.features_denseblock3_denselayer30_norm1_running_mean, ctx.features_denseblock3_denselayer30_norm1_running_var, 1e-05f); //features.denseblock3.denselayer30.norm1
	t4::tensor4f x1360 = t4::ReluInplace(x1359); //features.denseblock3.denselayer30.relu1
	t4::release(x1359);
	t4::tensor4f x1361 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1360, ctx.features_denseblock3_denselayer30_conv1_weight); //features.denseblock3.denselayer30.conv1
	t4::release(x1360);
	t4::tensor4f x1362 = t4::BatchNormalizationInplace(x1361, ctx.features_denseblock3_denselayer30_norm2_weight, ctx.features_denseblock3_denselayer30_norm2_bias, ctx.features_denseblock3_denselayer30_norm2_running_mean, ctx.features_denseblock3_denselayer30_norm2_running_var, 1e-05f); //features.denseblock3.denselayer30.norm2
	t4::release(x1361);
	t4::tensor4f x1363 = t4::ReluInplace(x1362); //features.denseblock3.denselayer30.relu2
	t4::release(x1362);
	t4::tensor4f x1364 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1363, ctx.features_denseblock3_denselayer30_conv2_weight); //features.denseblock3.denselayer30.conv2
	t4::release(x1363);
	t4::tensor4f x1365 = t4::Concat<1>(x1358, x1364); //features.denseblock3.denselayer30
	t4::release(x1358, x1364);
	t4::tensor4f x1366 = t4::BatchNormalization(x1365, ctx.features_denseblock3_denselayer31_norm1_weight, ctx.features_denseblock3_denselayer31_norm1_bias, ctx.features_denseblock3_denselayer31_norm1_running_mean, ctx.features_denseblock3_denselayer31_norm1_running_var, 1e-05f); //features.denseblock3.denselayer31.norm1
	t4::tensor4f x1367 = t4::ReluInplace(x1366); //features.denseblock3.denselayer31.relu1
	t4::release(x1366);
	t4::tensor4f x1368 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1367, ctx.features_denseblock3_denselayer31_conv1_weight); //features.denseblock3.denselayer31.conv1
	t4::release(x1367);
	t4::tensor4f x1369 = t4::BatchNormalizationInplace(x1368, ctx.features_denseblock3_denselayer31_norm2_weight, ctx.features_denseblock3_denselayer31_norm2_bias, ctx.features_denseblock3_denselayer31_norm2_running_mean, ctx.features_denseblock3_denselayer31_norm2_running_var, 1e-05f); //features.denseblock3.denselayer31.norm2
	t4::release(x1368);
	t4::tensor4f x1370 = t4::ReluInplace(x1369); //features.denseblock3.denselayer31.relu2
	t4::release(x1369);
	t4::tensor4f x1371 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1370, ctx.features_denseblock3_denselayer31_conv2_weight); //features.denseblock3.denselayer31.conv2
	t4::release(x1370);
	t4::tensor4f x1372 = t4::Concat<1>(x1365, x1371); //features.denseblock3.denselayer31
	t4::release(x1365, x1371);
	t4::tensor4f x1373 = t4::BatchNormalization(x1372, ctx.features_denseblock3_denselayer32_norm1_weight, ctx.features_denseblock3_denselayer32_norm1_bias, ctx.features_denseblock3_denselayer32_norm1_running_mean, ctx.features_denseblock3_denselayer32_norm1_running_var, 1e-05f); //features.denseblock3.denselayer32.norm1
	t4::tensor4f x1374 = t4::ReluInplace(x1373); //features.denseblock3.denselayer32.relu1
	t4::release(x1373);
	t4::tensor4f x1375 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1374, ctx.features_denseblock3_denselayer32_conv1_weight); //features.denseblock3.denselayer32.conv1
	t4::release(x1374);
	t4::tensor4f x1376 = t4::BatchNormalizationInplace(x1375, ctx.features_denseblock3_denselayer32_norm2_weight, ctx.features_denseblock3_denselayer32_norm2_bias, ctx.features_denseblock3_denselayer32_norm2_running_mean, ctx.features_denseblock3_denselayer32_norm2_running_var, 1e-05f); //features.denseblock3.denselayer32.norm2
	t4::release(x1375);
	t4::tensor4f x1377 = t4::ReluInplace(x1376); //features.denseblock3.denselayer32.relu2
	t4::release(x1376);
	t4::tensor4f x1378 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1377, ctx.features_denseblock3_denselayer32_conv2_weight); //features.denseblock3.denselayer32.conv2
	t4::release(x1377);
	t4::tensor4f x1379 = t4::Concat<1>(x1372, x1378); //features.denseblock3.denselayer32
	t4::release(x1372, x1378);
	t4::tensor4f x1380 = t4::BatchNormalizationInplace(x1379, ctx.features_transition3_norm_weight, ctx.features_transition3_norm_bias, ctx.features_transition3_norm_running_mean, ctx.features_transition3_norm_running_var, 1e-05f); //features.transition3.norm
	t4::release(x1379);
	t4::tensor4f x1381 = t4::ReluInplace(x1380); //features.transition3.relu
	t4::release(x1380);
	t4::tensor4f x1382 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1381, ctx.features_transition3_conv_weight); //features.transition3.conv
	t4::release(x1381);
	t4::tensor4f x1383 = t4::Pad<t4::constant>(x1382, 0, 0, 0, 0, 0, 0, 0, 0); //features.transition3.pool
	t4::release(x1382);
	t4::tensor4f x1384 = t4::AveragePool2d<2, 2, 2, 2, 0, 0>(x1383); //features.transition3.pool
	t4::release(x1383);
	t4::tensor4f x1385 = t4::BatchNormalization(x1384, ctx.features_denseblock4_denselayer1_norm1_weight, ctx.features_denseblock4_denselayer1_norm1_bias, ctx.features_denseblock4_denselayer1_norm1_running_mean, ctx.features_denseblock4_denselayer1_norm1_running_var, 1e-05f); //features.denseblock4.denselayer1.norm1
	t4::tensor4f x1386 = t4::ReluInplace(x1385); //features.denseblock4.denselayer1.relu1
	t4::release(x1385);
	t4::tensor4f x1387 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1386, ctx.features_denseblock4_denselayer1_conv1_weight); //features.denseblock4.denselayer1.conv1
	t4::release(x1386);
	t4::tensor4f x1388 = t4::BatchNormalizationInplace(x1387, ctx.features_denseblock4_denselayer1_norm2_weight, ctx.features_denseblock4_denselayer1_norm2_bias, ctx.features_denseblock4_denselayer1_norm2_running_mean, ctx.features_denseblock4_denselayer1_norm2_running_var, 1e-05f); //features.denseblock4.denselayer1.norm2
	t4::release(x1387);
	t4::tensor4f x1389 = t4::ReluInplace(x1388); //features.denseblock4.denselayer1.relu2
	t4::release(x1388);
	t4::tensor4f x1390 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1389, ctx.features_denseblock4_denselayer1_conv2_weight); //features.denseblock4.denselayer1.conv2
	t4::release(x1389);
	t4::tensor4f x1391 = t4::Concat<1>(x1384, x1390); //features.denseblock4.denselayer1
	t4::release(x1384, x1390);
	t4::tensor4f x1392 = t4::BatchNormalization(x1391, ctx.features_denseblock4_denselayer2_norm1_weight, ctx.features_denseblock4_denselayer2_norm1_bias, ctx.features_denseblock4_denselayer2_norm1_running_mean, ctx.features_denseblock4_denselayer2_norm1_running_var, 1e-05f); //features.denseblock4.denselayer2.norm1
	t4::tensor4f x1393 = t4::ReluInplace(x1392); //features.denseblock4.denselayer2.relu1
	t4::release(x1392);
	t4::tensor4f x1394 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1393, ctx.features_denseblock4_denselayer2_conv1_weight); //features.denseblock4.denselayer2.conv1
	t4::release(x1393);
	t4::tensor4f x1395 = t4::BatchNormalizationInplace(x1394, ctx.features_denseblock4_denselayer2_norm2_weight, ctx.features_denseblock4_denselayer2_norm2_bias, ctx.features_denseblock4_denselayer2_norm2_running_mean, ctx.features_denseblock4_denselayer2_norm2_running_var, 1e-05f); //features.denseblock4.denselayer2.norm2
	t4::release(x1394);
	t4::tensor4f x1396 = t4::ReluInplace(x1395); //features.denseblock4.denselayer2.relu2
	t4::release(x1395);
	t4::tensor4f x1397 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1396, ctx.features_denseblock4_denselayer2_conv2_weight); //features.denseblock4.denselayer2.conv2
	t4::release(x1396);
	t4::tensor4f x1398 = t4::Concat<1>(x1391, x1397); //features.denseblock4.denselayer2
	t4::release(x1391, x1397);
	t4::tensor4f x1399 = t4::BatchNormalization(x1398, ctx.features_denseblock4_denselayer3_norm1_weight, ctx.features_denseblock4_denselayer3_norm1_bias, ctx.features_denseblock4_denselayer3_norm1_running_mean, ctx.features_denseblock4_denselayer3_norm1_running_var, 1e-05f); //features.denseblock4.denselayer3.norm1
	t4::tensor4f x1400 = t4::ReluInplace(x1399); //features.denseblock4.denselayer3.relu1
	t4::release(x1399);
	t4::tensor4f x1401 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1400, ctx.features_denseblock4_denselayer3_conv1_weight); //features.denseblock4.denselayer3.conv1
	t4::release(x1400);
	t4::tensor4f x1402 = t4::BatchNormalizationInplace(x1401, ctx.features_denseblock4_denselayer3_norm2_weight, ctx.features_denseblock4_denselayer3_norm2_bias, ctx.features_denseblock4_denselayer3_norm2_running_mean, ctx.features_denseblock4_denselayer3_norm2_running_var, 1e-05f); //features.denseblock4.denselayer3.norm2
	t4::release(x1401);
	t4::tensor4f x1403 = t4::ReluInplace(x1402); //features.denseblock4.denselayer3.relu2
	t4::release(x1402);
	t4::tensor4f x1404 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1403, ctx.features_denseblock4_denselayer3_conv2_weight); //features.denseblock4.denselayer3.conv2
	t4::release(x1403);
	t4::tensor4f x1405 = t4::Concat<1>(x1398, x1404); //features.denseblock4.denselayer3
	t4::release(x1398, x1404);
	t4::tensor4f x1406 = t4::BatchNormalization(x1405, ctx.features_denseblock4_denselayer4_norm1_weight, ctx.features_denseblock4_denselayer4_norm1_bias, ctx.features_denseblock4_denselayer4_norm1_running_mean, ctx.features_denseblock4_denselayer4_norm1_running_var, 1e-05f); //features.denseblock4.denselayer4.norm1
	t4::tensor4f x1407 = t4::ReluInplace(x1406); //features.denseblock4.denselayer4.relu1
	t4::release(x1406);
	t4::tensor4f x1408 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1407, ctx.features_denseblock4_denselayer4_conv1_weight); //features.denseblock4.denselayer4.conv1
	t4::release(x1407);
	t4::tensor4f x1409 = t4::BatchNormalizationInplace(x1408, ctx.features_denseblock4_denselayer4_norm2_weight, ctx.features_denseblock4_denselayer4_norm2_bias, ctx.features_denseblock4_denselayer4_norm2_running_mean, ctx.features_denseblock4_denselayer4_norm2_running_var, 1e-05f); //features.denseblock4.denselayer4.norm2
	t4::release(x1408);
	t4::tensor4f x1410 = t4::ReluInplace(x1409); //features.denseblock4.denselayer4.relu2
	t4::release(x1409);
	t4::tensor4f x1411 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1410, ctx.features_denseblock4_denselayer4_conv2_weight); //features.denseblock4.denselayer4.conv2
	t4::release(x1410);
	t4::tensor4f x1412 = t4::Concat<1>(x1405, x1411); //features.denseblock4.denselayer4
	t4::release(x1405, x1411);
	t4::tensor4f x1413 = t4::BatchNormalization(x1412, ctx.features_denseblock4_denselayer5_norm1_weight, ctx.features_denseblock4_denselayer5_norm1_bias, ctx.features_denseblock4_denselayer5_norm1_running_mean, ctx.features_denseblock4_denselayer5_norm1_running_var, 1e-05f); //features.denseblock4.denselayer5.norm1
	t4::tensor4f x1414 = t4::ReluInplace(x1413); //features.denseblock4.denselayer5.relu1
	t4::release(x1413);
	t4::tensor4f x1415 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1414, ctx.features_denseblock4_denselayer5_conv1_weight); //features.denseblock4.denselayer5.conv1
	t4::release(x1414);
	t4::tensor4f x1416 = t4::BatchNormalizationInplace(x1415, ctx.features_denseblock4_denselayer5_norm2_weight, ctx.features_denseblock4_denselayer5_norm2_bias, ctx.features_denseblock4_denselayer5_norm2_running_mean, ctx.features_denseblock4_denselayer5_norm2_running_var, 1e-05f); //features.denseblock4.denselayer5.norm2
	t4::release(x1415);
	t4::tensor4f x1417 = t4::ReluInplace(x1416); //features.denseblock4.denselayer5.relu2
	t4::release(x1416);
	t4::tensor4f x1418 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1417, ctx.features_denseblock4_denselayer5_conv2_weight); //features.denseblock4.denselayer5.conv2
	t4::release(x1417);
	t4::tensor4f x1419 = t4::Concat<1>(x1412, x1418); //features.denseblock4.denselayer5
	t4::release(x1412, x1418);
	t4::tensor4f x1420 = t4::BatchNormalization(x1419, ctx.features_denseblock4_denselayer6_norm1_weight, ctx.features_denseblock4_denselayer6_norm1_bias, ctx.features_denseblock4_denselayer6_norm1_running_mean, ctx.features_denseblock4_denselayer6_norm1_running_var, 1e-05f); //features.denseblock4.denselayer6.norm1
	t4::tensor4f x1421 = t4::ReluInplace(x1420); //features.denseblock4.denselayer6.relu1
	t4::release(x1420);
	t4::tensor4f x1422 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1421, ctx.features_denseblock4_denselayer6_conv1_weight); //features.denseblock4.denselayer6.conv1
	t4::release(x1421);
	t4::tensor4f x1423 = t4::BatchNormalizationInplace(x1422, ctx.features_denseblock4_denselayer6_norm2_weight, ctx.features_denseblock4_denselayer6_norm2_bias, ctx.features_denseblock4_denselayer6_norm2_running_mean, ctx.features_denseblock4_denselayer6_norm2_running_var, 1e-05f); //features.denseblock4.denselayer6.norm2
	t4::release(x1422);
	t4::tensor4f x1424 = t4::ReluInplace(x1423); //features.denseblock4.denselayer6.relu2
	t4::release(x1423);
	t4::tensor4f x1425 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1424, ctx.features_denseblock4_denselayer6_conv2_weight); //features.denseblock4.denselayer6.conv2
	t4::release(x1424);
	t4::tensor4f x1426 = t4::Concat<1>(x1419, x1425); //features.denseblock4.denselayer6
	t4::release(x1419, x1425);
	t4::tensor4f x1427 = t4::BatchNormalization(x1426, ctx.features_denseblock4_denselayer7_norm1_weight, ctx.features_denseblock4_denselayer7_norm1_bias, ctx.features_denseblock4_denselayer7_norm1_running_mean, ctx.features_denseblock4_denselayer7_norm1_running_var, 1e-05f); //features.denseblock4.denselayer7.norm1
	t4::tensor4f x1428 = t4::ReluInplace(x1427); //features.denseblock4.denselayer7.relu1
	t4::release(x1427);
	t4::tensor4f x1429 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1428, ctx.features_denseblock4_denselayer7_conv1_weight); //features.denseblock4.denselayer7.conv1
	t4::release(x1428);
	t4::tensor4f x1430 = t4::BatchNormalizationInplace(x1429, ctx.features_denseblock4_denselayer7_norm2_weight, ctx.features_denseblock4_denselayer7_norm2_bias, ctx.features_denseblock4_denselayer7_norm2_running_mean, ctx.features_denseblock4_denselayer7_norm2_running_var, 1e-05f); //features.denseblock4.denselayer7.norm2
	t4::release(x1429);
	t4::tensor4f x1431 = t4::ReluInplace(x1430); //features.denseblock4.denselayer7.relu2
	t4::release(x1430);
	t4::tensor4f x1432 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1431, ctx.features_denseblock4_denselayer7_conv2_weight); //features.denseblock4.denselayer7.conv2
	t4::release(x1431);
	t4::tensor4f x1433 = t4::Concat<1>(x1426, x1432); //features.denseblock4.denselayer7
	t4::release(x1426, x1432);
	t4::tensor4f x1434 = t4::BatchNormalization(x1433, ctx.features_denseblock4_denselayer8_norm1_weight, ctx.features_denseblock4_denselayer8_norm1_bias, ctx.features_denseblock4_denselayer8_norm1_running_mean, ctx.features_denseblock4_denselayer8_norm1_running_var, 1e-05f); //features.denseblock4.denselayer8.norm1
	t4::tensor4f x1435 = t4::ReluInplace(x1434); //features.denseblock4.denselayer8.relu1
	t4::release(x1434);
	t4::tensor4f x1436 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1435, ctx.features_denseblock4_denselayer8_conv1_weight); //features.denseblock4.denselayer8.conv1
	t4::release(x1435);
	t4::tensor4f x1437 = t4::BatchNormalizationInplace(x1436, ctx.features_denseblock4_denselayer8_norm2_weight, ctx.features_denseblock4_denselayer8_norm2_bias, ctx.features_denseblock4_denselayer8_norm2_running_mean, ctx.features_denseblock4_denselayer8_norm2_running_var, 1e-05f); //features.denseblock4.denselayer8.norm2
	t4::release(x1436);
	t4::tensor4f x1438 = t4::ReluInplace(x1437); //features.denseblock4.denselayer8.relu2
	t4::release(x1437);
	t4::tensor4f x1439 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1438, ctx.features_denseblock4_denselayer8_conv2_weight); //features.denseblock4.denselayer8.conv2
	t4::release(x1438);
	t4::tensor4f x1440 = t4::Concat<1>(x1433, x1439); //features.denseblock4.denselayer8
	t4::release(x1433, x1439);
	t4::tensor4f x1441 = t4::BatchNormalization(x1440, ctx.features_denseblock4_denselayer9_norm1_weight, ctx.features_denseblock4_denselayer9_norm1_bias, ctx.features_denseblock4_denselayer9_norm1_running_mean, ctx.features_denseblock4_denselayer9_norm1_running_var, 1e-05f); //features.denseblock4.denselayer9.norm1
	t4::tensor4f x1442 = t4::ReluInplace(x1441); //features.denseblock4.denselayer9.relu1
	t4::release(x1441);
	t4::tensor4f x1443 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1442, ctx.features_denseblock4_denselayer9_conv1_weight); //features.denseblock4.denselayer9.conv1
	t4::release(x1442);
	t4::tensor4f x1444 = t4::BatchNormalizationInplace(x1443, ctx.features_denseblock4_denselayer9_norm2_weight, ctx.features_denseblock4_denselayer9_norm2_bias, ctx.features_denseblock4_denselayer9_norm2_running_mean, ctx.features_denseblock4_denselayer9_norm2_running_var, 1e-05f); //features.denseblock4.denselayer9.norm2
	t4::release(x1443);
	t4::tensor4f x1445 = t4::ReluInplace(x1444); //features.denseblock4.denselayer9.relu2
	t4::release(x1444);
	t4::tensor4f x1446 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1445, ctx.features_denseblock4_denselayer9_conv2_weight); //features.denseblock4.denselayer9.conv2
	t4::release(x1445);
	t4::tensor4f x1447 = t4::Concat<1>(x1440, x1446); //features.denseblock4.denselayer9
	t4::release(x1440, x1446);
	t4::tensor4f x1448 = t4::BatchNormalization(x1447, ctx.features_denseblock4_denselayer10_norm1_weight, ctx.features_denseblock4_denselayer10_norm1_bias, ctx.features_denseblock4_denselayer10_norm1_running_mean, ctx.features_denseblock4_denselayer10_norm1_running_var, 1e-05f); //features.denseblock4.denselayer10.norm1
	t4::tensor4f x1449 = t4::ReluInplace(x1448); //features.denseblock4.denselayer10.relu1
	t4::release(x1448);
	t4::tensor4f x1450 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1449, ctx.features_denseblock4_denselayer10_conv1_weight); //features.denseblock4.denselayer10.conv1
	t4::release(x1449);
	t4::tensor4f x1451 = t4::BatchNormalizationInplace(x1450, ctx.features_denseblock4_denselayer10_norm2_weight, ctx.features_denseblock4_denselayer10_norm2_bias, ctx.features_denseblock4_denselayer10_norm2_running_mean, ctx.features_denseblock4_denselayer10_norm2_running_var, 1e-05f); //features.denseblock4.denselayer10.norm2
	t4::release(x1450);
	t4::tensor4f x1452 = t4::ReluInplace(x1451); //features.denseblock4.denselayer10.relu2
	t4::release(x1451);
	t4::tensor4f x1453 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1452, ctx.features_denseblock4_denselayer10_conv2_weight); //features.denseblock4.denselayer10.conv2
	t4::release(x1452);
	t4::tensor4f x1454 = t4::Concat<1>(x1447, x1453); //features.denseblock4.denselayer10
	t4::release(x1447, x1453);
	t4::tensor4f x1455 = t4::BatchNormalization(x1454, ctx.features_denseblock4_denselayer11_norm1_weight, ctx.features_denseblock4_denselayer11_norm1_bias, ctx.features_denseblock4_denselayer11_norm1_running_mean, ctx.features_denseblock4_denselayer11_norm1_running_var, 1e-05f); //features.denseblock4.denselayer11.norm1
	t4::tensor4f x1456 = t4::ReluInplace(x1455); //features.denseblock4.denselayer11.relu1
	t4::release(x1455);
	t4::tensor4f x1457 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1456, ctx.features_denseblock4_denselayer11_conv1_weight); //features.denseblock4.denselayer11.conv1
	t4::release(x1456);
	t4::tensor4f x1458 = t4::BatchNormalizationInplace(x1457, ctx.features_denseblock4_denselayer11_norm2_weight, ctx.features_denseblock4_denselayer11_norm2_bias, ctx.features_denseblock4_denselayer11_norm2_running_mean, ctx.features_denseblock4_denselayer11_norm2_running_var, 1e-05f); //features.denseblock4.denselayer11.norm2
	t4::release(x1457);
	t4::tensor4f x1459 = t4::ReluInplace(x1458); //features.denseblock4.denselayer11.relu2
	t4::release(x1458);
	t4::tensor4f x1460 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1459, ctx.features_denseblock4_denselayer11_conv2_weight); //features.denseblock4.denselayer11.conv2
	t4::release(x1459);
	t4::tensor4f x1461 = t4::Concat<1>(x1454, x1460); //features.denseblock4.denselayer11
	t4::release(x1454, x1460);
	t4::tensor4f x1462 = t4::BatchNormalization(x1461, ctx.features_denseblock4_denselayer12_norm1_weight, ctx.features_denseblock4_denselayer12_norm1_bias, ctx.features_denseblock4_denselayer12_norm1_running_mean, ctx.features_denseblock4_denselayer12_norm1_running_var, 1e-05f); //features.denseblock4.denselayer12.norm1
	t4::tensor4f x1463 = t4::ReluInplace(x1462); //features.denseblock4.denselayer12.relu1
	t4::release(x1462);
	t4::tensor4f x1464 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1463, ctx.features_denseblock4_denselayer12_conv1_weight); //features.denseblock4.denselayer12.conv1
	t4::release(x1463);
	t4::tensor4f x1465 = t4::BatchNormalizationInplace(x1464, ctx.features_denseblock4_denselayer12_norm2_weight, ctx.features_denseblock4_denselayer12_norm2_bias, ctx.features_denseblock4_denselayer12_norm2_running_mean, ctx.features_denseblock4_denselayer12_norm2_running_var, 1e-05f); //features.denseblock4.denselayer12.norm2
	t4::release(x1464);
	t4::tensor4f x1466 = t4::ReluInplace(x1465); //features.denseblock4.denselayer12.relu2
	t4::release(x1465);
	t4::tensor4f x1467 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1466, ctx.features_denseblock4_denselayer12_conv2_weight); //features.denseblock4.denselayer12.conv2
	t4::release(x1466);
	t4::tensor4f x1468 = t4::Concat<1>(x1461, x1467); //features.denseblock4.denselayer12
	t4::release(x1461, x1467);
	t4::tensor4f x1469 = t4::BatchNormalization(x1468, ctx.features_denseblock4_denselayer13_norm1_weight, ctx.features_denseblock4_denselayer13_norm1_bias, ctx.features_denseblock4_denselayer13_norm1_running_mean, ctx.features_denseblock4_denselayer13_norm1_running_var, 1e-05f); //features.denseblock4.denselayer13.norm1
	t4::tensor4f x1470 = t4::ReluInplace(x1469); //features.denseblock4.denselayer13.relu1
	t4::release(x1469);
	t4::tensor4f x1471 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1470, ctx.features_denseblock4_denselayer13_conv1_weight); //features.denseblock4.denselayer13.conv1
	t4::release(x1470);
	t4::tensor4f x1472 = t4::BatchNormalizationInplace(x1471, ctx.features_denseblock4_denselayer13_norm2_weight, ctx.features_denseblock4_denselayer13_norm2_bias, ctx.features_denseblock4_denselayer13_norm2_running_mean, ctx.features_denseblock4_denselayer13_norm2_running_var, 1e-05f); //features.denseblock4.denselayer13.norm2
	t4::release(x1471);
	t4::tensor4f x1473 = t4::ReluInplace(x1472); //features.denseblock4.denselayer13.relu2
	t4::release(x1472);
	t4::tensor4f x1474 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1473, ctx.features_denseblock4_denselayer13_conv2_weight); //features.denseblock4.denselayer13.conv2
	t4::release(x1473);
	t4::tensor4f x1475 = t4::Concat<1>(x1468, x1474); //features.denseblock4.denselayer13
	t4::release(x1468, x1474);
	t4::tensor4f x1476 = t4::BatchNormalization(x1475, ctx.features_denseblock4_denselayer14_norm1_weight, ctx.features_denseblock4_denselayer14_norm1_bias, ctx.features_denseblock4_denselayer14_norm1_running_mean, ctx.features_denseblock4_denselayer14_norm1_running_var, 1e-05f); //features.denseblock4.denselayer14.norm1
	t4::tensor4f x1477 = t4::ReluInplace(x1476); //features.denseblock4.denselayer14.relu1
	t4::release(x1476);
	t4::tensor4f x1478 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1477, ctx.features_denseblock4_denselayer14_conv1_weight); //features.denseblock4.denselayer14.conv1
	t4::release(x1477);
	t4::tensor4f x1479 = t4::BatchNormalizationInplace(x1478, ctx.features_denseblock4_denselayer14_norm2_weight, ctx.features_denseblock4_denselayer14_norm2_bias, ctx.features_denseblock4_denselayer14_norm2_running_mean, ctx.features_denseblock4_denselayer14_norm2_running_var, 1e-05f); //features.denseblock4.denselayer14.norm2
	t4::release(x1478);
	t4::tensor4f x1480 = t4::ReluInplace(x1479); //features.denseblock4.denselayer14.relu2
	t4::release(x1479);
	t4::tensor4f x1481 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1480, ctx.features_denseblock4_denselayer14_conv2_weight); //features.denseblock4.denselayer14.conv2
	t4::release(x1480);
	t4::tensor4f x1482 = t4::Concat<1>(x1475, x1481); //features.denseblock4.denselayer14
	t4::release(x1475, x1481);
	t4::tensor4f x1483 = t4::BatchNormalization(x1482, ctx.features_denseblock4_denselayer15_norm1_weight, ctx.features_denseblock4_denselayer15_norm1_bias, ctx.features_denseblock4_denselayer15_norm1_running_mean, ctx.features_denseblock4_denselayer15_norm1_running_var, 1e-05f); //features.denseblock4.denselayer15.norm1
	t4::tensor4f x1484 = t4::ReluInplace(x1483); //features.denseblock4.denselayer15.relu1
	t4::release(x1483);
	t4::tensor4f x1485 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1484, ctx.features_denseblock4_denselayer15_conv1_weight); //features.denseblock4.denselayer15.conv1
	t4::release(x1484);
	t4::tensor4f x1486 = t4::BatchNormalizationInplace(x1485, ctx.features_denseblock4_denselayer15_norm2_weight, ctx.features_denseblock4_denselayer15_norm2_bias, ctx.features_denseblock4_denselayer15_norm2_running_mean, ctx.features_denseblock4_denselayer15_norm2_running_var, 1e-05f); //features.denseblock4.denselayer15.norm2
	t4::release(x1485);
	t4::tensor4f x1487 = t4::ReluInplace(x1486); //features.denseblock4.denselayer15.relu2
	t4::release(x1486);
	t4::tensor4f x1488 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1487, ctx.features_denseblock4_denselayer15_conv2_weight); //features.denseblock4.denselayer15.conv2
	t4::release(x1487);
	t4::tensor4f x1489 = t4::Concat<1>(x1482, x1488); //features.denseblock4.denselayer15
	t4::release(x1482, x1488);
	t4::tensor4f x1490 = t4::BatchNormalization(x1489, ctx.features_denseblock4_denselayer16_norm1_weight, ctx.features_denseblock4_denselayer16_norm1_bias, ctx.features_denseblock4_denselayer16_norm1_running_mean, ctx.features_denseblock4_denselayer16_norm1_running_var, 1e-05f); //features.denseblock4.denselayer16.norm1
	t4::tensor4f x1491 = t4::ReluInplace(x1490); //features.denseblock4.denselayer16.relu1
	t4::release(x1490);
	t4::tensor4f x1492 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1491, ctx.features_denseblock4_denselayer16_conv1_weight); //features.denseblock4.denselayer16.conv1
	t4::release(x1491);
	t4::tensor4f x1493 = t4::BatchNormalizationInplace(x1492, ctx.features_denseblock4_denselayer16_norm2_weight, ctx.features_denseblock4_denselayer16_norm2_bias, ctx.features_denseblock4_denselayer16_norm2_running_mean, ctx.features_denseblock4_denselayer16_norm2_running_var, 1e-05f); //features.denseblock4.denselayer16.norm2
	t4::release(x1492);
	t4::tensor4f x1494 = t4::ReluInplace(x1493); //features.denseblock4.denselayer16.relu2
	t4::release(x1493);
	t4::tensor4f x1495 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1494, ctx.features_denseblock4_denselayer16_conv2_weight); //features.denseblock4.denselayer16.conv2
	t4::release(x1494);
	t4::tensor4f x1496 = t4::Concat<1>(x1489, x1495); //features.denseblock4.denselayer16
	t4::release(x1489, x1495);
	t4::tensor4f x1497 = t4::BatchNormalization(x1496, ctx.features_denseblock4_denselayer17_norm1_weight, ctx.features_denseblock4_denselayer17_norm1_bias, ctx.features_denseblock4_denselayer17_norm1_running_mean, ctx.features_denseblock4_denselayer17_norm1_running_var, 1e-05f); //features.denseblock4.denselayer17.norm1
	t4::tensor4f x1498 = t4::ReluInplace(x1497); //features.denseblock4.denselayer17.relu1
	t4::release(x1497);
	t4::tensor4f x1499 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1498, ctx.features_denseblock4_denselayer17_conv1_weight); //features.denseblock4.denselayer17.conv1
	t4::release(x1498);
	t4::tensor4f x1500 = t4::BatchNormalizationInplace(x1499, ctx.features_denseblock4_denselayer17_norm2_weight, ctx.features_denseblock4_denselayer17_norm2_bias, ctx.features_denseblock4_denselayer17_norm2_running_mean, ctx.features_denseblock4_denselayer17_norm2_running_var, 1e-05f); //features.denseblock4.denselayer17.norm2
	t4::release(x1499);
	t4::tensor4f x1501 = t4::ReluInplace(x1500); //features.denseblock4.denselayer17.relu2
	t4::release(x1500);
	t4::tensor4f x1502 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1501, ctx.features_denseblock4_denselayer17_conv2_weight); //features.denseblock4.denselayer17.conv2
	t4::release(x1501);
	t4::tensor4f x1503 = t4::Concat<1>(x1496, x1502); //features.denseblock4.denselayer17
	t4::release(x1496, x1502);
	t4::tensor4f x1504 = t4::BatchNormalization(x1503, ctx.features_denseblock4_denselayer18_norm1_weight, ctx.features_denseblock4_denselayer18_norm1_bias, ctx.features_denseblock4_denselayer18_norm1_running_mean, ctx.features_denseblock4_denselayer18_norm1_running_var, 1e-05f); //features.denseblock4.denselayer18.norm1
	t4::tensor4f x1505 = t4::ReluInplace(x1504); //features.denseblock4.denselayer18.relu1
	t4::release(x1504);
	t4::tensor4f x1506 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1505, ctx.features_denseblock4_denselayer18_conv1_weight); //features.denseblock4.denselayer18.conv1
	t4::release(x1505);
	t4::tensor4f x1507 = t4::BatchNormalizationInplace(x1506, ctx.features_denseblock4_denselayer18_norm2_weight, ctx.features_denseblock4_denselayer18_norm2_bias, ctx.features_denseblock4_denselayer18_norm2_running_mean, ctx.features_denseblock4_denselayer18_norm2_running_var, 1e-05f); //features.denseblock4.denselayer18.norm2
	t4::release(x1506);
	t4::tensor4f x1508 = t4::ReluInplace(x1507); //features.denseblock4.denselayer18.relu2
	t4::release(x1507);
	t4::tensor4f x1509 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1508, ctx.features_denseblock4_denselayer18_conv2_weight); //features.denseblock4.denselayer18.conv2
	t4::release(x1508);
	t4::tensor4f x1510 = t4::Concat<1>(x1503, x1509); //features.denseblock4.denselayer18
	t4::release(x1503, x1509);
	t4::tensor4f x1511 = t4::BatchNormalization(x1510, ctx.features_denseblock4_denselayer19_norm1_weight, ctx.features_denseblock4_denselayer19_norm1_bias, ctx.features_denseblock4_denselayer19_norm1_running_mean, ctx.features_denseblock4_denselayer19_norm1_running_var, 1e-05f); //features.denseblock4.denselayer19.norm1
	t4::tensor4f x1512 = t4::ReluInplace(x1511); //features.denseblock4.denselayer19.relu1
	t4::release(x1511);
	t4::tensor4f x1513 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1512, ctx.features_denseblock4_denselayer19_conv1_weight); //features.denseblock4.denselayer19.conv1
	t4::release(x1512);
	t4::tensor4f x1514 = t4::BatchNormalizationInplace(x1513, ctx.features_denseblock4_denselayer19_norm2_weight, ctx.features_denseblock4_denselayer19_norm2_bias, ctx.features_denseblock4_denselayer19_norm2_running_mean, ctx.features_denseblock4_denselayer19_norm2_running_var, 1e-05f); //features.denseblock4.denselayer19.norm2
	t4::release(x1513);
	t4::tensor4f x1515 = t4::ReluInplace(x1514); //features.denseblock4.denselayer19.relu2
	t4::release(x1514);
	t4::tensor4f x1516 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1515, ctx.features_denseblock4_denselayer19_conv2_weight); //features.denseblock4.denselayer19.conv2
	t4::release(x1515);
	t4::tensor4f x1517 = t4::Concat<1>(x1510, x1516); //features.denseblock4.denselayer19
	t4::release(x1510, x1516);
	t4::tensor4f x1518 = t4::BatchNormalization(x1517, ctx.features_denseblock4_denselayer20_norm1_weight, ctx.features_denseblock4_denselayer20_norm1_bias, ctx.features_denseblock4_denselayer20_norm1_running_mean, ctx.features_denseblock4_denselayer20_norm1_running_var, 1e-05f); //features.denseblock4.denselayer20.norm1
	t4::tensor4f x1519 = t4::ReluInplace(x1518); //features.denseblock4.denselayer20.relu1
	t4::release(x1518);
	t4::tensor4f x1520 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1519, ctx.features_denseblock4_denselayer20_conv1_weight); //features.denseblock4.denselayer20.conv1
	t4::release(x1519);
	t4::tensor4f x1521 = t4::BatchNormalizationInplace(x1520, ctx.features_denseblock4_denselayer20_norm2_weight, ctx.features_denseblock4_denselayer20_norm2_bias, ctx.features_denseblock4_denselayer20_norm2_running_mean, ctx.features_denseblock4_denselayer20_norm2_running_var, 1e-05f); //features.denseblock4.denselayer20.norm2
	t4::release(x1520);
	t4::tensor4f x1522 = t4::ReluInplace(x1521); //features.denseblock4.denselayer20.relu2
	t4::release(x1521);
	t4::tensor4f x1523 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1522, ctx.features_denseblock4_denselayer20_conv2_weight); //features.denseblock4.denselayer20.conv2
	t4::release(x1522);
	t4::tensor4f x1524 = t4::Concat<1>(x1517, x1523); //features.denseblock4.denselayer20
	t4::release(x1517, x1523);
	t4::tensor4f x1525 = t4::BatchNormalization(x1524, ctx.features_denseblock4_denselayer21_norm1_weight, ctx.features_denseblock4_denselayer21_norm1_bias, ctx.features_denseblock4_denselayer21_norm1_running_mean, ctx.features_denseblock4_denselayer21_norm1_running_var, 1e-05f); //features.denseblock4.denselayer21.norm1
	t4::tensor4f x1526 = t4::ReluInplace(x1525); //features.denseblock4.denselayer21.relu1
	t4::release(x1525);
	t4::tensor4f x1527 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1526, ctx.features_denseblock4_denselayer21_conv1_weight); //features.denseblock4.denselayer21.conv1
	t4::release(x1526);
	t4::tensor4f x1528 = t4::BatchNormalizationInplace(x1527, ctx.features_denseblock4_denselayer21_norm2_weight, ctx.features_denseblock4_denselayer21_norm2_bias, ctx.features_denseblock4_denselayer21_norm2_running_mean, ctx.features_denseblock4_denselayer21_norm2_running_var, 1e-05f); //features.denseblock4.denselayer21.norm2
	t4::release(x1527);
	t4::tensor4f x1529 = t4::ReluInplace(x1528); //features.denseblock4.denselayer21.relu2
	t4::release(x1528);
	t4::tensor4f x1530 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1529, ctx.features_denseblock4_denselayer21_conv2_weight); //features.denseblock4.denselayer21.conv2
	t4::release(x1529);
	t4::tensor4f x1531 = t4::Concat<1>(x1524, x1530); //features.denseblock4.denselayer21
	t4::release(x1524, x1530);
	t4::tensor4f x1532 = t4::BatchNormalization(x1531, ctx.features_denseblock4_denselayer22_norm1_weight, ctx.features_denseblock4_denselayer22_norm1_bias, ctx.features_denseblock4_denselayer22_norm1_running_mean, ctx.features_denseblock4_denselayer22_norm1_running_var, 1e-05f); //features.denseblock4.denselayer22.norm1
	t4::tensor4f x1533 = t4::ReluInplace(x1532); //features.denseblock4.denselayer22.relu1
	t4::release(x1532);
	t4::tensor4f x1534 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1533, ctx.features_denseblock4_denselayer22_conv1_weight); //features.denseblock4.denselayer22.conv1
	t4::release(x1533);
	t4::tensor4f x1535 = t4::BatchNormalizationInplace(x1534, ctx.features_denseblock4_denselayer22_norm2_weight, ctx.features_denseblock4_denselayer22_norm2_bias, ctx.features_denseblock4_denselayer22_norm2_running_mean, ctx.features_denseblock4_denselayer22_norm2_running_var, 1e-05f); //features.denseblock4.denselayer22.norm2
	t4::release(x1534);
	t4::tensor4f x1536 = t4::ReluInplace(x1535); //features.denseblock4.denselayer22.relu2
	t4::release(x1535);
	t4::tensor4f x1537 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1536, ctx.features_denseblock4_denselayer22_conv2_weight); //features.denseblock4.denselayer22.conv2
	t4::release(x1536);
	t4::tensor4f x1538 = t4::Concat<1>(x1531, x1537); //features.denseblock4.denselayer22
	t4::release(x1531, x1537);
	t4::tensor4f x1539 = t4::BatchNormalization(x1538, ctx.features_denseblock4_denselayer23_norm1_weight, ctx.features_denseblock4_denselayer23_norm1_bias, ctx.features_denseblock4_denselayer23_norm1_running_mean, ctx.features_denseblock4_denselayer23_norm1_running_var, 1e-05f); //features.denseblock4.denselayer23.norm1
	t4::tensor4f x1540 = t4::ReluInplace(x1539); //features.denseblock4.denselayer23.relu1
	t4::release(x1539);
	t4::tensor4f x1541 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1540, ctx.features_denseblock4_denselayer23_conv1_weight); //features.denseblock4.denselayer23.conv1
	t4::release(x1540);
	t4::tensor4f x1542 = t4::BatchNormalizationInplace(x1541, ctx.features_denseblock4_denselayer23_norm2_weight, ctx.features_denseblock4_denselayer23_norm2_bias, ctx.features_denseblock4_denselayer23_norm2_running_mean, ctx.features_denseblock4_denselayer23_norm2_running_var, 1e-05f); //features.denseblock4.denselayer23.norm2
	t4::release(x1541);
	t4::tensor4f x1543 = t4::ReluInplace(x1542); //features.denseblock4.denselayer23.relu2
	t4::release(x1542);
	t4::tensor4f x1544 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1543, ctx.features_denseblock4_denselayer23_conv2_weight); //features.denseblock4.denselayer23.conv2
	t4::release(x1543);
	t4::tensor4f x1545 = t4::Concat<1>(x1538, x1544); //features.denseblock4.denselayer23
	t4::release(x1538, x1544);
	t4::tensor4f x1546 = t4::BatchNormalization(x1545, ctx.features_denseblock4_denselayer24_norm1_weight, ctx.features_denseblock4_denselayer24_norm1_bias, ctx.features_denseblock4_denselayer24_norm1_running_mean, ctx.features_denseblock4_denselayer24_norm1_running_var, 1e-05f); //features.denseblock4.denselayer24.norm1
	t4::tensor4f x1547 = t4::ReluInplace(x1546); //features.denseblock4.denselayer24.relu1
	t4::release(x1546);
	t4::tensor4f x1548 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1547, ctx.features_denseblock4_denselayer24_conv1_weight); //features.denseblock4.denselayer24.conv1
	t4::release(x1547);
	t4::tensor4f x1549 = t4::BatchNormalizationInplace(x1548, ctx.features_denseblock4_denselayer24_norm2_weight, ctx.features_denseblock4_denselayer24_norm2_bias, ctx.features_denseblock4_denselayer24_norm2_running_mean, ctx.features_denseblock4_denselayer24_norm2_running_var, 1e-05f); //features.denseblock4.denselayer24.norm2
	t4::release(x1548);
	t4::tensor4f x1550 = t4::ReluInplace(x1549); //features.denseblock4.denselayer24.relu2
	t4::release(x1549);
	t4::tensor4f x1551 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1550, ctx.features_denseblock4_denselayer24_conv2_weight); //features.denseblock4.denselayer24.conv2
	t4::release(x1550);
	t4::tensor4f x1552 = t4::Concat<1>(x1545, x1551); //features.denseblock4.denselayer24
	t4::release(x1545, x1551);
	t4::tensor4f x1553 = t4::BatchNormalization(x1552, ctx.features_denseblock4_denselayer25_norm1_weight, ctx.features_denseblock4_denselayer25_norm1_bias, ctx.features_denseblock4_denselayer25_norm1_running_mean, ctx.features_denseblock4_denselayer25_norm1_running_var, 1e-05f); //features.denseblock4.denselayer25.norm1
	t4::tensor4f x1554 = t4::ReluInplace(x1553); //features.denseblock4.denselayer25.relu1
	t4::release(x1553);
	t4::tensor4f x1555 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1554, ctx.features_denseblock4_denselayer25_conv1_weight); //features.denseblock4.denselayer25.conv1
	t4::release(x1554);
	t4::tensor4f x1556 = t4::BatchNormalizationInplace(x1555, ctx.features_denseblock4_denselayer25_norm2_weight, ctx.features_denseblock4_denselayer25_norm2_bias, ctx.features_denseblock4_denselayer25_norm2_running_mean, ctx.features_denseblock4_denselayer25_norm2_running_var, 1e-05f); //features.denseblock4.denselayer25.norm2
	t4::release(x1555);
	t4::tensor4f x1557 = t4::ReluInplace(x1556); //features.denseblock4.denselayer25.relu2
	t4::release(x1556);
	t4::tensor4f x1558 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1557, ctx.features_denseblock4_denselayer25_conv2_weight); //features.denseblock4.denselayer25.conv2
	t4::release(x1557);
	t4::tensor4f x1559 = t4::Concat<1>(x1552, x1558); //features.denseblock4.denselayer25
	t4::release(x1552, x1558);
	t4::tensor4f x1560 = t4::BatchNormalization(x1559, ctx.features_denseblock4_denselayer26_norm1_weight, ctx.features_denseblock4_denselayer26_norm1_bias, ctx.features_denseblock4_denselayer26_norm1_running_mean, ctx.features_denseblock4_denselayer26_norm1_running_var, 1e-05f); //features.denseblock4.denselayer26.norm1
	t4::tensor4f x1561 = t4::ReluInplace(x1560); //features.denseblock4.denselayer26.relu1
	t4::release(x1560);
	t4::tensor4f x1562 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1561, ctx.features_denseblock4_denselayer26_conv1_weight); //features.denseblock4.denselayer26.conv1
	t4::release(x1561);
	t4::tensor4f x1563 = t4::BatchNormalizationInplace(x1562, ctx.features_denseblock4_denselayer26_norm2_weight, ctx.features_denseblock4_denselayer26_norm2_bias, ctx.features_denseblock4_denselayer26_norm2_running_mean, ctx.features_denseblock4_denselayer26_norm2_running_var, 1e-05f); //features.denseblock4.denselayer26.norm2
	t4::release(x1562);
	t4::tensor4f x1564 = t4::ReluInplace(x1563); //features.denseblock4.denselayer26.relu2
	t4::release(x1563);
	t4::tensor4f x1565 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1564, ctx.features_denseblock4_denselayer26_conv2_weight); //features.denseblock4.denselayer26.conv2
	t4::release(x1564);
	t4::tensor4f x1566 = t4::Concat<1>(x1559, x1565); //features.denseblock4.denselayer26
	t4::release(x1559, x1565);
	t4::tensor4f x1567 = t4::BatchNormalization(x1566, ctx.features_denseblock4_denselayer27_norm1_weight, ctx.features_denseblock4_denselayer27_norm1_bias, ctx.features_denseblock4_denselayer27_norm1_running_mean, ctx.features_denseblock4_denselayer27_norm1_running_var, 1e-05f); //features.denseblock4.denselayer27.norm1
	t4::tensor4f x1568 = t4::ReluInplace(x1567); //features.denseblock4.denselayer27.relu1
	t4::release(x1567);
	t4::tensor4f x1569 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1568, ctx.features_denseblock4_denselayer27_conv1_weight); //features.denseblock4.denselayer27.conv1
	t4::release(x1568);
	t4::tensor4f x1570 = t4::BatchNormalizationInplace(x1569, ctx.features_denseblock4_denselayer27_norm2_weight, ctx.features_denseblock4_denselayer27_norm2_bias, ctx.features_denseblock4_denselayer27_norm2_running_mean, ctx.features_denseblock4_denselayer27_norm2_running_var, 1e-05f); //features.denseblock4.denselayer27.norm2
	t4::release(x1569);
	t4::tensor4f x1571 = t4::ReluInplace(x1570); //features.denseblock4.denselayer27.relu2
	t4::release(x1570);
	t4::tensor4f x1572 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1571, ctx.features_denseblock4_denselayer27_conv2_weight); //features.denseblock4.denselayer27.conv2
	t4::release(x1571);
	t4::tensor4f x1573 = t4::Concat<1>(x1566, x1572); //features.denseblock4.denselayer27
	t4::release(x1566, x1572);
	t4::tensor4f x1574 = t4::BatchNormalization(x1573, ctx.features_denseblock4_denselayer28_norm1_weight, ctx.features_denseblock4_denselayer28_norm1_bias, ctx.features_denseblock4_denselayer28_norm1_running_mean, ctx.features_denseblock4_denselayer28_norm1_running_var, 1e-05f); //features.denseblock4.denselayer28.norm1
	t4::tensor4f x1575 = t4::ReluInplace(x1574); //features.denseblock4.denselayer28.relu1
	t4::release(x1574);
	t4::tensor4f x1576 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1575, ctx.features_denseblock4_denselayer28_conv1_weight); //features.denseblock4.denselayer28.conv1
	t4::release(x1575);
	t4::tensor4f x1577 = t4::BatchNormalizationInplace(x1576, ctx.features_denseblock4_denselayer28_norm2_weight, ctx.features_denseblock4_denselayer28_norm2_bias, ctx.features_denseblock4_denselayer28_norm2_running_mean, ctx.features_denseblock4_denselayer28_norm2_running_var, 1e-05f); //features.denseblock4.denselayer28.norm2
	t4::release(x1576);
	t4::tensor4f x1578 = t4::ReluInplace(x1577); //features.denseblock4.denselayer28.relu2
	t4::release(x1577);
	t4::tensor4f x1579 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1578, ctx.features_denseblock4_denselayer28_conv2_weight); //features.denseblock4.denselayer28.conv2
	t4::release(x1578);
	t4::tensor4f x1580 = t4::Concat<1>(x1573, x1579); //features.denseblock4.denselayer28
	t4::release(x1573, x1579);
	t4::tensor4f x1581 = t4::BatchNormalization(x1580, ctx.features_denseblock4_denselayer29_norm1_weight, ctx.features_denseblock4_denselayer29_norm1_bias, ctx.features_denseblock4_denselayer29_norm1_running_mean, ctx.features_denseblock4_denselayer29_norm1_running_var, 1e-05f); //features.denseblock4.denselayer29.norm1
	t4::tensor4f x1582 = t4::ReluInplace(x1581); //features.denseblock4.denselayer29.relu1
	t4::release(x1581);
	t4::tensor4f x1583 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1582, ctx.features_denseblock4_denselayer29_conv1_weight); //features.denseblock4.denselayer29.conv1
	t4::release(x1582);
	t4::tensor4f x1584 = t4::BatchNormalizationInplace(x1583, ctx.features_denseblock4_denselayer29_norm2_weight, ctx.features_denseblock4_denselayer29_norm2_bias, ctx.features_denseblock4_denselayer29_norm2_running_mean, ctx.features_denseblock4_denselayer29_norm2_running_var, 1e-05f); //features.denseblock4.denselayer29.norm2
	t4::release(x1583);
	t4::tensor4f x1585 = t4::ReluInplace(x1584); //features.denseblock4.denselayer29.relu2
	t4::release(x1584);
	t4::tensor4f x1586 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1585, ctx.features_denseblock4_denselayer29_conv2_weight); //features.denseblock4.denselayer29.conv2
	t4::release(x1585);
	t4::tensor4f x1587 = t4::Concat<1>(x1580, x1586); //features.denseblock4.denselayer29
	t4::release(x1580, x1586);
	t4::tensor4f x1588 = t4::BatchNormalization(x1587, ctx.features_denseblock4_denselayer30_norm1_weight, ctx.features_denseblock4_denselayer30_norm1_bias, ctx.features_denseblock4_denselayer30_norm1_running_mean, ctx.features_denseblock4_denselayer30_norm1_running_var, 1e-05f); //features.denseblock4.denselayer30.norm1
	t4::tensor4f x1589 = t4::ReluInplace(x1588); //features.denseblock4.denselayer30.relu1
	t4::release(x1588);
	t4::tensor4f x1590 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1589, ctx.features_denseblock4_denselayer30_conv1_weight); //features.denseblock4.denselayer30.conv1
	t4::release(x1589);
	t4::tensor4f x1591 = t4::BatchNormalizationInplace(x1590, ctx.features_denseblock4_denselayer30_norm2_weight, ctx.features_denseblock4_denselayer30_norm2_bias, ctx.features_denseblock4_denselayer30_norm2_running_mean, ctx.features_denseblock4_denselayer30_norm2_running_var, 1e-05f); //features.denseblock4.denselayer30.norm2
	t4::release(x1590);
	t4::tensor4f x1592 = t4::ReluInplace(x1591); //features.denseblock4.denselayer30.relu2
	t4::release(x1591);
	t4::tensor4f x1593 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1592, ctx.features_denseblock4_denselayer30_conv2_weight); //features.denseblock4.denselayer30.conv2
	t4::release(x1592);
	t4::tensor4f x1594 = t4::Concat<1>(x1587, x1593); //features.denseblock4.denselayer30
	t4::release(x1587, x1593);
	t4::tensor4f x1595 = t4::BatchNormalization(x1594, ctx.features_denseblock4_denselayer31_norm1_weight, ctx.features_denseblock4_denselayer31_norm1_bias, ctx.features_denseblock4_denselayer31_norm1_running_mean, ctx.features_denseblock4_denselayer31_norm1_running_var, 1e-05f); //features.denseblock4.denselayer31.norm1
	t4::tensor4f x1596 = t4::ReluInplace(x1595); //features.denseblock4.denselayer31.relu1
	t4::release(x1595);
	t4::tensor4f x1597 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1596, ctx.features_denseblock4_denselayer31_conv1_weight); //features.denseblock4.denselayer31.conv1
	t4::release(x1596);
	t4::tensor4f x1598 = t4::BatchNormalizationInplace(x1597, ctx.features_denseblock4_denselayer31_norm2_weight, ctx.features_denseblock4_denselayer31_norm2_bias, ctx.features_denseblock4_denselayer31_norm2_running_mean, ctx.features_denseblock4_denselayer31_norm2_running_var, 1e-05f); //features.denseblock4.denselayer31.norm2
	t4::release(x1597);
	t4::tensor4f x1599 = t4::ReluInplace(x1598); //features.denseblock4.denselayer31.relu2
	t4::release(x1598);
	t4::tensor4f x1600 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1599, ctx.features_denseblock4_denselayer31_conv2_weight); //features.denseblock4.denselayer31.conv2
	t4::release(x1599);
	t4::tensor4f x1601 = t4::Concat<1>(x1594, x1600); //features.denseblock4.denselayer31
	t4::release(x1594, x1600);
	t4::tensor4f x1602 = t4::BatchNormalization(x1601, ctx.features_denseblock4_denselayer32_norm1_weight, ctx.features_denseblock4_denselayer32_norm1_bias, ctx.features_denseblock4_denselayer32_norm1_running_mean, ctx.features_denseblock4_denselayer32_norm1_running_var, 1e-05f); //features.denseblock4.denselayer32.norm1
	t4::tensor4f x1603 = t4::ReluInplace(x1602); //features.denseblock4.denselayer32.relu1
	t4::release(x1602);
	t4::tensor4f x1604 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x1603, ctx.features_denseblock4_denselayer32_conv1_weight); //features.denseblock4.denselayer32.conv1
	t4::release(x1603);
	t4::tensor4f x1605 = t4::BatchNormalizationInplace(x1604, ctx.features_denseblock4_denselayer32_norm2_weight, ctx.features_denseblock4_denselayer32_norm2_bias, ctx.features_denseblock4_denselayer32_norm2_running_mean, ctx.features_denseblock4_denselayer32_norm2_running_var, 1e-05f); //features.denseblock4.denselayer32.norm2
	t4::release(x1604);
	t4::tensor4f x1606 = t4::ReluInplace(x1605); //features.denseblock4.denselayer32.relu2
	t4::release(x1605);
	t4::tensor4f x1607 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x1606, ctx.features_denseblock4_denselayer32_conv2_weight); //features.denseblock4.denselayer32.conv2
	t4::release(x1606);
	t4::tensor4f x1608 = t4::Concat<1>(x1601, x1607); //features.denseblock4.denselayer32
	t4::release(x1601, x1607);
	t4::tensor4f x1609 = t4::BatchNormalizationInplace(x1608, ctx.features_norm5_weight, ctx.features_norm5_bias, ctx.features_norm5_running_mean, ctx.features_norm5_running_var, 1e-05f); //features.norm5
	t4::release(x1608);
	t4::tensor4f x1610 = t4::ReluInplace(x1609);
	t4::release(x1609);
	t4::tensor4f x1611 = t4::GlobalAveragePool2d(x1610);
	t4::tensor0i x1612 = t4::Constant<t4::int64>(0);
	t4::tensor1i x1613 = t4::Shape(x1610);
	t4::release(x1610);
	t4::tensor0i x1614 = t4::Gather(x1613, x1612);
	t4::release(x1613, x1612);
	t4::tensor0i x1615 = t4::Constant<t4::int64>(-1);
	t4::tensor1i x1616 = t4::Unsqueeze<0>(x1614);
	t4::release(x1614);
	t4::tensor1i x1617 = t4::Unsqueeze<0>(x1615);
	t4::release(x1615);
	t4::tensor1i x1618 = t4::Concat<0>(x1616, x1617);
	t4::release(x1616, x1617);
	t4::tensor2f x1619 = t4::Reshape<2>(x1611, x1618);
	t4::release(x1611, x1618);
	t4::tensor2f x1620 = t4::Linear(x1619, ctx.classifier_weight, ctx.classifier_bias); //classifier
	t4::release(x1619);
	return x1620;
}
