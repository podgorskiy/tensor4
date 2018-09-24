#include "ResNet.h"


ResNet ResNetLoad(const char* filename)
{
	ResNet ctx;
	t4::model_dict dict = t4::load(filename);
	dict.load(ctx.conv1_weight, "conv1.weight", 64, 3, 7, 7);
	dict.load(ctx.bn1_weight, "bn1.weight", 64);
	dict.load(ctx.bn1_bias, "bn1.bias", 64);
	dict.load(ctx.bn1_running_mean, "bn1.running_mean", 64);
	dict.load(ctx.bn1_running_var, "bn1.running_var", 64);
	dict.load(ctx.layer1_0_conv1_weight, "layer1.0.conv1.weight", 64, 64, 1, 1);
	dict.load(ctx.layer1_0_bn1_weight, "layer1.0.bn1.weight", 64);
	dict.load(ctx.layer1_0_bn1_bias, "layer1.0.bn1.bias", 64);
	dict.load(ctx.layer1_0_bn1_running_mean, "layer1.0.bn1.running_mean", 64);
	dict.load(ctx.layer1_0_bn1_running_var, "layer1.0.bn1.running_var", 64);
	dict.load(ctx.layer1_0_conv2_weight, "layer1.0.conv2.weight", 64, 64, 3, 3);
	dict.load(ctx.layer1_0_bn2_weight, "layer1.0.bn2.weight", 64);
	dict.load(ctx.layer1_0_bn2_bias, "layer1.0.bn2.bias", 64);
	dict.load(ctx.layer1_0_bn2_running_mean, "layer1.0.bn2.running_mean", 64);
	dict.load(ctx.layer1_0_bn2_running_var, "layer1.0.bn2.running_var", 64);
	dict.load(ctx.layer1_0_conv3_weight, "layer1.0.conv3.weight", 256, 64, 1, 1);
	dict.load(ctx.layer1_0_bn3_weight, "layer1.0.bn3.weight", 256);
	dict.load(ctx.layer1_0_bn3_bias, "layer1.0.bn3.bias", 256);
	dict.load(ctx.layer1_0_bn3_running_mean, "layer1.0.bn3.running_mean", 256);
	dict.load(ctx.layer1_0_bn3_running_var, "layer1.0.bn3.running_var", 256);
	dict.load(ctx.layer1_0_downsample_0_weight, "layer1.0.downsample.0.weight", 256, 64, 1, 1);
	dict.load(ctx.layer1_0_downsample_1_weight, "layer1.0.downsample.1.weight", 256);
	dict.load(ctx.layer1_0_downsample_1_bias, "layer1.0.downsample.1.bias", 256);
	dict.load(ctx.layer1_0_downsample_1_running_mean, "layer1.0.downsample.1.running_mean", 256);
	dict.load(ctx.layer1_0_downsample_1_running_var, "layer1.0.downsample.1.running_var", 256);
	dict.load(ctx.layer1_1_conv1_weight, "layer1.1.conv1.weight", 64, 256, 1, 1);
	dict.load(ctx.layer1_1_bn1_weight, "layer1.1.bn1.weight", 64);
	dict.load(ctx.layer1_1_bn1_bias, "layer1.1.bn1.bias", 64);
	dict.load(ctx.layer1_1_bn1_running_mean, "layer1.1.bn1.running_mean", 64);
	dict.load(ctx.layer1_1_bn1_running_var, "layer1.1.bn1.running_var", 64);
	dict.load(ctx.layer1_1_conv2_weight, "layer1.1.conv2.weight", 64, 64, 3, 3);
	dict.load(ctx.layer1_1_bn2_weight, "layer1.1.bn2.weight", 64);
	dict.load(ctx.layer1_1_bn2_bias, "layer1.1.bn2.bias", 64);
	dict.load(ctx.layer1_1_bn2_running_mean, "layer1.1.bn2.running_mean", 64);
	dict.load(ctx.layer1_1_bn2_running_var, "layer1.1.bn2.running_var", 64);
	dict.load(ctx.layer1_1_conv3_weight, "layer1.1.conv3.weight", 256, 64, 1, 1);
	dict.load(ctx.layer1_1_bn3_weight, "layer1.1.bn3.weight", 256);
	dict.load(ctx.layer1_1_bn3_bias, "layer1.1.bn3.bias", 256);
	dict.load(ctx.layer1_1_bn3_running_mean, "layer1.1.bn3.running_mean", 256);
	dict.load(ctx.layer1_1_bn3_running_var, "layer1.1.bn3.running_var", 256);
	dict.load(ctx.layer1_2_conv1_weight, "layer1.2.conv1.weight", 64, 256, 1, 1);
	dict.load(ctx.layer1_2_bn1_weight, "layer1.2.bn1.weight", 64);
	dict.load(ctx.layer1_2_bn1_bias, "layer1.2.bn1.bias", 64);
	dict.load(ctx.layer1_2_bn1_running_mean, "layer1.2.bn1.running_mean", 64);
	dict.load(ctx.layer1_2_bn1_running_var, "layer1.2.bn1.running_var", 64);
	dict.load(ctx.layer1_2_conv2_weight, "layer1.2.conv2.weight", 64, 64, 3, 3);
	dict.load(ctx.layer1_2_bn2_weight, "layer1.2.bn2.weight", 64);
	dict.load(ctx.layer1_2_bn2_bias, "layer1.2.bn2.bias", 64);
	dict.load(ctx.layer1_2_bn2_running_mean, "layer1.2.bn2.running_mean", 64);
	dict.load(ctx.layer1_2_bn2_running_var, "layer1.2.bn2.running_var", 64);
	dict.load(ctx.layer1_2_conv3_weight, "layer1.2.conv3.weight", 256, 64, 1, 1);
	dict.load(ctx.layer1_2_bn3_weight, "layer1.2.bn3.weight", 256);
	dict.load(ctx.layer1_2_bn3_bias, "layer1.2.bn3.bias", 256);
	dict.load(ctx.layer1_2_bn3_running_mean, "layer1.2.bn3.running_mean", 256);
	dict.load(ctx.layer1_2_bn3_running_var, "layer1.2.bn3.running_var", 256);
	dict.load(ctx.layer2_0_conv1_weight, "layer2.0.conv1.weight", 128, 256, 1, 1);
	dict.load(ctx.layer2_0_bn1_weight, "layer2.0.bn1.weight", 128);
	dict.load(ctx.layer2_0_bn1_bias, "layer2.0.bn1.bias", 128);
	dict.load(ctx.layer2_0_bn1_running_mean, "layer2.0.bn1.running_mean", 128);
	dict.load(ctx.layer2_0_bn1_running_var, "layer2.0.bn1.running_var", 128);
	dict.load(ctx.layer2_0_conv2_weight, "layer2.0.conv2.weight", 128, 128, 3, 3);
	dict.load(ctx.layer2_0_bn2_weight, "layer2.0.bn2.weight", 128);
	dict.load(ctx.layer2_0_bn2_bias, "layer2.0.bn2.bias", 128);
	dict.load(ctx.layer2_0_bn2_running_mean, "layer2.0.bn2.running_mean", 128);
	dict.load(ctx.layer2_0_bn2_running_var, "layer2.0.bn2.running_var", 128);
	dict.load(ctx.layer2_0_conv3_weight, "layer2.0.conv3.weight", 512, 128, 1, 1);
	dict.load(ctx.layer2_0_bn3_weight, "layer2.0.bn3.weight", 512);
	dict.load(ctx.layer2_0_bn3_bias, "layer2.0.bn3.bias", 512);
	dict.load(ctx.layer2_0_bn3_running_mean, "layer2.0.bn3.running_mean", 512);
	dict.load(ctx.layer2_0_bn3_running_var, "layer2.0.bn3.running_var", 512);
	dict.load(ctx.layer2_0_downsample_0_weight, "layer2.0.downsample.0.weight", 512, 256, 1, 1);
	dict.load(ctx.layer2_0_downsample_1_weight, "layer2.0.downsample.1.weight", 512);
	dict.load(ctx.layer2_0_downsample_1_bias, "layer2.0.downsample.1.bias", 512);
	dict.load(ctx.layer2_0_downsample_1_running_mean, "layer2.0.downsample.1.running_mean", 512);
	dict.load(ctx.layer2_0_downsample_1_running_var, "layer2.0.downsample.1.running_var", 512);
	dict.load(ctx.layer2_1_conv1_weight, "layer2.1.conv1.weight", 128, 512, 1, 1);
	dict.load(ctx.layer2_1_bn1_weight, "layer2.1.bn1.weight", 128);
	dict.load(ctx.layer2_1_bn1_bias, "layer2.1.bn1.bias", 128);
	dict.load(ctx.layer2_1_bn1_running_mean, "layer2.1.bn1.running_mean", 128);
	dict.load(ctx.layer2_1_bn1_running_var, "layer2.1.bn1.running_var", 128);
	dict.load(ctx.layer2_1_conv2_weight, "layer2.1.conv2.weight", 128, 128, 3, 3);
	dict.load(ctx.layer2_1_bn2_weight, "layer2.1.bn2.weight", 128);
	dict.load(ctx.layer2_1_bn2_bias, "layer2.1.bn2.bias", 128);
	dict.load(ctx.layer2_1_bn2_running_mean, "layer2.1.bn2.running_mean", 128);
	dict.load(ctx.layer2_1_bn2_running_var, "layer2.1.bn2.running_var", 128);
	dict.load(ctx.layer2_1_conv3_weight, "layer2.1.conv3.weight", 512, 128, 1, 1);
	dict.load(ctx.layer2_1_bn3_weight, "layer2.1.bn3.weight", 512);
	dict.load(ctx.layer2_1_bn3_bias, "layer2.1.bn3.bias", 512);
	dict.load(ctx.layer2_1_bn3_running_mean, "layer2.1.bn3.running_mean", 512);
	dict.load(ctx.layer2_1_bn3_running_var, "layer2.1.bn3.running_var", 512);
	dict.load(ctx.layer2_2_conv1_weight, "layer2.2.conv1.weight", 128, 512, 1, 1);
	dict.load(ctx.layer2_2_bn1_weight, "layer2.2.bn1.weight", 128);
	dict.load(ctx.layer2_2_bn1_bias, "layer2.2.bn1.bias", 128);
	dict.load(ctx.layer2_2_bn1_running_mean, "layer2.2.bn1.running_mean", 128);
	dict.load(ctx.layer2_2_bn1_running_var, "layer2.2.bn1.running_var", 128);
	dict.load(ctx.layer2_2_conv2_weight, "layer2.2.conv2.weight", 128, 128, 3, 3);
	dict.load(ctx.layer2_2_bn2_weight, "layer2.2.bn2.weight", 128);
	dict.load(ctx.layer2_2_bn2_bias, "layer2.2.bn2.bias", 128);
	dict.load(ctx.layer2_2_bn2_running_mean, "layer2.2.bn2.running_mean", 128);
	dict.load(ctx.layer2_2_bn2_running_var, "layer2.2.bn2.running_var", 128);
	dict.load(ctx.layer2_2_conv3_weight, "layer2.2.conv3.weight", 512, 128, 1, 1);
	dict.load(ctx.layer2_2_bn3_weight, "layer2.2.bn3.weight", 512);
	dict.load(ctx.layer2_2_bn3_bias, "layer2.2.bn3.bias", 512);
	dict.load(ctx.layer2_2_bn3_running_mean, "layer2.2.bn3.running_mean", 512);
	dict.load(ctx.layer2_2_bn3_running_var, "layer2.2.bn3.running_var", 512);
	dict.load(ctx.layer2_3_conv1_weight, "layer2.3.conv1.weight", 128, 512, 1, 1);
	dict.load(ctx.layer2_3_bn1_weight, "layer2.3.bn1.weight", 128);
	dict.load(ctx.layer2_3_bn1_bias, "layer2.3.bn1.bias", 128);
	dict.load(ctx.layer2_3_bn1_running_mean, "layer2.3.bn1.running_mean", 128);
	dict.load(ctx.layer2_3_bn1_running_var, "layer2.3.bn1.running_var", 128);
	dict.load(ctx.layer2_3_conv2_weight, "layer2.3.conv2.weight", 128, 128, 3, 3);
	dict.load(ctx.layer2_3_bn2_weight, "layer2.3.bn2.weight", 128);
	dict.load(ctx.layer2_3_bn2_bias, "layer2.3.bn2.bias", 128);
	dict.load(ctx.layer2_3_bn2_running_mean, "layer2.3.bn2.running_mean", 128);
	dict.load(ctx.layer2_3_bn2_running_var, "layer2.3.bn2.running_var", 128);
	dict.load(ctx.layer2_3_conv3_weight, "layer2.3.conv3.weight", 512, 128, 1, 1);
	dict.load(ctx.layer2_3_bn3_weight, "layer2.3.bn3.weight", 512);
	dict.load(ctx.layer2_3_bn3_bias, "layer2.3.bn3.bias", 512);
	dict.load(ctx.layer2_3_bn3_running_mean, "layer2.3.bn3.running_mean", 512);
	dict.load(ctx.layer2_3_bn3_running_var, "layer2.3.bn3.running_var", 512);
	dict.load(ctx.layer3_0_conv1_weight, "layer3.0.conv1.weight", 256, 512, 1, 1);
	dict.load(ctx.layer3_0_bn1_weight, "layer3.0.bn1.weight", 256);
	dict.load(ctx.layer3_0_bn1_bias, "layer3.0.bn1.bias", 256);
	dict.load(ctx.layer3_0_bn1_running_mean, "layer3.0.bn1.running_mean", 256);
	dict.load(ctx.layer3_0_bn1_running_var, "layer3.0.bn1.running_var", 256);
	dict.load(ctx.layer3_0_conv2_weight, "layer3.0.conv2.weight", 256, 256, 3, 3);
	dict.load(ctx.layer3_0_bn2_weight, "layer3.0.bn2.weight", 256);
	dict.load(ctx.layer3_0_bn2_bias, "layer3.0.bn2.bias", 256);
	dict.load(ctx.layer3_0_bn2_running_mean, "layer3.0.bn2.running_mean", 256);
	dict.load(ctx.layer3_0_bn2_running_var, "layer3.0.bn2.running_var", 256);
	dict.load(ctx.layer3_0_conv3_weight, "layer3.0.conv3.weight", 1024, 256, 1, 1);
	dict.load(ctx.layer3_0_bn3_weight, "layer3.0.bn3.weight", 1024);
	dict.load(ctx.layer3_0_bn3_bias, "layer3.0.bn3.bias", 1024);
	dict.load(ctx.layer3_0_bn3_running_mean, "layer3.0.bn3.running_mean", 1024);
	dict.load(ctx.layer3_0_bn3_running_var, "layer3.0.bn3.running_var", 1024);
	dict.load(ctx.layer3_0_downsample_0_weight, "layer3.0.downsample.0.weight", 1024, 512, 1, 1);
	dict.load(ctx.layer3_0_downsample_1_weight, "layer3.0.downsample.1.weight", 1024);
	dict.load(ctx.layer3_0_downsample_1_bias, "layer3.0.downsample.1.bias", 1024);
	dict.load(ctx.layer3_0_downsample_1_running_mean, "layer3.0.downsample.1.running_mean", 1024);
	dict.load(ctx.layer3_0_downsample_1_running_var, "layer3.0.downsample.1.running_var", 1024);
	dict.load(ctx.layer3_1_conv1_weight, "layer3.1.conv1.weight", 256, 1024, 1, 1);
	dict.load(ctx.layer3_1_bn1_weight, "layer3.1.bn1.weight", 256);
	dict.load(ctx.layer3_1_bn1_bias, "layer3.1.bn1.bias", 256);
	dict.load(ctx.layer3_1_bn1_running_mean, "layer3.1.bn1.running_mean", 256);
	dict.load(ctx.layer3_1_bn1_running_var, "layer3.1.bn1.running_var", 256);
	dict.load(ctx.layer3_1_conv2_weight, "layer3.1.conv2.weight", 256, 256, 3, 3);
	dict.load(ctx.layer3_1_bn2_weight, "layer3.1.bn2.weight", 256);
	dict.load(ctx.layer3_1_bn2_bias, "layer3.1.bn2.bias", 256);
	dict.load(ctx.layer3_1_bn2_running_mean, "layer3.1.bn2.running_mean", 256);
	dict.load(ctx.layer3_1_bn2_running_var, "layer3.1.bn2.running_var", 256);
	dict.load(ctx.layer3_1_conv3_weight, "layer3.1.conv3.weight", 1024, 256, 1, 1);
	dict.load(ctx.layer3_1_bn3_weight, "layer3.1.bn3.weight", 1024);
	dict.load(ctx.layer3_1_bn3_bias, "layer3.1.bn3.bias", 1024);
	dict.load(ctx.layer3_1_bn3_running_mean, "layer3.1.bn3.running_mean", 1024);
	dict.load(ctx.layer3_1_bn3_running_var, "layer3.1.bn3.running_var", 1024);
	dict.load(ctx.layer3_2_conv1_weight, "layer3.2.conv1.weight", 256, 1024, 1, 1);
	dict.load(ctx.layer3_2_bn1_weight, "layer3.2.bn1.weight", 256);
	dict.load(ctx.layer3_2_bn1_bias, "layer3.2.bn1.bias", 256);
	dict.load(ctx.layer3_2_bn1_running_mean, "layer3.2.bn1.running_mean", 256);
	dict.load(ctx.layer3_2_bn1_running_var, "layer3.2.bn1.running_var", 256);
	dict.load(ctx.layer3_2_conv2_weight, "layer3.2.conv2.weight", 256, 256, 3, 3);
	dict.load(ctx.layer3_2_bn2_weight, "layer3.2.bn2.weight", 256);
	dict.load(ctx.layer3_2_bn2_bias, "layer3.2.bn2.bias", 256);
	dict.load(ctx.layer3_2_bn2_running_mean, "layer3.2.bn2.running_mean", 256);
	dict.load(ctx.layer3_2_bn2_running_var, "layer3.2.bn2.running_var", 256);
	dict.load(ctx.layer3_2_conv3_weight, "layer3.2.conv3.weight", 1024, 256, 1, 1);
	dict.load(ctx.layer3_2_bn3_weight, "layer3.2.bn3.weight", 1024);
	dict.load(ctx.layer3_2_bn3_bias, "layer3.2.bn3.bias", 1024);
	dict.load(ctx.layer3_2_bn3_running_mean, "layer3.2.bn3.running_mean", 1024);
	dict.load(ctx.layer3_2_bn3_running_var, "layer3.2.bn3.running_var", 1024);
	dict.load(ctx.layer3_3_conv1_weight, "layer3.3.conv1.weight", 256, 1024, 1, 1);
	dict.load(ctx.layer3_3_bn1_weight, "layer3.3.bn1.weight", 256);
	dict.load(ctx.layer3_3_bn1_bias, "layer3.3.bn1.bias", 256);
	dict.load(ctx.layer3_3_bn1_running_mean, "layer3.3.bn1.running_mean", 256);
	dict.load(ctx.layer3_3_bn1_running_var, "layer3.3.bn1.running_var", 256);
	dict.load(ctx.layer3_3_conv2_weight, "layer3.3.conv2.weight", 256, 256, 3, 3);
	dict.load(ctx.layer3_3_bn2_weight, "layer3.3.bn2.weight", 256);
	dict.load(ctx.layer3_3_bn2_bias, "layer3.3.bn2.bias", 256);
	dict.load(ctx.layer3_3_bn2_running_mean, "layer3.3.bn2.running_mean", 256);
	dict.load(ctx.layer3_3_bn2_running_var, "layer3.3.bn2.running_var", 256);
	dict.load(ctx.layer3_3_conv3_weight, "layer3.3.conv3.weight", 1024, 256, 1, 1);
	dict.load(ctx.layer3_3_bn3_weight, "layer3.3.bn3.weight", 1024);
	dict.load(ctx.layer3_3_bn3_bias, "layer3.3.bn3.bias", 1024);
	dict.load(ctx.layer3_3_bn3_running_mean, "layer3.3.bn3.running_mean", 1024);
	dict.load(ctx.layer3_3_bn3_running_var, "layer3.3.bn3.running_var", 1024);
	dict.load(ctx.layer3_4_conv1_weight, "layer3.4.conv1.weight", 256, 1024, 1, 1);
	dict.load(ctx.layer3_4_bn1_weight, "layer3.4.bn1.weight", 256);
	dict.load(ctx.layer3_4_bn1_bias, "layer3.4.bn1.bias", 256);
	dict.load(ctx.layer3_4_bn1_running_mean, "layer3.4.bn1.running_mean", 256);
	dict.load(ctx.layer3_4_bn1_running_var, "layer3.4.bn1.running_var", 256);
	dict.load(ctx.layer3_4_conv2_weight, "layer3.4.conv2.weight", 256, 256, 3, 3);
	dict.load(ctx.layer3_4_bn2_weight, "layer3.4.bn2.weight", 256);
	dict.load(ctx.layer3_4_bn2_bias, "layer3.4.bn2.bias", 256);
	dict.load(ctx.layer3_4_bn2_running_mean, "layer3.4.bn2.running_mean", 256);
	dict.load(ctx.layer3_4_bn2_running_var, "layer3.4.bn2.running_var", 256);
	dict.load(ctx.layer3_4_conv3_weight, "layer3.4.conv3.weight", 1024, 256, 1, 1);
	dict.load(ctx.layer3_4_bn3_weight, "layer3.4.bn3.weight", 1024);
	dict.load(ctx.layer3_4_bn3_bias, "layer3.4.bn3.bias", 1024);
	dict.load(ctx.layer3_4_bn3_running_mean, "layer3.4.bn3.running_mean", 1024);
	dict.load(ctx.layer3_4_bn3_running_var, "layer3.4.bn3.running_var", 1024);
	dict.load(ctx.layer3_5_conv1_weight, "layer3.5.conv1.weight", 256, 1024, 1, 1);
	dict.load(ctx.layer3_5_bn1_weight, "layer3.5.bn1.weight", 256);
	dict.load(ctx.layer3_5_bn1_bias, "layer3.5.bn1.bias", 256);
	dict.load(ctx.layer3_5_bn1_running_mean, "layer3.5.bn1.running_mean", 256);
	dict.load(ctx.layer3_5_bn1_running_var, "layer3.5.bn1.running_var", 256);
	dict.load(ctx.layer3_5_conv2_weight, "layer3.5.conv2.weight", 256, 256, 3, 3);
	dict.load(ctx.layer3_5_bn2_weight, "layer3.5.bn2.weight", 256);
	dict.load(ctx.layer3_5_bn2_bias, "layer3.5.bn2.bias", 256);
	dict.load(ctx.layer3_5_bn2_running_mean, "layer3.5.bn2.running_mean", 256);
	dict.load(ctx.layer3_5_bn2_running_var, "layer3.5.bn2.running_var", 256);
	dict.load(ctx.layer3_5_conv3_weight, "layer3.5.conv3.weight", 1024, 256, 1, 1);
	dict.load(ctx.layer3_5_bn3_weight, "layer3.5.bn3.weight", 1024);
	dict.load(ctx.layer3_5_bn3_bias, "layer3.5.bn3.bias", 1024);
	dict.load(ctx.layer3_5_bn3_running_mean, "layer3.5.bn3.running_mean", 1024);
	dict.load(ctx.layer3_5_bn3_running_var, "layer3.5.bn3.running_var", 1024);
	dict.load(ctx.layer4_0_conv1_weight, "layer4.0.conv1.weight", 512, 1024, 1, 1);
	dict.load(ctx.layer4_0_bn1_weight, "layer4.0.bn1.weight", 512);
	dict.load(ctx.layer4_0_bn1_bias, "layer4.0.bn1.bias", 512);
	dict.load(ctx.layer4_0_bn1_running_mean, "layer4.0.bn1.running_mean", 512);
	dict.load(ctx.layer4_0_bn1_running_var, "layer4.0.bn1.running_var", 512);
	dict.load(ctx.layer4_0_conv2_weight, "layer4.0.conv2.weight", 512, 512, 3, 3);
	dict.load(ctx.layer4_0_bn2_weight, "layer4.0.bn2.weight", 512);
	dict.load(ctx.layer4_0_bn2_bias, "layer4.0.bn2.bias", 512);
	dict.load(ctx.layer4_0_bn2_running_mean, "layer4.0.bn2.running_mean", 512);
	dict.load(ctx.layer4_0_bn2_running_var, "layer4.0.bn2.running_var", 512);
	dict.load(ctx.layer4_0_conv3_weight, "layer4.0.conv3.weight", 2048, 512, 1, 1);
	dict.load(ctx.layer4_0_bn3_weight, "layer4.0.bn3.weight", 2048);
	dict.load(ctx.layer4_0_bn3_bias, "layer4.0.bn3.bias", 2048);
	dict.load(ctx.layer4_0_bn3_running_mean, "layer4.0.bn3.running_mean", 2048);
	dict.load(ctx.layer4_0_bn3_running_var, "layer4.0.bn3.running_var", 2048);
	dict.load(ctx.layer4_0_downsample_0_weight, "layer4.0.downsample.0.weight", 2048, 1024, 1, 1);
	dict.load(ctx.layer4_0_downsample_1_weight, "layer4.0.downsample.1.weight", 2048);
	dict.load(ctx.layer4_0_downsample_1_bias, "layer4.0.downsample.1.bias", 2048);
	dict.load(ctx.layer4_0_downsample_1_running_mean, "layer4.0.downsample.1.running_mean", 2048);
	dict.load(ctx.layer4_0_downsample_1_running_var, "layer4.0.downsample.1.running_var", 2048);
	dict.load(ctx.layer4_1_conv1_weight, "layer4.1.conv1.weight", 512, 2048, 1, 1);
	dict.load(ctx.layer4_1_bn1_weight, "layer4.1.bn1.weight", 512);
	dict.load(ctx.layer4_1_bn1_bias, "layer4.1.bn1.bias", 512);
	dict.load(ctx.layer4_1_bn1_running_mean, "layer4.1.bn1.running_mean", 512);
	dict.load(ctx.layer4_1_bn1_running_var, "layer4.1.bn1.running_var", 512);
	dict.load(ctx.layer4_1_conv2_weight, "layer4.1.conv2.weight", 512, 512, 3, 3);
	dict.load(ctx.layer4_1_bn2_weight, "layer4.1.bn2.weight", 512);
	dict.load(ctx.layer4_1_bn2_bias, "layer4.1.bn2.bias", 512);
	dict.load(ctx.layer4_1_bn2_running_mean, "layer4.1.bn2.running_mean", 512);
	dict.load(ctx.layer4_1_bn2_running_var, "layer4.1.bn2.running_var", 512);
	dict.load(ctx.layer4_1_conv3_weight, "layer4.1.conv3.weight", 2048, 512, 1, 1);
	dict.load(ctx.layer4_1_bn3_weight, "layer4.1.bn3.weight", 2048);
	dict.load(ctx.layer4_1_bn3_bias, "layer4.1.bn3.bias", 2048);
	dict.load(ctx.layer4_1_bn3_running_mean, "layer4.1.bn3.running_mean", 2048);
	dict.load(ctx.layer4_1_bn3_running_var, "layer4.1.bn3.running_var", 2048);
	dict.load(ctx.layer4_2_conv1_weight, "layer4.2.conv1.weight", 512, 2048, 1, 1);
	dict.load(ctx.layer4_2_bn1_weight, "layer4.2.bn1.weight", 512);
	dict.load(ctx.layer4_2_bn1_bias, "layer4.2.bn1.bias", 512);
	dict.load(ctx.layer4_2_bn1_running_mean, "layer4.2.bn1.running_mean", 512);
	dict.load(ctx.layer4_2_bn1_running_var, "layer4.2.bn1.running_var", 512);
	dict.load(ctx.layer4_2_conv2_weight, "layer4.2.conv2.weight", 512, 512, 3, 3);
	dict.load(ctx.layer4_2_bn2_weight, "layer4.2.bn2.weight", 512);
	dict.load(ctx.layer4_2_bn2_bias, "layer4.2.bn2.bias", 512);
	dict.load(ctx.layer4_2_bn2_running_mean, "layer4.2.bn2.running_mean", 512);
	dict.load(ctx.layer4_2_bn2_running_var, "layer4.2.bn2.running_var", 512);
	dict.load(ctx.layer4_2_conv3_weight, "layer4.2.conv3.weight", 2048, 512, 1, 1);
	dict.load(ctx.layer4_2_bn3_weight, "layer4.2.bn3.weight", 2048);
	dict.load(ctx.layer4_2_bn3_bias, "layer4.2.bn3.bias", 2048);
	dict.load(ctx.layer4_2_bn3_running_mean, "layer4.2.bn3.running_mean", 2048);
	dict.load(ctx.layer4_2_bn3_running_var, "layer4.2.bn3.running_var", 2048);
	dict.load(ctx.fc_weight, "fc.weight", 1000, 2048);
	dict.load(ctx.fc_bias, "fc.bias", 1000);
	return ctx;
}


t4::tensor2f ResNetForward(const ResNet& ctx, t4::tensor4f x0)
{
	t4::tensor4f x268 = t4::Conv2d<7, 7, 2, 2, 3, 3, 1, 1>(x0, ctx.conv1_weight); //conv1
	t4::free(x0);
	t4::tensor4f x269 = t4::BatchNormalization(x268, ctx.bn1_weight, ctx.bn1_bias, ctx.bn1_running_mean, ctx.bn1_running_var, 1e-05f); //bn1
	t4::free(x268);
	t4::tensor4f x270 = t4::Relu(x269); //relu
	t4::free(x269);
	t4::tensor4f x271 = t4::MaxPool2d<3, 3, 2, 2, 1, 1>(x270); //maxpool
	t4::free(x270);
	t4::tensor4f x272 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x271, ctx.layer1_0_conv1_weight); //layer1.0.conv1
	t4::tensor4f x273 = t4::BatchNormalization(x272, ctx.layer1_0_bn1_weight, ctx.layer1_0_bn1_bias, ctx.layer1_0_bn1_running_mean, ctx.layer1_0_bn1_running_var, 1e-05f); //layer1.0.bn1
	t4::free(x272);
	t4::tensor4f x274 = t4::Relu(x273); //layer1.0.relu
	t4::free(x273);
	t4::tensor4f x275 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x274, ctx.layer1_0_conv2_weight); //layer1.0.conv2
	t4::free(x274);
	t4::tensor4f x276 = t4::BatchNormalization(x275, ctx.layer1_0_bn2_weight, ctx.layer1_0_bn2_bias, ctx.layer1_0_bn2_running_mean, ctx.layer1_0_bn2_running_var, 1e-05f); //layer1.0.bn2
	t4::free(x275);
	t4::tensor4f x277 = t4::Relu(x276); //layer1.0.relu
	t4::free(x276);
	t4::tensor4f x278 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x277, ctx.layer1_0_conv3_weight); //layer1.0.conv3
	t4::free(x277);
	t4::tensor4f x279 = t4::BatchNormalization(x278, ctx.layer1_0_bn3_weight, ctx.layer1_0_bn3_bias, ctx.layer1_0_bn3_running_mean, ctx.layer1_0_bn3_running_var, 1e-05f); //layer1.0.bn3
	t4::free(x278);
	t4::tensor4f x280 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x271, ctx.layer1_0_downsample_0_weight); //layer1.0.downsample.0
	t4::free(x271);
	t4::tensor4f x281 = t4::BatchNormalization(x280, ctx.layer1_0_downsample_1_weight, ctx.layer1_0_downsample_1_bias, ctx.layer1_0_downsample_1_running_mean, ctx.layer1_0_downsample_1_running_var, 1e-05f); //layer1.0.downsample.1
	t4::free(x280);
	t4::tensor4f x282 = t4::Add(x279, x281); //layer1.0
	t4::free(x279, x281);
	t4::tensor4f x283 = t4::Relu(x282); //layer1.0.relu
	t4::free(x282);
	t4::tensor4f x284 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x283, ctx.layer1_1_conv1_weight); //layer1.1.conv1
	t4::tensor4f x285 = t4::BatchNormalization(x284, ctx.layer1_1_bn1_weight, ctx.layer1_1_bn1_bias, ctx.layer1_1_bn1_running_mean, ctx.layer1_1_bn1_running_var, 1e-05f); //layer1.1.bn1
	t4::free(x284);
	t4::tensor4f x286 = t4::Relu(x285); //layer1.1.relu
	t4::free(x285);
	t4::tensor4f x287 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x286, ctx.layer1_1_conv2_weight); //layer1.1.conv2
	t4::free(x286);
	t4::tensor4f x288 = t4::BatchNormalization(x287, ctx.layer1_1_bn2_weight, ctx.layer1_1_bn2_bias, ctx.layer1_1_bn2_running_mean, ctx.layer1_1_bn2_running_var, 1e-05f); //layer1.1.bn2
	t4::free(x287);
	t4::tensor4f x289 = t4::Relu(x288); //layer1.1.relu
	t4::free(x288);
	t4::tensor4f x290 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x289, ctx.layer1_1_conv3_weight); //layer1.1.conv3
	t4::free(x289);
	t4::tensor4f x291 = t4::BatchNormalization(x290, ctx.layer1_1_bn3_weight, ctx.layer1_1_bn3_bias, ctx.layer1_1_bn3_running_mean, ctx.layer1_1_bn3_running_var, 1e-05f); //layer1.1.bn3
	t4::free(x290);
	t4::tensor4f x292 = t4::Add(x291, x283); //layer1.1
	t4::free(x283, x291);
	t4::tensor4f x293 = t4::Relu(x292); //layer1.1.relu
	t4::free(x292);
	t4::tensor4f x294 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x293, ctx.layer1_2_conv1_weight); //layer1.2.conv1
	t4::tensor4f x295 = t4::BatchNormalization(x294, ctx.layer1_2_bn1_weight, ctx.layer1_2_bn1_bias, ctx.layer1_2_bn1_running_mean, ctx.layer1_2_bn1_running_var, 1e-05f); //layer1.2.bn1
	t4::free(x294);
	t4::tensor4f x296 = t4::Relu(x295); //layer1.2.relu
	t4::free(x295);
	t4::tensor4f x297 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x296, ctx.layer1_2_conv2_weight); //layer1.2.conv2
	t4::free(x296);
	t4::tensor4f x298 = t4::BatchNormalization(x297, ctx.layer1_2_bn2_weight, ctx.layer1_2_bn2_bias, ctx.layer1_2_bn2_running_mean, ctx.layer1_2_bn2_running_var, 1e-05f); //layer1.2.bn2
	t4::free(x297);
	t4::tensor4f x299 = t4::Relu(x298); //layer1.2.relu
	t4::free(x298);
	t4::tensor4f x300 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x299, ctx.layer1_2_conv3_weight); //layer1.2.conv3
	t4::free(x299);
	t4::tensor4f x301 = t4::BatchNormalization(x300, ctx.layer1_2_bn3_weight, ctx.layer1_2_bn3_bias, ctx.layer1_2_bn3_running_mean, ctx.layer1_2_bn3_running_var, 1e-05f); //layer1.2.bn3
	t4::free(x300);
	t4::tensor4f x302 = t4::Add(x301, x293); //layer1.2
	t4::free(x293, x301);
	t4::tensor4f x303 = t4::Relu(x302); //layer1.2.relu
	t4::free(x302);
	t4::tensor4f x304 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x303, ctx.layer2_0_conv1_weight); //layer2.0.conv1
	t4::tensor4f x305 = t4::BatchNormalization(x304, ctx.layer2_0_bn1_weight, ctx.layer2_0_bn1_bias, ctx.layer2_0_bn1_running_mean, ctx.layer2_0_bn1_running_var, 1e-05f); //layer2.0.bn1
	t4::free(x304);
	t4::tensor4f x306 = t4::Relu(x305); //layer2.0.relu
	t4::free(x305);
	t4::tensor4f x307 = t4::Conv2d<3, 3, 2, 2, 1, 1, 1, 1>(x306, ctx.layer2_0_conv2_weight); //layer2.0.conv2
	t4::free(x306);
	t4::tensor4f x308 = t4::BatchNormalization(x307, ctx.layer2_0_bn2_weight, ctx.layer2_0_bn2_bias, ctx.layer2_0_bn2_running_mean, ctx.layer2_0_bn2_running_var, 1e-05f); //layer2.0.bn2
	t4::free(x307);
	t4::tensor4f x309 = t4::Relu(x308); //layer2.0.relu
	t4::free(x308);
	t4::tensor4f x310 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x309, ctx.layer2_0_conv3_weight); //layer2.0.conv3
	t4::free(x309);
	t4::tensor4f x311 = t4::BatchNormalization(x310, ctx.layer2_0_bn3_weight, ctx.layer2_0_bn3_bias, ctx.layer2_0_bn3_running_mean, ctx.layer2_0_bn3_running_var, 1e-05f); //layer2.0.bn3
	t4::free(x310);
	t4::tensor4f x312 = t4::Conv2d<1, 1, 2, 2, 0, 0, 1, 1>(x303, ctx.layer2_0_downsample_0_weight); //layer2.0.downsample.0
	t4::free(x303);
	t4::tensor4f x313 = t4::BatchNormalization(x312, ctx.layer2_0_downsample_1_weight, ctx.layer2_0_downsample_1_bias, ctx.layer2_0_downsample_1_running_mean, ctx.layer2_0_downsample_1_running_var, 1e-05f); //layer2.0.downsample.1
	t4::free(x312);
	t4::tensor4f x314 = t4::Add(x311, x313); //layer2.0
	t4::free(x311, x313);
	t4::tensor4f x315 = t4::Relu(x314); //layer2.0.relu
	t4::free(x314);
	t4::tensor4f x316 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x315, ctx.layer2_1_conv1_weight); //layer2.1.conv1
	t4::tensor4f x317 = t4::BatchNormalization(x316, ctx.layer2_1_bn1_weight, ctx.layer2_1_bn1_bias, ctx.layer2_1_bn1_running_mean, ctx.layer2_1_bn1_running_var, 1e-05f); //layer2.1.bn1
	t4::free(x316);
	t4::tensor4f x318 = t4::Relu(x317); //layer2.1.relu
	t4::free(x317);
	t4::tensor4f x319 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x318, ctx.layer2_1_conv2_weight); //layer2.1.conv2
	t4::free(x318);
	t4::tensor4f x320 = t4::BatchNormalization(x319, ctx.layer2_1_bn2_weight, ctx.layer2_1_bn2_bias, ctx.layer2_1_bn2_running_mean, ctx.layer2_1_bn2_running_var, 1e-05f); //layer2.1.bn2
	t4::free(x319);
	t4::tensor4f x321 = t4::Relu(x320); //layer2.1.relu
	t4::free(x320);
	t4::tensor4f x322 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x321, ctx.layer2_1_conv3_weight); //layer2.1.conv3
	t4::free(x321);
	t4::tensor4f x323 = t4::BatchNormalization(x322, ctx.layer2_1_bn3_weight, ctx.layer2_1_bn3_bias, ctx.layer2_1_bn3_running_mean, ctx.layer2_1_bn3_running_var, 1e-05f); //layer2.1.bn3
	t4::free(x322);
	t4::tensor4f x324 = t4::Add(x323, x315); //layer2.1
	t4::free(x315, x323);
	t4::tensor4f x325 = t4::Relu(x324); //layer2.1.relu
	t4::free(x324);
	t4::tensor4f x326 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x325, ctx.layer2_2_conv1_weight); //layer2.2.conv1
	t4::tensor4f x327 = t4::BatchNormalization(x326, ctx.layer2_2_bn1_weight, ctx.layer2_2_bn1_bias, ctx.layer2_2_bn1_running_mean, ctx.layer2_2_bn1_running_var, 1e-05f); //layer2.2.bn1
	t4::free(x326);
	t4::tensor4f x328 = t4::Relu(x327); //layer2.2.relu
	t4::free(x327);
	t4::tensor4f x329 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x328, ctx.layer2_2_conv2_weight); //layer2.2.conv2
	t4::free(x328);
	t4::tensor4f x330 = t4::BatchNormalization(x329, ctx.layer2_2_bn2_weight, ctx.layer2_2_bn2_bias, ctx.layer2_2_bn2_running_mean, ctx.layer2_2_bn2_running_var, 1e-05f); //layer2.2.bn2
	t4::free(x329);
	t4::tensor4f x331 = t4::Relu(x330); //layer2.2.relu
	t4::free(x330);
	t4::tensor4f x332 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x331, ctx.layer2_2_conv3_weight); //layer2.2.conv3
	t4::free(x331);
	t4::tensor4f x333 = t4::BatchNormalization(x332, ctx.layer2_2_bn3_weight, ctx.layer2_2_bn3_bias, ctx.layer2_2_bn3_running_mean, ctx.layer2_2_bn3_running_var, 1e-05f); //layer2.2.bn3
	t4::free(x332);
	t4::tensor4f x334 = t4::Add(x333, x325); //layer2.2
	t4::free(x325, x333);
	t4::tensor4f x335 = t4::Relu(x334); //layer2.2.relu
	t4::free(x334);
	t4::tensor4f x336 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x335, ctx.layer2_3_conv1_weight); //layer2.3.conv1
	t4::tensor4f x337 = t4::BatchNormalization(x336, ctx.layer2_3_bn1_weight, ctx.layer2_3_bn1_bias, ctx.layer2_3_bn1_running_mean, ctx.layer2_3_bn1_running_var, 1e-05f); //layer2.3.bn1
	t4::free(x336);
	t4::tensor4f x338 = t4::Relu(x337); //layer2.3.relu
	t4::free(x337);
	t4::tensor4f x339 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x338, ctx.layer2_3_conv2_weight); //layer2.3.conv2
	t4::free(x338);
	t4::tensor4f x340 = t4::BatchNormalization(x339, ctx.layer2_3_bn2_weight, ctx.layer2_3_bn2_bias, ctx.layer2_3_bn2_running_mean, ctx.layer2_3_bn2_running_var, 1e-05f); //layer2.3.bn2
	t4::free(x339);
	t4::tensor4f x341 = t4::Relu(x340); //layer2.3.relu
	t4::free(x340);
	t4::tensor4f x342 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x341, ctx.layer2_3_conv3_weight); //layer2.3.conv3
	t4::free(x341);
	t4::tensor4f x343 = t4::BatchNormalization(x342, ctx.layer2_3_bn3_weight, ctx.layer2_3_bn3_bias, ctx.layer2_3_bn3_running_mean, ctx.layer2_3_bn3_running_var, 1e-05f); //layer2.3.bn3
	t4::free(x342);
	t4::tensor4f x344 = t4::Add(x343, x335); //layer2.3
	t4::free(x335, x343);
	t4::tensor4f x345 = t4::Relu(x344); //layer2.3.relu
	t4::free(x344);
	t4::tensor4f x346 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x345, ctx.layer3_0_conv1_weight); //layer3.0.conv1
	t4::tensor4f x347 = t4::BatchNormalization(x346, ctx.layer3_0_bn1_weight, ctx.layer3_0_bn1_bias, ctx.layer3_0_bn1_running_mean, ctx.layer3_0_bn1_running_var, 1e-05f); //layer3.0.bn1
	t4::free(x346);
	t4::tensor4f x348 = t4::Relu(x347); //layer3.0.relu
	t4::free(x347);
	t4::tensor4f x349 = t4::Conv2d<3, 3, 2, 2, 1, 1, 1, 1>(x348, ctx.layer3_0_conv2_weight); //layer3.0.conv2
	t4::free(x348);
	t4::tensor4f x350 = t4::BatchNormalization(x349, ctx.layer3_0_bn2_weight, ctx.layer3_0_bn2_bias, ctx.layer3_0_bn2_running_mean, ctx.layer3_0_bn2_running_var, 1e-05f); //layer3.0.bn2
	t4::free(x349);
	t4::tensor4f x351 = t4::Relu(x350); //layer3.0.relu
	t4::free(x350);
	t4::tensor4f x352 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x351, ctx.layer3_0_conv3_weight); //layer3.0.conv3
	t4::free(x351);
	t4::tensor4f x353 = t4::BatchNormalization(x352, ctx.layer3_0_bn3_weight, ctx.layer3_0_bn3_bias, ctx.layer3_0_bn3_running_mean, ctx.layer3_0_bn3_running_var, 1e-05f); //layer3.0.bn3
	t4::free(x352);
	t4::tensor4f x354 = t4::Conv2d<1, 1, 2, 2, 0, 0, 1, 1>(x345, ctx.layer3_0_downsample_0_weight); //layer3.0.downsample.0
	t4::free(x345);
	t4::tensor4f x355 = t4::BatchNormalization(x354, ctx.layer3_0_downsample_1_weight, ctx.layer3_0_downsample_1_bias, ctx.layer3_0_downsample_1_running_mean, ctx.layer3_0_downsample_1_running_var, 1e-05f); //layer3.0.downsample.1
	t4::free(x354);
	t4::tensor4f x356 = t4::Add(x353, x355); //layer3.0
	t4::free(x353, x355);
	t4::tensor4f x357 = t4::Relu(x356); //layer3.0.relu
	t4::free(x356);
	t4::tensor4f x358 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x357, ctx.layer3_1_conv1_weight); //layer3.1.conv1
	t4::tensor4f x359 = t4::BatchNormalization(x358, ctx.layer3_1_bn1_weight, ctx.layer3_1_bn1_bias, ctx.layer3_1_bn1_running_mean, ctx.layer3_1_bn1_running_var, 1e-05f); //layer3.1.bn1
	t4::free(x358);
	t4::tensor4f x360 = t4::Relu(x359); //layer3.1.relu
	t4::free(x359);
	t4::tensor4f x361 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x360, ctx.layer3_1_conv2_weight); //layer3.1.conv2
	t4::free(x360);
	t4::tensor4f x362 = t4::BatchNormalization(x361, ctx.layer3_1_bn2_weight, ctx.layer3_1_bn2_bias, ctx.layer3_1_bn2_running_mean, ctx.layer3_1_bn2_running_var, 1e-05f); //layer3.1.bn2
	t4::free(x361);
	t4::tensor4f x363 = t4::Relu(x362); //layer3.1.relu
	t4::free(x362);
	t4::tensor4f x364 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x363, ctx.layer3_1_conv3_weight); //layer3.1.conv3
	t4::free(x363);
	t4::tensor4f x365 = t4::BatchNormalization(x364, ctx.layer3_1_bn3_weight, ctx.layer3_1_bn3_bias, ctx.layer3_1_bn3_running_mean, ctx.layer3_1_bn3_running_var, 1e-05f); //layer3.1.bn3
	t4::free(x364);
	t4::tensor4f x366 = t4::Add(x365, x357); //layer3.1
	t4::free(x357, x365);
	t4::tensor4f x367 = t4::Relu(x366); //layer3.1.relu
	t4::free(x366);
	t4::tensor4f x368 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x367, ctx.layer3_2_conv1_weight); //layer3.2.conv1
	t4::tensor4f x369 = t4::BatchNormalization(x368, ctx.layer3_2_bn1_weight, ctx.layer3_2_bn1_bias, ctx.layer3_2_bn1_running_mean, ctx.layer3_2_bn1_running_var, 1e-05f); //layer3.2.bn1
	t4::free(x368);
	t4::tensor4f x370 = t4::Relu(x369); //layer3.2.relu
	t4::free(x369);
	t4::tensor4f x371 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x370, ctx.layer3_2_conv2_weight); //layer3.2.conv2
	t4::free(x370);
	t4::tensor4f x372 = t4::BatchNormalization(x371, ctx.layer3_2_bn2_weight, ctx.layer3_2_bn2_bias, ctx.layer3_2_bn2_running_mean, ctx.layer3_2_bn2_running_var, 1e-05f); //layer3.2.bn2
	t4::free(x371);
	t4::tensor4f x373 = t4::Relu(x372); //layer3.2.relu
	t4::free(x372);
	t4::tensor4f x374 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x373, ctx.layer3_2_conv3_weight); //layer3.2.conv3
	t4::free(x373);
	t4::tensor4f x375 = t4::BatchNormalization(x374, ctx.layer3_2_bn3_weight, ctx.layer3_2_bn3_bias, ctx.layer3_2_bn3_running_mean, ctx.layer3_2_bn3_running_var, 1e-05f); //layer3.2.bn3
	t4::free(x374);
	t4::tensor4f x376 = t4::Add(x375, x367); //layer3.2
	t4::free(x367, x375);
	t4::tensor4f x377 = t4::Relu(x376); //layer3.2.relu
	t4::free(x376);
	t4::tensor4f x378 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x377, ctx.layer3_3_conv1_weight); //layer3.3.conv1
	t4::tensor4f x379 = t4::BatchNormalization(x378, ctx.layer3_3_bn1_weight, ctx.layer3_3_bn1_bias, ctx.layer3_3_bn1_running_mean, ctx.layer3_3_bn1_running_var, 1e-05f); //layer3.3.bn1
	t4::free(x378);
	t4::tensor4f x380 = t4::Relu(x379); //layer3.3.relu
	t4::free(x379);
	t4::tensor4f x381 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x380, ctx.layer3_3_conv2_weight); //layer3.3.conv2
	t4::free(x380);
	t4::tensor4f x382 = t4::BatchNormalization(x381, ctx.layer3_3_bn2_weight, ctx.layer3_3_bn2_bias, ctx.layer3_3_bn2_running_mean, ctx.layer3_3_bn2_running_var, 1e-05f); //layer3.3.bn2
	t4::free(x381);
	t4::tensor4f x383 = t4::Relu(x382); //layer3.3.relu
	t4::free(x382);
	t4::tensor4f x384 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x383, ctx.layer3_3_conv3_weight); //layer3.3.conv3
	t4::free(x383);
	t4::tensor4f x385 = t4::BatchNormalization(x384, ctx.layer3_3_bn3_weight, ctx.layer3_3_bn3_bias, ctx.layer3_3_bn3_running_mean, ctx.layer3_3_bn3_running_var, 1e-05f); //layer3.3.bn3
	t4::free(x384);
	t4::tensor4f x386 = t4::Add(x385, x377); //layer3.3
	t4::free(x377, x385);
	t4::tensor4f x387 = t4::Relu(x386); //layer3.3.relu
	t4::free(x386);
	t4::tensor4f x388 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x387, ctx.layer3_4_conv1_weight); //layer3.4.conv1
	t4::tensor4f x389 = t4::BatchNormalization(x388, ctx.layer3_4_bn1_weight, ctx.layer3_4_bn1_bias, ctx.layer3_4_bn1_running_mean, ctx.layer3_4_bn1_running_var, 1e-05f); //layer3.4.bn1
	t4::free(x388);
	t4::tensor4f x390 = t4::Relu(x389); //layer3.4.relu
	t4::free(x389);
	t4::tensor4f x391 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x390, ctx.layer3_4_conv2_weight); //layer3.4.conv2
	t4::free(x390);
	t4::tensor4f x392 = t4::BatchNormalization(x391, ctx.layer3_4_bn2_weight, ctx.layer3_4_bn2_bias, ctx.layer3_4_bn2_running_mean, ctx.layer3_4_bn2_running_var, 1e-05f); //layer3.4.bn2
	t4::free(x391);
	t4::tensor4f x393 = t4::Relu(x392); //layer3.4.relu
	t4::free(x392);
	t4::tensor4f x394 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x393, ctx.layer3_4_conv3_weight); //layer3.4.conv3
	t4::free(x393);
	t4::tensor4f x395 = t4::BatchNormalization(x394, ctx.layer3_4_bn3_weight, ctx.layer3_4_bn3_bias, ctx.layer3_4_bn3_running_mean, ctx.layer3_4_bn3_running_var, 1e-05f); //layer3.4.bn3
	t4::free(x394);
	t4::tensor4f x396 = t4::Add(x395, x387); //layer3.4
	t4::free(x387, x395);
	t4::tensor4f x397 = t4::Relu(x396); //layer3.4.relu
	t4::free(x396);
	t4::tensor4f x398 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x397, ctx.layer3_5_conv1_weight); //layer3.5.conv1
	t4::tensor4f x399 = t4::BatchNormalization(x398, ctx.layer3_5_bn1_weight, ctx.layer3_5_bn1_bias, ctx.layer3_5_bn1_running_mean, ctx.layer3_5_bn1_running_var, 1e-05f); //layer3.5.bn1
	t4::free(x398);
	t4::tensor4f x400 = t4::Relu(x399); //layer3.5.relu
	t4::free(x399);
	t4::tensor4f x401 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x400, ctx.layer3_5_conv2_weight); //layer3.5.conv2
	t4::free(x400);
	t4::tensor4f x402 = t4::BatchNormalization(x401, ctx.layer3_5_bn2_weight, ctx.layer3_5_bn2_bias, ctx.layer3_5_bn2_running_mean, ctx.layer3_5_bn2_running_var, 1e-05f); //layer3.5.bn2
	t4::free(x401);
	t4::tensor4f x403 = t4::Relu(x402); //layer3.5.relu
	t4::free(x402);
	t4::tensor4f x404 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x403, ctx.layer3_5_conv3_weight); //layer3.5.conv3
	t4::free(x403);
	t4::tensor4f x405 = t4::BatchNormalization(x404, ctx.layer3_5_bn3_weight, ctx.layer3_5_bn3_bias, ctx.layer3_5_bn3_running_mean, ctx.layer3_5_bn3_running_var, 1e-05f); //layer3.5.bn3
	t4::free(x404);
	t4::tensor4f x406 = t4::Add(x405, x397); //layer3.5
	t4::free(x397, x405);
	t4::tensor4f x407 = t4::Relu(x406); //layer3.5.relu
	t4::free(x406);
	t4::tensor4f x408 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x407, ctx.layer4_0_conv1_weight); //layer4.0.conv1
	t4::tensor4f x409 = t4::BatchNormalization(x408, ctx.layer4_0_bn1_weight, ctx.layer4_0_bn1_bias, ctx.layer4_0_bn1_running_mean, ctx.layer4_0_bn1_running_var, 1e-05f); //layer4.0.bn1
	t4::free(x408);
	t4::tensor4f x410 = t4::Relu(x409); //layer4.0.relu
	t4::free(x409);
	t4::tensor4f x411 = t4::Conv2d<3, 3, 2, 2, 1, 1, 1, 1>(x410, ctx.layer4_0_conv2_weight); //layer4.0.conv2
	t4::free(x410);
	t4::tensor4f x412 = t4::BatchNormalization(x411, ctx.layer4_0_bn2_weight, ctx.layer4_0_bn2_bias, ctx.layer4_0_bn2_running_mean, ctx.layer4_0_bn2_running_var, 1e-05f); //layer4.0.bn2
	t4::free(x411);
	t4::tensor4f x413 = t4::Relu(x412); //layer4.0.relu
	t4::free(x412);
	t4::tensor4f x414 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x413, ctx.layer4_0_conv3_weight); //layer4.0.conv3
	t4::free(x413);
	t4::tensor4f x415 = t4::BatchNormalization(x414, ctx.layer4_0_bn3_weight, ctx.layer4_0_bn3_bias, ctx.layer4_0_bn3_running_mean, ctx.layer4_0_bn3_running_var, 1e-05f); //layer4.0.bn3
	t4::free(x414);
	t4::tensor4f x416 = t4::Conv2d<1, 1, 2, 2, 0, 0, 1, 1>(x407, ctx.layer4_0_downsample_0_weight); //layer4.0.downsample.0
	t4::free(x407);
	t4::tensor4f x417 = t4::BatchNormalization(x416, ctx.layer4_0_downsample_1_weight, ctx.layer4_0_downsample_1_bias, ctx.layer4_0_downsample_1_running_mean, ctx.layer4_0_downsample_1_running_var, 1e-05f); //layer4.0.downsample.1
	t4::free(x416);
	t4::tensor4f x418 = t4::Add(x415, x417); //layer4.0
	t4::free(x415, x417);
	t4::tensor4f x419 = t4::Relu(x418); //layer4.0.relu
	t4::free(x418);
	t4::tensor4f x420 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x419, ctx.layer4_1_conv1_weight); //layer4.1.conv1
	t4::tensor4f x421 = t4::BatchNormalization(x420, ctx.layer4_1_bn1_weight, ctx.layer4_1_bn1_bias, ctx.layer4_1_bn1_running_mean, ctx.layer4_1_bn1_running_var, 1e-05f); //layer4.1.bn1
	t4::free(x420);
	t4::tensor4f x422 = t4::Relu(x421); //layer4.1.relu
	t4::free(x421);
	t4::tensor4f x423 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x422, ctx.layer4_1_conv2_weight); //layer4.1.conv2
	t4::free(x422);
	t4::tensor4f x424 = t4::BatchNormalization(x423, ctx.layer4_1_bn2_weight, ctx.layer4_1_bn2_bias, ctx.layer4_1_bn2_running_mean, ctx.layer4_1_bn2_running_var, 1e-05f); //layer4.1.bn2
	t4::free(x423);
	t4::tensor4f x425 = t4::Relu(x424); //layer4.1.relu
	t4::free(x424);
	t4::tensor4f x426 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x425, ctx.layer4_1_conv3_weight); //layer4.1.conv3
	t4::free(x425);
	t4::tensor4f x427 = t4::BatchNormalization(x426, ctx.layer4_1_bn3_weight, ctx.layer4_1_bn3_bias, ctx.layer4_1_bn3_running_mean, ctx.layer4_1_bn3_running_var, 1e-05f); //layer4.1.bn3
	t4::free(x426);
	t4::tensor4f x428 = t4::Add(x427, x419); //layer4.1
	t4::free(x419, x427);
	t4::tensor4f x429 = t4::Relu(x428); //layer4.1.relu
	t4::free(x428);
	t4::tensor4f x430 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x429, ctx.layer4_2_conv1_weight); //layer4.2.conv1
	t4::tensor4f x431 = t4::BatchNormalization(x430, ctx.layer4_2_bn1_weight, ctx.layer4_2_bn1_bias, ctx.layer4_2_bn1_running_mean, ctx.layer4_2_bn1_running_var, 1e-05f); //layer4.2.bn1
	t4::free(x430);
	t4::tensor4f x432 = t4::Relu(x431); //layer4.2.relu
	t4::free(x431);
	t4::tensor4f x433 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x432, ctx.layer4_2_conv2_weight); //layer4.2.conv2
	t4::free(x432);
	t4::tensor4f x434 = t4::BatchNormalization(x433, ctx.layer4_2_bn2_weight, ctx.layer4_2_bn2_bias, ctx.layer4_2_bn2_running_mean, ctx.layer4_2_bn2_running_var, 1e-05f); //layer4.2.bn2
	t4::free(x433);
	t4::tensor4f x435 = t4::Relu(x434); //layer4.2.relu
	t4::free(x434);
	t4::tensor4f x436 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x435, ctx.layer4_2_conv3_weight); //layer4.2.conv3
	t4::free(x435);
	t4::tensor4f x437 = t4::BatchNormalization(x436, ctx.layer4_2_bn3_weight, ctx.layer4_2_bn3_bias, ctx.layer4_2_bn3_running_mean, ctx.layer4_2_bn3_running_var, 1e-05f); //layer4.2.bn3
	t4::free(x436);
	t4::tensor4f x438 = t4::Add(x437, x429); //layer4.2
	t4::free(x429, x437);
	t4::tensor4f x439 = t4::Relu(x438); //layer4.2.relu
	t4::free(x438);
	t4::tensor4f x440 = t4::AveragePool2d<7, 7, 1, 1, 0, 0>(x439); //avgpool
	t4::free(x439);
	t4::tensor2f x441 = t4::Flatten<1>(x440);
	t4::free(x440);
	t4::tensor2f x442 = t4::Linear(x441, ctx.fc_weight, ctx.fc_bias); //fc
	t4::free(x441);
	return x442;
}
