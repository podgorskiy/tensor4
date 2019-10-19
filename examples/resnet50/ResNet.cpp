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


t4::tensor2f ResNetForward(const ResNet& ctx, t4::tensor4f xinput_1)
{
	t4::tensor4f x321 = t4::Conv2d<7, 7, 2, 2, 3, 3, 1, 1>(xinput_1, ctx.conv1_weight); //conv1
	t4::release(xinput_1);
	t4::tensor4f x322 = t4::BatchNormalizationInplace(x321, ctx.bn1_weight, ctx.bn1_bias, ctx.bn1_running_mean, ctx.bn1_running_var, 1e-05f); //bn1
	t4::release(x321);
	t4::tensor4f x323 = t4::ReluInplace(x322); //relu
	t4::release(x322);
	t4::tensor4f x324 = t4::MaxPool2d<3, 3, 2, 2, 1, 1>(x323); //maxpool
	t4::release(x323);
	t4::tensor4f x325 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x324, ctx.layer1_0_conv1_weight); //layer1.0.conv1
	t4::tensor4f x326 = t4::BatchNormalizationInplace(x325, ctx.layer1_0_bn1_weight, ctx.layer1_0_bn1_bias, ctx.layer1_0_bn1_running_mean, ctx.layer1_0_bn1_running_var, 1e-05f); //layer1.0.bn1
	t4::release(x325);
	t4::tensor4f x327 = t4::ReluInplace(x326); //layer1.0.relu
	t4::release(x326);
	t4::tensor4f x328 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x327, ctx.layer1_0_conv2_weight); //layer1.0.conv2
	t4::release(x327);
	t4::tensor4f x329 = t4::BatchNormalizationInplace(x328, ctx.layer1_0_bn2_weight, ctx.layer1_0_bn2_bias, ctx.layer1_0_bn2_running_mean, ctx.layer1_0_bn2_running_var, 1e-05f); //layer1.0.bn2
	t4::release(x328);
	t4::tensor4f x330 = t4::ReluInplace(x329); //layer1.0.relu
	t4::release(x329);
	t4::tensor4f x331 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x330, ctx.layer1_0_conv3_weight); //layer1.0.conv3
	t4::release(x330);
	t4::tensor4f x332 = t4::BatchNormalizationInplace(x331, ctx.layer1_0_bn3_weight, ctx.layer1_0_bn3_bias, ctx.layer1_0_bn3_running_mean, ctx.layer1_0_bn3_running_var, 1e-05f); //layer1.0.bn3
	t4::release(x331);
	t4::tensor4f x333 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x324, ctx.layer1_0_downsample_0_weight); //layer1.0.downsample.0
	t4::release(x324);
	t4::tensor4f x334 = t4::BatchNormalizationInplace(x333, ctx.layer1_0_downsample_1_weight, ctx.layer1_0_downsample_1_bias, ctx.layer1_0_downsample_1_running_mean, ctx.layer1_0_downsample_1_running_var, 1e-05f); //layer1.0.downsample.1
	t4::release(x333);
	t4::tensor4f x335 = t4::Add(x332, x334); //layer1.0
	t4::release(x332, x334);
	t4::tensor4f x336 = t4::ReluInplace(x335); //layer1.0.relu
	t4::release(x335);
	t4::tensor4f x337 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x336, ctx.layer1_1_conv1_weight); //layer1.1.conv1
	t4::tensor4f x338 = t4::BatchNormalizationInplace(x337, ctx.layer1_1_bn1_weight, ctx.layer1_1_bn1_bias, ctx.layer1_1_bn1_running_mean, ctx.layer1_1_bn1_running_var, 1e-05f); //layer1.1.bn1
	t4::release(x337);
	t4::tensor4f x339 = t4::ReluInplace(x338); //layer1.1.relu
	t4::release(x338);
	t4::tensor4f x340 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x339, ctx.layer1_1_conv2_weight); //layer1.1.conv2
	t4::release(x339);
	t4::tensor4f x341 = t4::BatchNormalizationInplace(x340, ctx.layer1_1_bn2_weight, ctx.layer1_1_bn2_bias, ctx.layer1_1_bn2_running_mean, ctx.layer1_1_bn2_running_var, 1e-05f); //layer1.1.bn2
	t4::release(x340);
	t4::tensor4f x342 = t4::ReluInplace(x341); //layer1.1.relu
	t4::release(x341);
	t4::tensor4f x343 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x342, ctx.layer1_1_conv3_weight); //layer1.1.conv3
	t4::release(x342);
	t4::tensor4f x344 = t4::BatchNormalizationInplace(x343, ctx.layer1_1_bn3_weight, ctx.layer1_1_bn3_bias, ctx.layer1_1_bn3_running_mean, ctx.layer1_1_bn3_running_var, 1e-05f); //layer1.1.bn3
	t4::release(x343);
	t4::tensor4f x345 = t4::Add(x344, x336); //layer1.1
	t4::release(x336, x344);
	t4::tensor4f x346 = t4::ReluInplace(x345); //layer1.1.relu
	t4::release(x345);
	t4::tensor4f x347 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x346, ctx.layer1_2_conv1_weight); //layer1.2.conv1
	t4::tensor4f x348 = t4::BatchNormalizationInplace(x347, ctx.layer1_2_bn1_weight, ctx.layer1_2_bn1_bias, ctx.layer1_2_bn1_running_mean, ctx.layer1_2_bn1_running_var, 1e-05f); //layer1.2.bn1
	t4::release(x347);
	t4::tensor4f x349 = t4::ReluInplace(x348); //layer1.2.relu
	t4::release(x348);
	t4::tensor4f x350 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x349, ctx.layer1_2_conv2_weight); //layer1.2.conv2
	t4::release(x349);
	t4::tensor4f x351 = t4::BatchNormalizationInplace(x350, ctx.layer1_2_bn2_weight, ctx.layer1_2_bn2_bias, ctx.layer1_2_bn2_running_mean, ctx.layer1_2_bn2_running_var, 1e-05f); //layer1.2.bn2
	t4::release(x350);
	t4::tensor4f x352 = t4::ReluInplace(x351); //layer1.2.relu
	t4::release(x351);
	t4::tensor4f x353 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x352, ctx.layer1_2_conv3_weight); //layer1.2.conv3
	t4::release(x352);
	t4::tensor4f x354 = t4::BatchNormalizationInplace(x353, ctx.layer1_2_bn3_weight, ctx.layer1_2_bn3_bias, ctx.layer1_2_bn3_running_mean, ctx.layer1_2_bn3_running_var, 1e-05f); //layer1.2.bn3
	t4::release(x353);
	t4::tensor4f x355 = t4::Add(x354, x346); //layer1.2
	t4::release(x346, x354);
	t4::tensor4f x356 = t4::ReluInplace(x355); //layer1.2.relu
	t4::release(x355);
	t4::tensor4f x357 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x356, ctx.layer2_0_conv1_weight); //layer2.0.conv1
	t4::tensor4f x358 = t4::BatchNormalizationInplace(x357, ctx.layer2_0_bn1_weight, ctx.layer2_0_bn1_bias, ctx.layer2_0_bn1_running_mean, ctx.layer2_0_bn1_running_var, 1e-05f); //layer2.0.bn1
	t4::release(x357);
	t4::tensor4f x359 = t4::ReluInplace(x358); //layer2.0.relu
	t4::release(x358);
	t4::tensor4f x360 = t4::Conv2d<3, 3, 2, 2, 1, 1, 1, 1>(x359, ctx.layer2_0_conv2_weight); //layer2.0.conv2
	t4::release(x359);
	t4::tensor4f x361 = t4::BatchNormalizationInplace(x360, ctx.layer2_0_bn2_weight, ctx.layer2_0_bn2_bias, ctx.layer2_0_bn2_running_mean, ctx.layer2_0_bn2_running_var, 1e-05f); //layer2.0.bn2
	t4::release(x360);
	t4::tensor4f x362 = t4::ReluInplace(x361); //layer2.0.relu
	t4::release(x361);
	t4::tensor4f x363 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x362, ctx.layer2_0_conv3_weight); //layer2.0.conv3
	t4::release(x362);
	t4::tensor4f x364 = t4::BatchNormalizationInplace(x363, ctx.layer2_0_bn3_weight, ctx.layer2_0_bn3_bias, ctx.layer2_0_bn3_running_mean, ctx.layer2_0_bn3_running_var, 1e-05f); //layer2.0.bn3
	t4::release(x363);
	t4::tensor4f x365 = t4::Conv2d<1, 1, 2, 2, 0, 0, 1, 1>(x356, ctx.layer2_0_downsample_0_weight); //layer2.0.downsample.0
	t4::release(x356);
	t4::tensor4f x366 = t4::BatchNormalizationInplace(x365, ctx.layer2_0_downsample_1_weight, ctx.layer2_0_downsample_1_bias, ctx.layer2_0_downsample_1_running_mean, ctx.layer2_0_downsample_1_running_var, 1e-05f); //layer2.0.downsample.1
	t4::release(x365);
	t4::tensor4f x367 = t4::Add(x364, x366); //layer2.0
	t4::release(x364, x366);
	t4::tensor4f x368 = t4::ReluInplace(x367); //layer2.0.relu
	t4::release(x367);
	t4::tensor4f x369 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x368, ctx.layer2_1_conv1_weight); //layer2.1.conv1
	t4::tensor4f x370 = t4::BatchNormalizationInplace(x369, ctx.layer2_1_bn1_weight, ctx.layer2_1_bn1_bias, ctx.layer2_1_bn1_running_mean, ctx.layer2_1_bn1_running_var, 1e-05f); //layer2.1.bn1
	t4::release(x369);
	t4::tensor4f x371 = t4::ReluInplace(x370); //layer2.1.relu
	t4::release(x370);
	t4::tensor4f x372 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x371, ctx.layer2_1_conv2_weight); //layer2.1.conv2
	t4::release(x371);
	t4::tensor4f x373 = t4::BatchNormalizationInplace(x372, ctx.layer2_1_bn2_weight, ctx.layer2_1_bn2_bias, ctx.layer2_1_bn2_running_mean, ctx.layer2_1_bn2_running_var, 1e-05f); //layer2.1.bn2
	t4::release(x372);
	t4::tensor4f x374 = t4::ReluInplace(x373); //layer2.1.relu
	t4::release(x373);
	t4::tensor4f x375 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x374, ctx.layer2_1_conv3_weight); //layer2.1.conv3
	t4::release(x374);
	t4::tensor4f x376 = t4::BatchNormalizationInplace(x375, ctx.layer2_1_bn3_weight, ctx.layer2_1_bn3_bias, ctx.layer2_1_bn3_running_mean, ctx.layer2_1_bn3_running_var, 1e-05f); //layer2.1.bn3
	t4::release(x375);
	t4::tensor4f x377 = t4::Add(x376, x368); //layer2.1
	t4::release(x368, x376);
	t4::tensor4f x378 = t4::ReluInplace(x377); //layer2.1.relu
	t4::release(x377);
	t4::tensor4f x379 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x378, ctx.layer2_2_conv1_weight); //layer2.2.conv1
	t4::tensor4f x380 = t4::BatchNormalizationInplace(x379, ctx.layer2_2_bn1_weight, ctx.layer2_2_bn1_bias, ctx.layer2_2_bn1_running_mean, ctx.layer2_2_bn1_running_var, 1e-05f); //layer2.2.bn1
	t4::release(x379);
	t4::tensor4f x381 = t4::ReluInplace(x380); //layer2.2.relu
	t4::release(x380);
	t4::tensor4f x382 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x381, ctx.layer2_2_conv2_weight); //layer2.2.conv2
	t4::release(x381);
	t4::tensor4f x383 = t4::BatchNormalizationInplace(x382, ctx.layer2_2_bn2_weight, ctx.layer2_2_bn2_bias, ctx.layer2_2_bn2_running_mean, ctx.layer2_2_bn2_running_var, 1e-05f); //layer2.2.bn2
	t4::release(x382);
	t4::tensor4f x384 = t4::ReluInplace(x383); //layer2.2.relu
	t4::release(x383);
	t4::tensor4f x385 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x384, ctx.layer2_2_conv3_weight); //layer2.2.conv3
	t4::release(x384);
	t4::tensor4f x386 = t4::BatchNormalizationInplace(x385, ctx.layer2_2_bn3_weight, ctx.layer2_2_bn3_bias, ctx.layer2_2_bn3_running_mean, ctx.layer2_2_bn3_running_var, 1e-05f); //layer2.2.bn3
	t4::release(x385);
	t4::tensor4f x387 = t4::Add(x386, x378); //layer2.2
	t4::release(x378, x386);
	t4::tensor4f x388 = t4::ReluInplace(x387); //layer2.2.relu
	t4::release(x387);
	t4::tensor4f x389 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x388, ctx.layer2_3_conv1_weight); //layer2.3.conv1
	t4::tensor4f x390 = t4::BatchNormalizationInplace(x389, ctx.layer2_3_bn1_weight, ctx.layer2_3_bn1_bias, ctx.layer2_3_bn1_running_mean, ctx.layer2_3_bn1_running_var, 1e-05f); //layer2.3.bn1
	t4::release(x389);
	t4::tensor4f x391 = t4::ReluInplace(x390); //layer2.3.relu
	t4::release(x390);
	t4::tensor4f x392 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x391, ctx.layer2_3_conv2_weight); //layer2.3.conv2
	t4::release(x391);
	t4::tensor4f x393 = t4::BatchNormalizationInplace(x392, ctx.layer2_3_bn2_weight, ctx.layer2_3_bn2_bias, ctx.layer2_3_bn2_running_mean, ctx.layer2_3_bn2_running_var, 1e-05f); //layer2.3.bn2
	t4::release(x392);
	t4::tensor4f x394 = t4::ReluInplace(x393); //layer2.3.relu
	t4::release(x393);
	t4::tensor4f x395 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x394, ctx.layer2_3_conv3_weight); //layer2.3.conv3
	t4::release(x394);
	t4::tensor4f x396 = t4::BatchNormalizationInplace(x395, ctx.layer2_3_bn3_weight, ctx.layer2_3_bn3_bias, ctx.layer2_3_bn3_running_mean, ctx.layer2_3_bn3_running_var, 1e-05f); //layer2.3.bn3
	t4::release(x395);
	t4::tensor4f x397 = t4::Add(x396, x388); //layer2.3
	t4::release(x388, x396);
	t4::tensor4f x398 = t4::ReluInplace(x397); //layer2.3.relu
	t4::release(x397);
	t4::tensor4f x399 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x398, ctx.layer3_0_conv1_weight); //layer3.0.conv1
	t4::tensor4f x400 = t4::BatchNormalizationInplace(x399, ctx.layer3_0_bn1_weight, ctx.layer3_0_bn1_bias, ctx.layer3_0_bn1_running_mean, ctx.layer3_0_bn1_running_var, 1e-05f); //layer3.0.bn1
	t4::release(x399);
	t4::tensor4f x401 = t4::ReluInplace(x400); //layer3.0.relu
	t4::release(x400);
	t4::tensor4f x402 = t4::Conv2d<3, 3, 2, 2, 1, 1, 1, 1>(x401, ctx.layer3_0_conv2_weight); //layer3.0.conv2
	t4::release(x401);
	t4::tensor4f x403 = t4::BatchNormalizationInplace(x402, ctx.layer3_0_bn2_weight, ctx.layer3_0_bn2_bias, ctx.layer3_0_bn2_running_mean, ctx.layer3_0_bn2_running_var, 1e-05f); //layer3.0.bn2
	t4::release(x402);
	t4::tensor4f x404 = t4::ReluInplace(x403); //layer3.0.relu
	t4::release(x403);
	t4::tensor4f x405 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x404, ctx.layer3_0_conv3_weight); //layer3.0.conv3
	t4::release(x404);
	t4::tensor4f x406 = t4::BatchNormalizationInplace(x405, ctx.layer3_0_bn3_weight, ctx.layer3_0_bn3_bias, ctx.layer3_0_bn3_running_mean, ctx.layer3_0_bn3_running_var, 1e-05f); //layer3.0.bn3
	t4::release(x405);
	t4::tensor4f x407 = t4::Conv2d<1, 1, 2, 2, 0, 0, 1, 1>(x398, ctx.layer3_0_downsample_0_weight); //layer3.0.downsample.0
	t4::release(x398);
	t4::tensor4f x408 = t4::BatchNormalizationInplace(x407, ctx.layer3_0_downsample_1_weight, ctx.layer3_0_downsample_1_bias, ctx.layer3_0_downsample_1_running_mean, ctx.layer3_0_downsample_1_running_var, 1e-05f); //layer3.0.downsample.1
	t4::release(x407);
	t4::tensor4f x409 = t4::Add(x406, x408); //layer3.0
	t4::release(x406, x408);
	t4::tensor4f x410 = t4::ReluInplace(x409); //layer3.0.relu
	t4::release(x409);
	t4::tensor4f x411 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x410, ctx.layer3_1_conv1_weight); //layer3.1.conv1
	t4::tensor4f x412 = t4::BatchNormalizationInplace(x411, ctx.layer3_1_bn1_weight, ctx.layer3_1_bn1_bias, ctx.layer3_1_bn1_running_mean, ctx.layer3_1_bn1_running_var, 1e-05f); //layer3.1.bn1
	t4::release(x411);
	t4::tensor4f x413 = t4::ReluInplace(x412); //layer3.1.relu
	t4::release(x412);
	t4::tensor4f x414 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x413, ctx.layer3_1_conv2_weight); //layer3.1.conv2
	t4::release(x413);
	t4::tensor4f x415 = t4::BatchNormalizationInplace(x414, ctx.layer3_1_bn2_weight, ctx.layer3_1_bn2_bias, ctx.layer3_1_bn2_running_mean, ctx.layer3_1_bn2_running_var, 1e-05f); //layer3.1.bn2
	t4::release(x414);
	t4::tensor4f x416 = t4::ReluInplace(x415); //layer3.1.relu
	t4::release(x415);
	t4::tensor4f x417 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x416, ctx.layer3_1_conv3_weight); //layer3.1.conv3
	t4::release(x416);
	t4::tensor4f x418 = t4::BatchNormalizationInplace(x417, ctx.layer3_1_bn3_weight, ctx.layer3_1_bn3_bias, ctx.layer3_1_bn3_running_mean, ctx.layer3_1_bn3_running_var, 1e-05f); //layer3.1.bn3
	t4::release(x417);
	t4::tensor4f x419 = t4::Add(x418, x410); //layer3.1
	t4::release(x410, x418);
	t4::tensor4f x420 = t4::ReluInplace(x419); //layer3.1.relu
	t4::release(x419);
	t4::tensor4f x421 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x420, ctx.layer3_2_conv1_weight); //layer3.2.conv1
	t4::tensor4f x422 = t4::BatchNormalizationInplace(x421, ctx.layer3_2_bn1_weight, ctx.layer3_2_bn1_bias, ctx.layer3_2_bn1_running_mean, ctx.layer3_2_bn1_running_var, 1e-05f); //layer3.2.bn1
	t4::release(x421);
	t4::tensor4f x423 = t4::ReluInplace(x422); //layer3.2.relu
	t4::release(x422);
	t4::tensor4f x424 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x423, ctx.layer3_2_conv2_weight); //layer3.2.conv2
	t4::release(x423);
	t4::tensor4f x425 = t4::BatchNormalizationInplace(x424, ctx.layer3_2_bn2_weight, ctx.layer3_2_bn2_bias, ctx.layer3_2_bn2_running_mean, ctx.layer3_2_bn2_running_var, 1e-05f); //layer3.2.bn2
	t4::release(x424);
	t4::tensor4f x426 = t4::ReluInplace(x425); //layer3.2.relu
	t4::release(x425);
	t4::tensor4f x427 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x426, ctx.layer3_2_conv3_weight); //layer3.2.conv3
	t4::release(x426);
	t4::tensor4f x428 = t4::BatchNormalizationInplace(x427, ctx.layer3_2_bn3_weight, ctx.layer3_2_bn3_bias, ctx.layer3_2_bn3_running_mean, ctx.layer3_2_bn3_running_var, 1e-05f); //layer3.2.bn3
	t4::release(x427);
	t4::tensor4f x429 = t4::Add(x428, x420); //layer3.2
	t4::release(x420, x428);
	t4::tensor4f x430 = t4::ReluInplace(x429); //layer3.2.relu
	t4::release(x429);
	t4::tensor4f x431 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x430, ctx.layer3_3_conv1_weight); //layer3.3.conv1
	t4::tensor4f x432 = t4::BatchNormalizationInplace(x431, ctx.layer3_3_bn1_weight, ctx.layer3_3_bn1_bias, ctx.layer3_3_bn1_running_mean, ctx.layer3_3_bn1_running_var, 1e-05f); //layer3.3.bn1
	t4::release(x431);
	t4::tensor4f x433 = t4::ReluInplace(x432); //layer3.3.relu
	t4::release(x432);
	t4::tensor4f x434 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x433, ctx.layer3_3_conv2_weight); //layer3.3.conv2
	t4::release(x433);
	t4::tensor4f x435 = t4::BatchNormalizationInplace(x434, ctx.layer3_3_bn2_weight, ctx.layer3_3_bn2_bias, ctx.layer3_3_bn2_running_mean, ctx.layer3_3_bn2_running_var, 1e-05f); //layer3.3.bn2
	t4::release(x434);
	t4::tensor4f x436 = t4::ReluInplace(x435); //layer3.3.relu
	t4::release(x435);
	t4::tensor4f x437 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x436, ctx.layer3_3_conv3_weight); //layer3.3.conv3
	t4::release(x436);
	t4::tensor4f x438 = t4::BatchNormalizationInplace(x437, ctx.layer3_3_bn3_weight, ctx.layer3_3_bn3_bias, ctx.layer3_3_bn3_running_mean, ctx.layer3_3_bn3_running_var, 1e-05f); //layer3.3.bn3
	t4::release(x437);
	t4::tensor4f x439 = t4::Add(x438, x430); //layer3.3
	t4::release(x430, x438);
	t4::tensor4f x440 = t4::ReluInplace(x439); //layer3.3.relu
	t4::release(x439);
	t4::tensor4f x441 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x440, ctx.layer3_4_conv1_weight); //layer3.4.conv1
	t4::tensor4f x442 = t4::BatchNormalizationInplace(x441, ctx.layer3_4_bn1_weight, ctx.layer3_4_bn1_bias, ctx.layer3_4_bn1_running_mean, ctx.layer3_4_bn1_running_var, 1e-05f); //layer3.4.bn1
	t4::release(x441);
	t4::tensor4f x443 = t4::ReluInplace(x442); //layer3.4.relu
	t4::release(x442);
	t4::tensor4f x444 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x443, ctx.layer3_4_conv2_weight); //layer3.4.conv2
	t4::release(x443);
	t4::tensor4f x445 = t4::BatchNormalizationInplace(x444, ctx.layer3_4_bn2_weight, ctx.layer3_4_bn2_bias, ctx.layer3_4_bn2_running_mean, ctx.layer3_4_bn2_running_var, 1e-05f); //layer3.4.bn2
	t4::release(x444);
	t4::tensor4f x446 = t4::ReluInplace(x445); //layer3.4.relu
	t4::release(x445);
	t4::tensor4f x447 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x446, ctx.layer3_4_conv3_weight); //layer3.4.conv3
	t4::release(x446);
	t4::tensor4f x448 = t4::BatchNormalizationInplace(x447, ctx.layer3_4_bn3_weight, ctx.layer3_4_bn3_bias, ctx.layer3_4_bn3_running_mean, ctx.layer3_4_bn3_running_var, 1e-05f); //layer3.4.bn3
	t4::release(x447);
	t4::tensor4f x449 = t4::Add(x448, x440); //layer3.4
	t4::release(x440, x448);
	t4::tensor4f x450 = t4::ReluInplace(x449); //layer3.4.relu
	t4::release(x449);
	t4::tensor4f x451 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x450, ctx.layer3_5_conv1_weight); //layer3.5.conv1
	t4::tensor4f x452 = t4::BatchNormalizationInplace(x451, ctx.layer3_5_bn1_weight, ctx.layer3_5_bn1_bias, ctx.layer3_5_bn1_running_mean, ctx.layer3_5_bn1_running_var, 1e-05f); //layer3.5.bn1
	t4::release(x451);
	t4::tensor4f x453 = t4::ReluInplace(x452); //layer3.5.relu
	t4::release(x452);
	t4::tensor4f x454 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x453, ctx.layer3_5_conv2_weight); //layer3.5.conv2
	t4::release(x453);
	t4::tensor4f x455 = t4::BatchNormalizationInplace(x454, ctx.layer3_5_bn2_weight, ctx.layer3_5_bn2_bias, ctx.layer3_5_bn2_running_mean, ctx.layer3_5_bn2_running_var, 1e-05f); //layer3.5.bn2
	t4::release(x454);
	t4::tensor4f x456 = t4::ReluInplace(x455); //layer3.5.relu
	t4::release(x455);
	t4::tensor4f x457 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x456, ctx.layer3_5_conv3_weight); //layer3.5.conv3
	t4::release(x456);
	t4::tensor4f x458 = t4::BatchNormalizationInplace(x457, ctx.layer3_5_bn3_weight, ctx.layer3_5_bn3_bias, ctx.layer3_5_bn3_running_mean, ctx.layer3_5_bn3_running_var, 1e-05f); //layer3.5.bn3
	t4::release(x457);
	t4::tensor4f x459 = t4::Add(x458, x450); //layer3.5
	t4::release(x450, x458);
	t4::tensor4f x460 = t4::ReluInplace(x459); //layer3.5.relu
	t4::release(x459);
	t4::tensor4f x461 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x460, ctx.layer4_0_conv1_weight); //layer4.0.conv1
	t4::tensor4f x462 = t4::BatchNormalizationInplace(x461, ctx.layer4_0_bn1_weight, ctx.layer4_0_bn1_bias, ctx.layer4_0_bn1_running_mean, ctx.layer4_0_bn1_running_var, 1e-05f); //layer4.0.bn1
	t4::release(x461);
	t4::tensor4f x463 = t4::ReluInplace(x462); //layer4.0.relu
	t4::release(x462);
	t4::tensor4f x464 = t4::Conv2d<3, 3, 2, 2, 1, 1, 1, 1>(x463, ctx.layer4_0_conv2_weight); //layer4.0.conv2
	t4::release(x463);
	t4::tensor4f x465 = t4::BatchNormalizationInplace(x464, ctx.layer4_0_bn2_weight, ctx.layer4_0_bn2_bias, ctx.layer4_0_bn2_running_mean, ctx.layer4_0_bn2_running_var, 1e-05f); //layer4.0.bn2
	t4::release(x464);
	t4::tensor4f x466 = t4::ReluInplace(x465); //layer4.0.relu
	t4::release(x465);
	t4::tensor4f x467 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x466, ctx.layer4_0_conv3_weight); //layer4.0.conv3
	t4::release(x466);
	t4::tensor4f x468 = t4::BatchNormalizationInplace(x467, ctx.layer4_0_bn3_weight, ctx.layer4_0_bn3_bias, ctx.layer4_0_bn3_running_mean, ctx.layer4_0_bn3_running_var, 1e-05f); //layer4.0.bn3
	t4::release(x467);
	t4::tensor4f x469 = t4::Conv2d<1, 1, 2, 2, 0, 0, 1, 1>(x460, ctx.layer4_0_downsample_0_weight); //layer4.0.downsample.0
	t4::release(x460);
	t4::tensor4f x470 = t4::BatchNormalizationInplace(x469, ctx.layer4_0_downsample_1_weight, ctx.layer4_0_downsample_1_bias, ctx.layer4_0_downsample_1_running_mean, ctx.layer4_0_downsample_1_running_var, 1e-05f); //layer4.0.downsample.1
	t4::release(x469);
	t4::tensor4f x471 = t4::Add(x468, x470); //layer4.0
	t4::release(x468, x470);
	t4::tensor4f x472 = t4::ReluInplace(x471); //layer4.0.relu
	t4::release(x471);
	t4::tensor4f x473 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x472, ctx.layer4_1_conv1_weight); //layer4.1.conv1
	t4::tensor4f x474 = t4::BatchNormalizationInplace(x473, ctx.layer4_1_bn1_weight, ctx.layer4_1_bn1_bias, ctx.layer4_1_bn1_running_mean, ctx.layer4_1_bn1_running_var, 1e-05f); //layer4.1.bn1
	t4::release(x473);
	t4::tensor4f x475 = t4::ReluInplace(x474); //layer4.1.relu
	t4::release(x474);
	t4::tensor4f x476 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x475, ctx.layer4_1_conv2_weight); //layer4.1.conv2
	t4::release(x475);
	t4::tensor4f x477 = t4::BatchNormalizationInplace(x476, ctx.layer4_1_bn2_weight, ctx.layer4_1_bn2_bias, ctx.layer4_1_bn2_running_mean, ctx.layer4_1_bn2_running_var, 1e-05f); //layer4.1.bn2
	t4::release(x476);
	t4::tensor4f x478 = t4::ReluInplace(x477); //layer4.1.relu
	t4::release(x477);
	t4::tensor4f x479 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x478, ctx.layer4_1_conv3_weight); //layer4.1.conv3
	t4::release(x478);
	t4::tensor4f x480 = t4::BatchNormalizationInplace(x479, ctx.layer4_1_bn3_weight, ctx.layer4_1_bn3_bias, ctx.layer4_1_bn3_running_mean, ctx.layer4_1_bn3_running_var, 1e-05f); //layer4.1.bn3
	t4::release(x479);
	t4::tensor4f x481 = t4::Add(x480, x472); //layer4.1
	t4::release(x472, x480);
	t4::tensor4f x482 = t4::ReluInplace(x481); //layer4.1.relu
	t4::release(x481);
	t4::tensor4f x483 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x482, ctx.layer4_2_conv1_weight); //layer4.2.conv1
	t4::tensor4f x484 = t4::BatchNormalizationInplace(x483, ctx.layer4_2_bn1_weight, ctx.layer4_2_bn1_bias, ctx.layer4_2_bn1_running_mean, ctx.layer4_2_bn1_running_var, 1e-05f); //layer4.2.bn1
	t4::release(x483);
	t4::tensor4f x485 = t4::ReluInplace(x484); //layer4.2.relu
	t4::release(x484);
	t4::tensor4f x486 = t4::Conv2d<3, 3, 1, 1, 1, 1, 1, 1>(x485, ctx.layer4_2_conv2_weight); //layer4.2.conv2
	t4::release(x485);
	t4::tensor4f x487 = t4::BatchNormalizationInplace(x486, ctx.layer4_2_bn2_weight, ctx.layer4_2_bn2_bias, ctx.layer4_2_bn2_running_mean, ctx.layer4_2_bn2_running_var, 1e-05f); //layer4.2.bn2
	t4::release(x486);
	t4::tensor4f x488 = t4::ReluInplace(x487); //layer4.2.relu
	t4::release(x487);
	t4::tensor4f x489 = t4::Conv2d<1, 1, 1, 1, 0, 0, 1, 1>(x488, ctx.layer4_2_conv3_weight); //layer4.2.conv3
	t4::release(x488);
	t4::tensor4f x490 = t4::BatchNormalizationInplace(x489, ctx.layer4_2_bn3_weight, ctx.layer4_2_bn3_bias, ctx.layer4_2_bn3_running_mean, ctx.layer4_2_bn3_running_var, 1e-05f); //layer4.2.bn3
	t4::release(x489);
	t4::tensor4f x491 = t4::Add(x490, x482); //layer4.2
	t4::release(x482, x490);
	t4::tensor4f x492 = t4::ReluInplace(x491); //layer4.2.relu
	t4::release(x491);
	t4::tensor4f x493 = t4::GlobalAveragePool2d(x492); //avgpool
	t4::release(x492);
	t4::tensor0i x494 = t4::Constant<t4::int64>(0);
	t4::tensor1i x495 = t4::Shape(x493);
	t4::tensor0i x496 = t4::Gather(x495, x494);
	t4::release(x495, x494);
	t4::tensor0i x497 = t4::Constant<t4::int64>(-1);
	t4::tensor1i x498 = t4::Unsqueeze<0>(x496);
	t4::release(x496);
	t4::tensor1i x499 = t4::Unsqueeze<0>(x497);
	t4::release(x497);
	t4::tensor1i x500 = t4::Concat<0>(x498, x499);
	t4::release(x498, x499);
	t4::tensor2f x501 = t4::Reshape<2>(x493, x500);
	t4::release(x493, x500);
	t4::tensor2f x502 = t4::Linear(x501, ctx.fc_weight, ctx.fc_bias); //fc
	t4::release(x501);
	return x502;
}
