#include "AlexNet.h"
#include "image_io.h"
#include "imagenet_classes.h"

int main()
{
	AlexNet net = AlexNetLoad("AlexNet.bin");

	t4::tensor4f input = image_io::imread("../common/alexnet224x224_input.png").expand();

	auto classes = imagenet::load_classes("../common/classes.txt");

	float mean[] = { 0.485f, 0.456f, 0.406f };
	float std[] = { 0.229f, 0.224f, 0.225f };

	auto tmean = t4::tensor4f::New({ 1, 3, 1, 1 }, mean);
	auto tstd = t4::tensor4f::New({ 1, 3, 1, 1 }, std);

	input = (input - tmean) / tstd;

	t4::tensor2f output;
	int time = 0;

	T4_ScopeProfiler(AlexNetForward);
	output = AlexNetForward(net, input);
	time += scopeVar_AlexNetForward.GetTime();
	
	printf("Time:%d\n", time);

	output = t4::Softmax<1>(output);

	t4::tensor2i sorted = t4::Argsort(output).Flip();

	for (int i = 0; i < 10; ++i)
	{
		int c = (int)sorted.ptr()[i];
		printf("%f%%: %s\n", output.ptr()[c] * 100.0f, classes[c].c_str());
	}
}
