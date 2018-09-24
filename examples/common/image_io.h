#pragma once
#include "tensor4.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#pragma warning (push)
#pragma warning (disable: 4996)
#include "stb_image.h"
#include "stb_image_write.h"
#pragma warning ( pop )


namespace image_io
{
	void imwrite(t4::tensor3f in, const char* filename);
	t4::tensor3 imread(const char* filename);

	namespace details
	{
		template<typename T>
		T* HWC2CHW(T* data, int H, int W, int C)
		{
			T* out = new T[H * W * C];
			for (int c = 0; c < C; ++c)
			{
				T* out_c = out + c * W * H;
				for (int h = 0; h < H; ++h)
				{
					T* data_h = data + h * W * C;
					for (int w = 0; w < W; ++w)
					{
						*out_c = data_h[w * C + c];
						++out_c;
					}
				}
			}
			return out;
		}

		template<typename T>
		T* CHW2HWC(T* data, int H, int W, int C)
		{
			T* out = new T[H * W * C];
			for (int c = 0; c < C; ++c)
			{
				T* data_c = data + c * W * H;
				for (int h = 0; h < H; ++h)
				{
					T* out_h = out + h * W * C;
					for (int w = 0; w < W; ++w)
					{
						out_h[w * C + c]  = *data_c;
						++data_c;
					}
				}
			}
			return out;
		}

		template<typename T>
		uint8_t* TOUINT8(T* data, int size)
		{
			uint8_t* out = new uint8_t[size];
			for (int i = 0; i < size; ++i)
			{
				T x = data[i];
				if (x < 0.0f) x = 0.0f;
				if (x > 1.0f) x = 1.0f;
				out[i] = (uint8_t)(x * 255.0f );
			}
			return out;
		}

		float* TOFLOAT(uint8_t* data, int size)
		{
			float* out = new float[size];
			for (int i = 0; i < size; ++i)
			{
				uint8_t x = data[i];
				out[i] = (float)(data[i]) / 255.0f;
			}
			return out;
		}
	}
	
	t4::tensor3f imread(const char* filename)
	{
		int x, y, c;
	#pragma warning (push)
	#pragma warning (disable: 4996)
		unsigned char* data_HWC = stbi_load_from_file(fopen(filename, "rb"), &x, &y, &c, 3);
	#pragma warning ( pop )
		float* data_HWCf = details::TOFLOAT(data_HWC, y * x * 3);

		float* data = details::HWC2CHW(data_HWCf, y, x, 3);
		delete[] data_HWCf;
		free(data_HWC);
	
		t4::tensor3f input = t4::tensor4f::New({ 3, y, x }, data);
		delete[] data;
		return input;
	}
	
	void imwrite(t4::tensor3f in, const char* filename)
	{
		int channels = t4::channels(in);
		if (channels > 3)
		{
			channels = 3;
		}
		float* result_t = details::CHW2HWC((float*)in.ptr(), t4::height(in), t4::width(in), 3);
		uint8_t* result_tb = details::TOUINT8(result_t, t4::height(in) * t4::width(in) * channels);
		delete[] result_t;
		stbi_write_png(filename, t4::width(in), t4::height(in), channels, result_tb, t4::width(in) * channels);
		delete[] result_tb;
	}
}
