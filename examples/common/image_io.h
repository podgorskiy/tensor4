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

	t4::tensor3f imread(const char* filename);

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

		static constexpr const char* base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
		static constexpr uint8_t base64_inverse[256] = {
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,62,128,62,128,63,
			52,53,54,55,56,57,58,59,60,61,128,128,128,128,128,128,
			128,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
			15,16,17,18,19,20,21,22,23,24,25,128,128,128,128,63,
			128,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
			41,42,43,44,45,46,47,48,49,50,51,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
			128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128
		};
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
	
		t4::tensor3f input = t4::tensor3f::New({ 3, y, x }, data);
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

	inline size_t base64_get_output_buffer_size(size_t src_len)
	{
		size_t total_block_count = (src_len + 2) / 3;
		return total_block_count * 4;
	}

	inline void base64_encode(const uint8_t* src, char* dst, size_t src_len)
	{
		for (const uint8_t* end = src + (src_len / 3) * 3; src < end; dst += 4, src += 3)
		{
			dst[0] = details::base64_chars[(src[0] & 0xfc) >> 2];
			dst[1] = details::base64_chars[(src[0] & 0x03) << 4 | (src[1] & 0xf0) >> 4];
			dst[2] = details::base64_chars[(src[1] & 0x0f) << 2 | (src[2] & 0xc0) >> 6];
			dst[3] = details::base64_chars[src[2] & 0x3f];
		}

		switch (src_len % 3)
		{
		case 1:
			dst[0] = details::base64_chars[(src[0] & 0xfc) >> 2];
			dst[1] = details::base64_chars[(src[0] & 0x03) << 4];
			dst[2] = '=';
			dst[3] = '=';
			break;
		case 2:
			dst[0] = details::base64_chars[(src[0] & 0xfc) >> 2];
			dst[1] = details::base64_chars[(src[0] & 0x03) << 4 | (src[1] & 0xf0) >> 4];
			dst[2] = details::base64_chars[(src[1] & 0x0f) << 2];
			dst[3] = '=';
			break;
		case 0:
			break;
		}
	}

	inline void base64_decode(const char* src, uint8_t* dst, size_t src_len, size_t& dst_len)
	{
		union
		{
			uint32_t val = 0;
			uint8_t data[4];
		};
		int count = 0;
		const uint8_t* dst_beg = dst;
		const char* src_end = src + src_len;

		while (src != src_end && *src != '\0' && *src != '=')
		{
			char c = *src++;
			uint8_t r = details::base64_inverse[c];
			if (r != 128)
			{
				val = (val << 6) | r;
				if (++count == 4)
				{
					dst[0] = data[2];
					dst[1] = data[1];
					dst[2] = data[0];
					dst += 3;
					count = 0;
				}
			}
		}
		val = val << (6 * ((4 - count) % 4));
		int pad = 0;
		while (src != src_end && *src != '\0' && pad < 3)
		{
			pad += *src == '=';
			if (details::base64_inverse[*src] != 128)
			{
				break;
			}
			++src;
		}
		if ((count + pad) % 4  != 0)
		{
			dst_len = 0;
			return;
		}
		switch (pad)
		{
		case 0:
			break;
		case 1:
			*dst++ = data[2];
			*dst++ = data[1];
			break;
		case 2:
			*dst++ = data[2];
			break;
		default:
			dst_len = 0;
			return;
		}
		dst_len = dst - dst_beg;
	}

	inline std::string base64_encode(const uint8_t* src, size_t src_len)
	{
		std::string data;
		data.resize(base64_get_output_buffer_size(src_len));
		base64_encode(src, &data[0], src_len);
		return data;
	}

	inline std::string base64_encode(const uint8_t* src, size_t src_len, const char* prefix)
	{
		std::string data;
		size_t prefix_length = strlen(prefix);
		data.resize(base64_get_output_buffer_size(src_len) + prefix_length);
		char* dst = &data[0];
		memcpy(dst, prefix, prefix_length);
		dst += prefix_length;
		base64_encode(src, dst, src_len);
		return data;
	}

	inline void write_to_string(void *context, void *data, int size)
	{
		std::string* str = (std::string*)context;
		int old_size = str->size();
		str->resize(old_size + size);
		memcpy(&(*str)[old_size], data, size);
	}

	inline std::string imwrite_to_base64(t4::tensor3f in)
	{
		std::string str;
		int channels = in.shape()[0];
		if (channels > 3)
		{
			channels = 3;
		}
		if (channels != 1)
		{
			float* result_t = details::CHW2HWC((float*)in.ptr(), t4::height(in), t4::width(in), 3);
			uint8_t* result_tb = details::TOUINT8(result_t, t4::height(in) * t4::width(in) * channels);
			delete[] result_t;
			stbi_write_png_to_func(write_to_string, &str, t4::width(in), t4::height(in), channels, result_tb, t4::width(in) * channels);
			delete[] result_tb;
		}
		else
		{
			uint8_t* result_tb = details::TOUINT8((float*)in.ptr(), t4::height(in) * t4::width(in));
			stbi_write_png_to_func(write_to_string, &str, t4::width(in), t4::height(in), 1, result_tb, t4::width(in));
			delete[] result_tb;
		}
		return base64_encode((const uint8_t*)str.c_str(), str.size(), "data:image/png;base64,");
	}
}
