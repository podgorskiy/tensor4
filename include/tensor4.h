// Copyright 2018 Stanislav Pidhorskyi.All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
#pragma once

#include <memory>
#include <array>
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <map>

#include <malloc.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

//#define USE_MKLDNN

#ifdef USE_MKLDNN
#include <mkl_cblas.h>
#endif

#if !defined(T4_USE_OMP)
#if defined(_OPENMP)
#define T4_USE_OMP 1
#else
#define T4_USE_OMP 0
#endif
#endif

#if T4_USE_OMP
#include <omp.h>
#endif

//#define T4_DO_TIME_PROFILING

#ifdef T4_DO_TIME_PROFILING
#define T4_ScopeProfiler(name) ::t4::ScopeProfiler scopeVar_##name(#name);
#else
#define T4_ScopeProfiler(X)
#endif

#if T4_USE_OMP
#define OMP_THREAD_ID omp_get_thread_num()
#define OMP_MAX_THREADS omp_get_max_threads()
#else
#define OMP_THREAD_ID 0
#define OMP_MAX_THREADS 1
#endif


namespace t4
{
	typedef int64_t int64;

	class ScopeProfiler
	{
	public:
		ScopeProfiler(const char* name)
		{
			m_name = name;
			m_startTime = std::chrono::high_resolution_clock::now();
		};

		~ScopeProfiler()
		{
			auto diffrence = std::chrono::high_resolution_clock::now() - m_startTime;
			unsigned long long diffrence_us = std::chrono::duration_cast<std::chrono::microseconds>(diffrence).count();
			int d = static_cast<int>(diffrence_us);
			printf("%-20s: %8dus\n", m_name, d);
		};

	private:
		const char* m_name;
		std::chrono::high_resolution_clock::time_point m_startTime;
	};

	namespace memory
	{
		enum {
			PAGE_4K = 4096,
			BLOCK_SIZE = 128
		};

		inline void* aligned_malloc(size_t size, int alignment)
		{
#ifdef _MSC_VER
			return _aligned_malloc(size, alignment);
#elif __MINGW32__
			return __mingw_aligned_malloc(size, alignment);
#else
			void* ptr = nullptr;
			int result = posix_memalign(&ptr, alignment, size);
			return (result == 0) ? ptr : nullptr;
#endif
		}

		inline void aligned_free(void *p)
		{
#ifdef _MSC_VER
			_aligned_free(p);
#else
			free(p);
#endif
		}
	}

	// Templated class for remresenting n-dimentional tensor
	// template args:
	//     T - datatype. Should be any of: float, double, int, int64_t, int32_t, int16_t
	//     D - number of dimentions
	template<typename T, int D>
	class tensor
	{
		template<typename T2, int D2>
		friend class tensor;
	public:
		enum
		{
			ndim = D
		};

		// Creates a tensor of given shape and copies data from provided raw data pointer
		static tensor<T, D> New(const std::array<int, D>& shape, T* data)
		{
			tensor<T, D> t = New(shape);
			memcpy(t.ptr(), data, (size_t)t.size()* sizeof(T));
			return t;
		}

		// Creates a tensor of given shape and takes provided shared pointer. 
		// If shared pointer is null, will initialize tensor with zeros
		static tensor<T, D> New(const std::array<int, D>& shape, std::shared_ptr<T> data = nullptr, int64 offset = 0)
		{
			tensor<T, D> t;
			t.m_ptr = data;
			t.m_shape = shape;
			t.m_offset = offset;
			if (data == nullptr)
			{
				t.m_ptr.reset(new T[(size_t)t.size()]);
				memset(t.ptr(), 0, (size_t)t.size() * sizeof(T));
			}
			return t;
		}

		// Creates a new tensor with the shape of this tensor.
		tensor<T, D> SameAs() const
		{
			return New(shape(), new T[(size_t)size()]);
		}

		// Fills tensor with specified value.
		// Arguments:
		//     value - the value to fill with.
		void Fill(T value)
		{
			T* x = ptr();
			int64 i = 0;
			for (int64 l = size(); i < l - 4; i += 4)
			{
				x[i + 0] = value;
				x[i + 1] = value;
				x[i + 2] = value;
				x[i + 3] = value;
			}

			for (int64 l = size(); i < l; ++i)
				x[i] = value;
		}

		// Copies data from provided tensor
		void Assign(tensor<T, D> x)
		{
			T* dst = ptr();
			const T* src = x.ptr();
			memcpy(dst, src, size() * sizeof(T));
		}

		// Returns contiguous, one-dimentional reference tensor. Does not copy memory, 
		// e.g. all modification to the flattened tensor will also affect this tensor 
		tensor<T, 1> Contiguous() const
		{
			tensor<T, 1> t;
			t.m_ptr = m_ptr;
			t.m_offset = m_offset;
			t.m_shape[0] = size();
			return t;
		}

		// Returns new, flattened into a 2D matrix tensor.
		// Arguments:
		//     d - indicates up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. 
		tensor<T, 2> Flatten(int d) const
		{
			int64 sizeA = 1;
			int64 sizeB = 1;
			for (int i = 0; i < d; ++i)
			{
				sizeA *= m_shape[i];
			}
			for (int i = d; i < ndim; ++i)
			{
				sizeB *= m_shape[i];
			}
			tensor<T, 2> t = tensor<T, 2>::New({ (int)sizeA, (int)sizeB });
			memcpy(t.ptr(), ptr(), (size_t)t.size() * sizeof(T));
			return t;
		}

		// Returns a reference sub tensor. Subtensor will have one axis less compared to this tensor.
		// Does not copy memory, e.g. all modification to the sub tensor will also affect this tensor 
		// Arguments:
		//     n - indicates the index of a slice of outermost axis from which the subtensor should be taken
		tensor<T, D - 1> Sub(int64 n) const
		{
			tensor<T, D - 1> t;
			t.m_ptr = m_ptr;
			for (int i = 1; i < ndim; ++i)
			{
				t.m_shape[i - 1] = m_shape[i];
			}
			t.m_offset = m_offset + n * t.size();
			return t;
		}

		// Returns a reference sub tensor. Subtensor will have two axis less compared to this tensor.
		// Does not copy memory, e.g. all modification to the sub tensor will also affect this tensor 
		// Arguments:
		//     n1 - indicates the index of a slice of outermost axis
		//     n2 - indicates the index of a slice of next axis
		tensor<T, D - 2> Sub(int64 n1, int64 n2) const
		{
			tensor<T, D - 2> t;
			t.m_ptr = m_ptr;
			for (int i = 2; i < ndim; ++i)
			{
				t.m_shape[i - 2] = m_shape[i];
			}
			t.m_offset = m_offset + n1 * t.size() * m_shape[1] + n2 * t.size();
			return t;
		}

		tensor<T, D + 1> expand()
		{
			std::array<int, D + 1> new_shape;
			new_shape[0] = 1;
			for (int i = 0; i < D; ++i)
			{
				new_shape[i + 1] = m_shape[i];
			}
			return tensor<T, D + 1>::New(new_shape, m_ptr, m_offset);
		}

		// Returns a new tensor of the indices that would sort an array along the given axis.
		// Default axis is -1, which means the last axis.
		tensor<int64, D> Argsort(int axis = -1) const
		{
			assert(axis == -1 || axis < ndim);
			axis = (axis == -1) ? ndim - 1 : axis;

			tensor<int64, D> indixes = tensor<int64, D>::New(shape());
			int64 elementCount = size();
			int64 count = shape()[axis];
			int64 sortInstances = elementCount / count;
			int64 stride = 1;
			if (axis != -1)
			{
				for (int i = axis + 1; i < ndim; ++i)
				{
					stride *= m_shape[i];
				}
			}

			int64* indixesPtr = indixes.ptr();
			const T* dataPtr = ptr();

			int threads_n = OMP_MAX_THREADS;
			const size_t size_per_thr = ((count * sizeof(int64) + memory::PAGE_4K - 1) / memory::PAGE_4K) * memory::PAGE_4K;
			int64 *copy_buffers = (int64*)memory::aligned_malloc(threads_n * size_per_thr, memory::PAGE_4K);

#if T4_USE_OMP
#pragma omp parallel for
#endif
			for (int i = 0; i < sortInstances; ++i)
			{
				int thread_id = OMP_THREAD_ID;
				int64 *copy_buff = copy_buffers + size_per_thr / sizeof(T) * thread_id;
				for (int j = 0; j < count; ++j)
				{
					copy_buff[j] = j;
				}
				const T* start = dataPtr + (i / stride) * count * stride + (i % stride);
				std::sort(copy_buff, copy_buff + count, [start, stride](int64 i1, int64 i2) {return start[i1 * stride] < start[i2 * stride]; });
				for (int j = 0; j < count; ++j)
				{
					indixesPtr[(i / stride) * count * stride + (i % stride) + j * stride] = copy_buff[j];
				}
			}
			memory::aligned_free(copy_buffers);
			return indixes;
		}

		// Returns a new tensor with element flipped along the given axis. 
		// Default axis is -1, which means the last axis.
		tensor<int64, D> Flip(int axis = -1) const
		{
			if (axis < 0)
			{
				axis += ndim;
			}
			assert(axis > 0 && axis < ndim);

			tensor<T, D> output = tensor<T, D>::New(shape());
			int64 elementCount = size();
			int64 count = shape()[axis];
			int64 flipInstances = elementCount / count;
			int64 stride = 1;
			if (axis != -1)
			{
				for (int i = axis + 1; i < ndim; ++i)
				{
					stride *= m_shape[i];
				}
			}

			int64* dst = output.ptr();
			const T* src = ptr();

#if T4_USE_OMP
#pragma omp parallel for
#endif
			for (int i = 0; i < flipInstances / stride; ++i)
			{
				T* dstp = dst + i * count * stride;
				const T* srcp = src + i * count * stride;
				for (int j = 0; j < count; ++j)
				{
					memcpy(dstp + j * stride, srcp + (count - 1 - j) * stride, stride * sizeof(T));
				}

			}

			return output;
		}
		
		// Returns a raw pointer to the data
		T* ptr()
		{
			return (T*)(m_ptr.get() + m_offset);
		}

		// Returns a raw const pointer to the data
		const T* ptr() const
		{
			return (T*)(m_ptr.get() + m_offset);
		}

		// Returns number of elements in the tensor. It is equal to the product of dimentions of all axis.
		int64 size() const
		{
			int64 size = 1;
			for (int i = 0; i < ndim; ++i)
			{
				size *= m_shape[i];
			}
			return size;
		}

		// Returns the shape of the tensor
		const std::array<int, D>& shape() const
		{
			return m_shape;
		}

	private:
		std::shared_ptr<T> m_ptr;
		std::array<int, D> m_shape;

		// Offset of the data of the tensor. The data the tensor starts from m_ptr.get() + m_offset.
		// It is used to create subtensors (which are effectivly slices) as references.
		int64 m_offset = 0;
	};

	template<typename T, int D>
	inline int width(const tensor<T, D>& t)
	{
		return t.shape()[D - 1];
	}

	template<typename T, int D>
	inline int height(const tensor<T, D>& t)
	{
		return t.shape()[D - 2];
	}

	template<typename T, int D>
	inline int channels(const tensor<T, D>& t)
	{
		return t.shape()[1];
	}

	template<typename T, int D>
	inline int number(const tensor<T, D>& t)
	{
		return t.shape()[0];
	}

	template<typename T, int D>
	inline tensor<int64, D> Argsort(const tensor<T, D>& x, int axis = -1)
	{
		return x.Argsort(axis);
	}

	typedef tensor<double, 4> tensor4d;
	typedef tensor<double, 3> tensor3d;
	typedef tensor<double, 2> tensor2d;
	typedef tensor<double, 1> tensor1d;
	typedef tensor<float, 4> tensor4f;
	typedef tensor<float, 3> tensor3f;
	typedef tensor<float, 2> tensor2f;
	typedef tensor<float, 1> tensor1f;
	typedef tensor<int64, 4> tensor4i;
	typedef tensor<int64, 3> tensor3i;
	typedef tensor<int64, 2> tensor2i;
	typedef tensor<int64, 1> tensor1i;

	namespace data_loading
	{
		template<typename T>
		inline bool check_type(const std::string& str)
		{
			return false;
		}

		template<>
		inline bool check_type<float>(const std::string& str)
		{
			return str == "float";
		}

		template<>
		inline bool check_type<double>(const std::string& str)
		{
			return str == "doubl";
		}

		template<>
		inline bool check_type<int32_t>(const std::string& str)
		{
			return str == "int32";
		}

		template<>
		inline bool check_type<int16_t>(const std::string& str)
		{
			return str == "int16";
		}

		// Reinterpret cast of shared pointer
		template< class T, class U >
		inline std::shared_ptr<T> reinterpret_pointer_cast(const std::shared_ptr<U>& r)
		{
			auto p = reinterpret_cast<typename std::shared_ptr<T>::element_type*>(r.get());
			return std::shared_ptr<T>(r, p);
		}

		inline size_t get_size(const std::string& type)
		{
			if (type == "float")
			{
				return 4;
			}
			else if (type == "doubl")
			{
				return 6;
			}
			else if (type == "int32")
			{
				return 4;
			}
			else if (type == "int16")
			{
				return 2;
			}
			return 0;
		}
	}

	// Holds parameters of the network.
	// Contains a map that associates weight names with entries.
	// Entries have type, number of dimentions and shape. 
	// Those properties are checked when the load method is used to create new weight tensor.
	class model_dict
	{
	private:
		struct Entry
		{
			std::string type;
			int ndim;
			uint32_t shape[4];
			std::shared_ptr<uint8_t> ptr;
		};

		std::map<std::string, Entry> m_parameters;

	public:
		template<typename T>
		void load(tensor<T, 4>& t, const char* name, int n, int c, int h, int w)
		{
			Entry entry = m_parameters[name];
			assert(entry.shape[0] == n);
			assert(entry.shape[1] == c);
			assert(entry.shape[2] == h);
			assert(entry.shape[3] == w);
			assert(data_loading::check_type<T>(entry.type));

			t = tensor<T, 4>::New({ n, c, h, w }, data_loading::reinterpret_pointer_cast<T>(entry.ptr));
		}

		template<typename T>
		void load(tensor<T, 3>& t, const char* name, int c, int h, int w)
		{
			Entry entry = m_parameters[name];
			assert(entry.shape[0] == c);
			assert(entry.shape[1] == h);
			assert(entry.shape[2] == w);
			assert(data_loading::check_type<T>(entry.type));

			t = tensor<T, 3>::New({ w }, data_loading::reinterpret_pointer_cast<T>(entry.ptr));
		}

		template<typename T>
		void load(tensor<T, 2>& t, const char* name, int h, int w)
		{
			Entry entry = m_parameters[name];
			assert(entry.shape[0] == h);
			assert(entry.shape[1] == w);
			assert(data_loading::check_type<T>(entry.type));

			t = tensor<T, 2>::New({ h, w }, data_loading::reinterpret_pointer_cast<T>(entry.ptr));
		}

		template<typename T>
		void load(tensor<T, 1>& t, const char* name, int w)
		{
			Entry entry = m_parameters[name];
			assert(entry.shape[0] == w);
			assert(data_loading::check_type<T>(entry.type));

			t = tensor<T, 1>::New({ w }, data_loading::reinterpret_pointer_cast<T>(entry.ptr));
		}

		void add_parameter(const std::string& name, const std::string& type, int ndim, uint32_t shape[4], std::shared_ptr<uint8_t> ptr)
		{
			m_parameters[name] = Entry({ type, ndim, shape[0], shape[1], shape[2], shape[3], ptr });
		}
	};

	// Loads network parameters from file. Creates new model_dict.
	inline model_dict load(const std::string& filename)
	{
		T4_ScopeProfiler(loading_time);
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4996)
#endif
		FILE* file = fopen(filename.c_str(), "rb");
#ifdef _MSC_VER
#pragma warning ( pop )
#endif
		fseek(file, 0L, SEEK_END);
		size_t file_size = ftell(file);
		fseek(file, 0L, SEEK_SET);

		model_dict md;

		while (true)
		{
			std::string weight_name;
			char buff;
			while (fread(&buff, 1, 1, file), buff != 0)
			{
				weight_name.push_back(buff);
			}
			char type[6];
			fread(&type, 1, 5, file);
			type[5] = 0;
			std::string type_str = type;
			uint8_t ndim = 0;
			fread(&ndim, 1, 1, file);

			uint32_t shape[4] = { 1, 1, 1, 1 };
			for (int i = 0; i < ndim; ++i)
			{
				fread(&shape[i], 1, 4, file);
			}

			std::shared_ptr<uint8_t> ptr;
			size_t param_size = data_loading::get_size(type) * shape[0] * shape[1] * shape[2] * shape[3];
			ptr.reset(new uint8_t[param_size]);
			fread(ptr.get(), param_size, 1, file);
			md.add_parameter(weight_name, type, ndim, shape, ptr);

			if (ftell(file) == file_size)
			{
				break;
			}
		}

		return md;
	}

	namespace details
	{
		template<typename T>
		inline void do_block(int LDA, int LDB, int LDC, int M, int N, int K, T* __restrict A, T* __restrict B, T* __restrict C, T* __restrict Bcopy)
		{
			int j, l, i;
			for (j = 0; j < N; ++j)
			{
				for (l = 0; l < K; ++l)
				{
					memcpy(Bcopy + l, B + j + l * LDB, sizeof(T));
				}
				for (i = 0; i < M; ++i)
				{
					T* __restrict _A = A + i * LDA;
					register float cij = C[j + i * LDC];
					for (l = 0; l < K; ++l)
					{
						cij += _A[l] * Bcopy[l];
					}
					C[j + i * LDC] = cij;
				}
			}
		}

		template<typename T>
		inline void do_block_nt(int LDA, int LDB, int LDC, int M, int N, int K, T* __restrict A, T* __restrict B, T* __restrict C)
		{
			int j, l, i;
			for (j = 0; j < N; ++j)
			{
				for (i = 0; i < M; ++i)
				{
					float cij = C[j + i * LDC];
					for (l = 0; l < K; ++l)
					{
						cij += (A + i * LDA)[l] * (B + j * LDB)[l];
					}
					memcpy(C + j + i * LDC, &cij, sizeof(T));
				}
			}
		}

		template<typename T1, typename T2>
		inline T2 min(T1 a, T2 b)
		{
			return a < b ? a : b;
		}

		// A: M x K
		// B: K x N
		// C: M x N
		template<typename T>
		inline void gemm_nn(int M, int N, int K, T* A, int LDA, T* B, int LDB, T* C, int LDC)
		{
#ifdef USE_MKLDNN
			float alpha = 1.0f;
			float betta = 1.0f;
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, K, alpha, B, LDB, A, LDA, betta, C, LDC);
			return;
#endif

			int threads_n = OMP_MAX_THREADS;
			const size_t elems_per_thr = K;
			const size_t size_per_thr = ((memory::BLOCK_SIZE * sizeof(T) + memory::PAGE_4K - 1) / memory::PAGE_4K) * memory::PAGE_4K;
			T *copy_buffers = (T*)memory::aligned_malloc(threads_n * size_per_thr, memory::PAGE_4K);

#if T4_USE_OMP
#pragma omp parallel for
#endif
			for (int j = 0; j < N; j += memory::BLOCK_SIZE)
			{
				int thread_id = OMP_THREAD_ID;

				int _N = min(memory::BLOCK_SIZE, N - j);
				for (int i = 0; i < M; i += memory::BLOCK_SIZE)
				{
					int _M = min(memory::BLOCK_SIZE, M - i);
					for (int l = 0; l < K; l += memory::BLOCK_SIZE)
					{
						int _K = min(memory::BLOCK_SIZE, K - l);
						do_block(LDA, LDB, LDC, _M, _N, _K, A + l + i * LDA, B + j + l * LDB, C + j + i*LDC, copy_buffers + size_per_thr / sizeof(T) * thread_id);
					}
				}
			}
			memory::aligned_free(copy_buffers);
		}

		template<typename T>
		inline void gemm_nt(int M, int N, int K, T* A, int LDA, T* B, int LDB, T* C, int LDC)
		{
#if T4_USE_OMP
#pragma omp parallel for
#endif
			for (int j = 0; j < N; j += memory::BLOCK_SIZE)
			{
				int _N = min(memory::BLOCK_SIZE, N - j);
				for (int i = 0; i < M; i += memory::BLOCK_SIZE)
				{
					int _M = min(memory::BLOCK_SIZE, M - i);
					for (int l = 0; l < K; l += memory::BLOCK_SIZE)
					{
						int _K = min(memory::BLOCK_SIZE, K - l);

						do_block_nt(LDA, LDB, LDC, _M, _N, _K, A + l + i * LDA, B + l + j * LDB, C + j + i*LDC);
					}
				}
			}
		}

		// Performs memory copy of elements of size sizeof(T) bytes with stride 
		// src_stride * sizeof(T) bytes from src buffer to dst buffer.
		// Is used for creating more generalized code, since when src_stride is 1
		// it will substitute template specialization with single call to memcpy
		template<int src_stride, typename T>
		struct memcpy_extended
		{
			static void strided(T* __restrict dst, const T* __restrict src, int count)
			{
				int x = 0;
				for (x = 0; x < count; x += 4)
				{
					memcpy(dst + x + 0, src + (x + 0) * src_stride, sizeof(T));
					memcpy(dst + x + 1, src + (x + 1) * src_stride, sizeof(T));
					memcpy(dst + x + 2, src + (x + 2) * src_stride, sizeof(T));
					memcpy(dst + x + 3, src + (x + 3) * src_stride, sizeof(T));
				}
				for (; x < count; ++x)
				{
					memcpy(dst + x, src + x * src_stride, sizeof(T));
				}
			}
		};

		// Specialization of the above function for the case when src_stride is 1
		template<typename T>
		struct memcpy_extended<1, T>
		{
			static void strided(T* __restrict dst, const T* __restrict src, int count)
			{
				memcpy(dst, src, sizeof(T) * count);
			}
		};

		// padding non-zero.
		template<int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, typename T>
		struct im2col_process_row
		{
			static void apply(T* __restrict dst, const T* __restrict src, int fh, int fw, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
			{
				int start = (pad_w - fw * dilation_w + stride_w - 1) / stride_w;
				int end = (inputWidth + pad_w - fw * dilation_w + stride_w - 1) / stride_w;
				int start_clipped = std::max(start, 0);
				int end_clipped = std::min(end, outputWidth);

				for (int y = 0; y < outputHeight; ++y)
				{
					int input_y = y * stride_h + fh * dilation_h - pad_h;
					if (input_y >= 0 && input_y < inputHeight)
					{
						if (start > 0)
						{
							memset(dst + y * outputWidth, 0, sizeof(T)*(size_t)start);
						}
						if (end < outputWidth)
						{
							memset(dst + y * outputWidth + end, 0, sizeof(T)*(size_t)(outputWidth - end));
						}

						memcpy_extended<stride_w, T>::strided(dst + y * outputWidth + start_clipped, src + input_y * inputWidth + start_clipped * stride_w + fw * dilation_w - pad_w, end_clipped - start_clipped);
					}
					else
					{
						memset(dst + y * outputWidth, 0, sizeof(T)*(size_t)outputWidth);
					}
				}
			}
		};

		// padding zero.
		template<int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h, int dilation_w, typename T>
		struct im2col_process_row<kernel_h, kernel_w, stride_h, stride_w, 0, 0, dilation_h, dilation_w, T>
		{
			static void apply(T* __restrict dst, const T* __restrict src, int fh, int fw, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
			{
				for (int y = 0; y < outputHeight; ++y)
				{
					int input_y = y * stride_h + fh * dilation_h;
					memcpy_extended<stride_w, T>::strided(dst + y * outputWidth, src + input_y * inputWidth + fw * dilation_w, outputWidth);
				}
			}
		};

		template<int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, typename T>
		inline void im2col(
			T* __restrict output,
			const T* __restrict input,
			int channels,
			int inputWidth,
			int inputHeight,
			int outputWidth,
			int outputHeight)
		{
			int64 channel_stride_in = (int64)inputHeight * inputWidth;
			int64 channel_stride_out = (int64)outputHeight * outputWidth;
			int column_size = channels * kernel_h * kernel_w;

#pragma omp parallel for
			for (int row = 0; row < column_size; ++row)
			{
				int channel = row / kernel_h / kernel_w;
				int fh = (row / kernel_w) % kernel_h;
				int fw = row % kernel_w;

				T* __restrict dst = output + row * channel_stride_out;
				const T* __restrict src = input + channel * channel_stride_in;

				im2col_process_row<kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, T>::apply(dst, src, fh, fw, inputWidth, inputHeight, outputWidth, outputHeight);
			}
		}


		template<int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, typename T>
		inline void col2im(
			T* __restrict output,
			const T* __restrict input,
			int channels,
			int inputWidth,
			int inputHeight,
			int outputWidth,
			int outputHeight)
		{
			int64 channel_stride_in = (int64)inputHeight * inputWidth;
			int64 channel_stride_out = (int64)outputHeight * outputWidth;
			int column_size = channels * kernel_h * kernel_w;
			for (int row = 0; row < column_size; ++row)
			{
				int channel = row / kernel_h / kernel_w;
				int fh = (row / kernel_w) % kernel_h;
				int fw = row % kernel_w;

				int start = (pad_w - fw * dilation_w + stride_w - 1) / stride_w;
				int end = (inputWidth + pad_w - fw * dilation_w + stride_w - 1) / stride_w;

				T* __restrict dst = output + channel * channel_stride_in;
				const T* __restrict src = input + row * channel_stride_out;

				for (int y = 0; y < outputHeight; ++y)
				{
					int input_y = y * stride_h + fh * dilation_h - pad_h;

					if (input_y >= 0 && input_y < inputHeight)
					{
						for (int x = start; x < end; ++x)
						{
							int input_x = x * stride_w + fw * dilation_w - pad_w;
							dst[input_y * inputWidth + input_x] += src[y * outputWidth + x];
						}
					}
				}
			}
		}
	}

	template<int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, typename T>
	inline tensor<T, 4> Conv2d(
		  tensor<T, 4> in
		, tensor<T, 4> kernel
		, tensor<T, 1> bias = tensor<T, 1>())
	{
		T4_ScopeProfiler(Conv2d);
		assert(channels(kernel) == channels(in));
		assert(kernel_h == height(kernel));
		assert(kernel_w == width(kernel));

		const int N = number(in);
		const int K = number(kernel);
		const int C = channels(kernel);
		const int Hin = height(in);
		const int Win = width(in);

		const int Hout = (Hin + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
		const int Wout = (Win + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

		T* __restrict columns = (T*)malloc(C * kernel_h * kernel_w * Hout * Wout * sizeof(T));

		details::im2col<kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w>(columns, in.ptr(), channels(in), Win, Hin, Wout, Hout);

		tensor<T, 4> out = tensor<T, 4>::New({ N, K, Hout, Wout });

		if (bias.ptr() != nullptr)
		{
			T* pbias = bias.ptr();
			for (int n = 0; n < N; ++n)
			{
				#pragma omp parallel for
				for (int c = 0; c < K; ++c)
				{
					tensor<T, 2> t = out.Sub(n, c);
					t.Fill(pbias[c]);
				}
			}
		}

		{
			T4_ScopeProfiler(Conv2d_gemm_nn);
			details::gemm_nn(K, Hout * Wout, kernel_h * kernel_w * C, kernel.ptr(), kernel_h * kernel_w * C, columns, Hout * Wout, out.ptr(), Hout * Wout);
		}

		free(columns);
		return out;
	}


	template<int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, typename T>
	inline tensor<T, 4> ConvTranspose2d(
		  tensor<T, 4> in
		, tensor<T, 4> kernel
		, tensor<T, 1> bias)
	{
		T4_ScopeProfiler(ConvTranspose2d);
		assert(number(kernel) == channels(in));
		assert(kernel_h == height(kernel));
		assert(kernel_w == width(kernel));

		const int N = number(in);
		const int K = channels(kernel);
		
		const int Hin = height(in);
		const int Win = width(in);

		const int Hout = (Hin - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
		const int Wout = (Win - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;
		
		T* columns = new T[K * kernel_h * kernel_w * Hin * Win];
		memset(columns, 0, K * kernel_h * kernel_w * Hin * Win * sizeof(T));

		{
			T4_ScopeProfiler(ConvTranspose2d_gemm_nn);
			int _M = K * kernel_h * kernel_w;
			int _K = number(kernel);
			T* __restrict AT = (T*)malloc(_M * _K * sizeof(T));
			const T* __restrict A = kernel.ptr();
			for (int i = 0; i < _M; ++i)
				for (int j = 0; j < _K; ++j)
					memcpy(AT + j + i * _K, A + i + j * _M, sizeof(T));

			gemm_nn(K * kernel_h * kernel_w, Hin * Win, number(kernel), AT, _K, in.ptr(), Hin * Win, columns, Hin * Win);
			free(AT);
		}

		tensor<T, 4> out = tensor<T, 4>::New({ N, K, Hout, Wout });

		if (bias.ptr() != nullptr)
		{
			T* pbias = bias.ptr();
			for (int n = 0; n < N; ++n)
				for (int c = 0; c < K; ++c)
				{
					tensor<T, 2> t = out.Sub(n, c);
					t.Fill(pbias[c]);
				}
		}
		{
			T4_ScopeProfiler(ConvTranspose2d_col2im);
			details::col2im<kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w>(out.ptr(), columns, K, Wout, Hout, Win, Hin);
		}
		
		return out;
	}

	template<typename T>
	inline tensor<T, 2> Linear(
		tensor<T, 2> in
		, tensor<T, 2> weight
		, tensor<T, 1> bias)
	{
		T4_ScopeProfiler(Linear);
		assert(width(in) == width(weight));
		assert(height(weight) == width(bias));
		const int N = number(in);
		const int Inputs = width(weight);
		const int Outputs = height(weight);

		tensor<T, 2> out = tensor<T, 2>::New({ N, Outputs });
		if (bias.ptr() != nullptr)
		{
			for (int n = 0; n < N; ++n)
			{
				out.Sub(n).Assign(bias);
			}
		}

		details::gemm_nt(N, Outputs, Inputs, in.ptr(), Inputs, weight.ptr(), Inputs, out.ptr(), Outputs);
		return out;
	}

	template<int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h = 1, int dilation_w = 1, typename T>
	inline tensor<T, 4> MaxPool2d(tensor<T, 4> in)
	{
		T4_ScopeProfiler(MaxPool2d);
		const int N = number(in);
		const int C = channels(in);
		const int Hin = height(in);
		const int Win = width(in);

		const int Hout = (Hin + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
		const int Wout = (Win + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

		tensor<T, 4> out = tensor<T, 4>::New({ N, C, Hout, Wout });

		for (int n = 0; n < N; ++n)
		{
#pragma omp parallel for
			for (int c = 0; c < C; ++c)
			{
				auto inSubtensor = in.Sub(n, c);
				const T* __restrict src = inSubtensor.ptr();

				auto outSubtensor = out.Sub(n, c);
				T* __restrict dst = outSubtensor.ptr();

				for (int i = 0; i < Hout; i++)
				{
					for (int j = 0; j < Wout; j++)
					{
						int start_h = i * stride_h - pad_h;
						int start_w = j * stride_w - pad_w;

						int end_h = std::min(start_h + (kernel_h - 1) * dilation_h + 1, Hin);
						int end_w = std::min(start_w + (kernel_w - 1) * dilation_w + 1, Win);

						start_h += ((std::max(-start_h, 0) + dilation_h - 1) / dilation_h) * dilation_h;
						start_w += ((std::max(-start_w, 0) + dilation_w - 1) / dilation_w) * dilation_w;

						T maxval = -std::numeric_limits<T>::max();
						for (int y = start_h; y < end_h; y += dilation_h)
						{
							for (int x = start_w; x < end_w; x += dilation_w)
							{
								T val = src[y * Win + x];
								if (val > maxval)
								{
									maxval = val;
								}
							}
						}
						dst[i * Wout + j] = maxval;
					}
				}
			}
		}
		return out;
	}


	template<int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h = 1, int dilation_w = 1, typename T>
	inline tensor<T, 4> AveragePool2d(tensor<T, 4> in)
	{
		T4_ScopeProfiler(MaxPool2d);
		const int N = number(in);
		const int C = channels(in);
		const int Hin = height(in);
		const int Win = width(in);

		const int Hout = (Hin + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
		const int Wout = (Win + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

		tensor<T, 4> out = tensor<T, 4>::New({ N, C, Hout, Wout });

		for (int n = 0; n < N; ++n)
		{
#pragma omp parallel for
			for (int c = 0; c < C; ++c)
			{
				auto inSubtensor = in.Sub(n, c);
				const T* __restrict src = inSubtensor.ptr();

				auto outSubtensor = out.Sub(n, c);
				T* __restrict dst = outSubtensor.ptr();

				for (int i = 0; i < Hout; i++)
				{
					for (int j = 0; j < Wout; j++)
					{
						int start_h = i * stride_h - pad_h;
						int start_w = j * stride_w - pad_w;

						int end_h = std::min(start_h + (kernel_h - 1) * dilation_h + 1, Hin);
						int end_w = std::min(start_w + (kernel_w - 1) * dilation_w + 1, Win);

						start_h += ((std::max(-start_h, 0) + dilation_h - 1) / dilation_h) * dilation_h;
						start_w += ((std::max(-start_w, 0) + dilation_w - 1) / dilation_w) * dilation_w;

						T sum = 0;
						for (int y = start_h; y < end_h; y += dilation_h)
						{
							for (int x = start_w; x < end_w; x += dilation_w)
							{
								sum += src[y * Win + x];
							}
						}
						dst[i * Wout + j] = sum / (end_h - start_h) / (end_w - start_w);
					}
				}
			}
		}
		return out;
	}

	enum PaddingType
	{
		reflect
	};

	// TODO: implement actual padding instead of just zero padiing
	template<PaddingType type, typename T>
	inline tensor<T, 4> Pad(
		tensor<T, 4> in
		, int p_x0_begin
		, int p_x1_begin
		, int p_x2_begin
		, int p_x3_begin
		, int p_x0_end
		, int p_x1_end
		, int p_x2_end
		, int p_x3_end)
	{
		T4_ScopeProfiler(Pad);
		tensor<T, 4> out = tensor<T, 4>::New(
		{ 
			number(in) + p_x0_begin + p_x0_end,
			channels(in) + p_x1_begin + p_x1_end,
			height(in) + p_x2_begin + p_x2_end,
			width(in) + p_x3_begin + p_x3_end
		});

		for (int n = 0; n < number(in); ++n)
		{
			auto in_n = in.Sub(n);
			auto out_n = out.Sub(n + p_x0_begin);
			for (int c = 0; c < channels(in); ++c)
			{
				auto in_c = in_n.Sub(c);
				auto out_c = out_n.Sub(c + p_x1_begin);
				for (int h = 0; h < height(in); ++h)
				{
					auto in_h = in_c.Sub(h);
					auto out_h = out_c.Sub(h + p_x2_begin);

					memcpy(out_h.ptr() + p_x3_begin, in_h.ptr(), width(in_h) * sizeof(T));
				}
			}
		}

		return out;
	}

	template<size_t D>
	inline std::array<int, D> BroadCastShape(const std::array<int, D>& a, const std::array<int, D>& b)
	{
		std::array<int, D> result;
		for (int i = 0; i < D; ++i)
		{
			if (a[i] != b[i])
			{
				assert(a[i] == 1 || b[i] == 1);
			}
			result[i] = std::max(a[i], b[i]);
		}
		return result;
	}

	template<size_t D>
	inline std::array<int, 4> ExpandShape(const std::array<int, D>& x)
	{
		std::array<int, 4> out = {1, 1, 1, 1};
		for (int i = 0; i < D; ++i)
		{
			out[D - i - 1] = x[D - i - 1];
		}
		return out;
	}

	template<size_t D>
	int64 ComputeWrappedIndex(int64 n, int64 c, int64 h, int64 w, const std::array<int, D>& s);

	template<>
	inline int64 ComputeWrappedIndex<4>(int64 n, int64 c, int64 h, int64 w, const std::array<int, 4>& s)
	{
		return (w % s[3]) + (h % s[2]) * s[3] + (c % s[1]) * s[3] * s[2] + (n % s[0]) * s[3] * s[2] * s[1];
	}

	template<>
	inline int64 ComputeWrappedIndex<3>(int64 n, int64 c, int64 h, int64 w, const std::array<int, 3>& s)
	{
		return (w % s[2]) + (h % s[1]) * s[2] + (c % s[0]) * s[2] * s[1];
	}

	template<>
	inline int64 ComputeWrappedIndex<2>(int64 n, int64 c, int64 h, int64 w, const std::array<int, 2>& s)
	{
		return (w % s[1]) + (h % s[0]) * s[1];
	}

	template<>
	inline int64 ComputeWrappedIndex<1>(int64 n, int64 c, int64 h, int64 w, const std::array<int, 1>& s)
	{
		return (w % s[0]);
	}


#define POINT_WISE(OP) \
		auto out = in.SameAs(); \
		T*  __restrict dst = out.ptr(); \
		const T* __restrict src = in.ptr(); \
		int64 i = 0; \
		for (int64 l = (int64)in.size(); i < l - 4; i += 4) \
		{ \
			{ T v = src[i + 0]; OP; dst[i + 0] = out; }\
			{ T v = src[i + 1]; OP; dst[i + 1] = out; }\
			{ T v = src[i + 2]; OP; dst[i + 2] = out; }\
			{ T v = src[i + 3]; OP; dst[i + 3] = out; }\
		} \
		for (int64 l = (int64)in.size(); i < l; ++i) \
		{ \
			{ T v = src[i]; OP; dst[i] = out; }\
		}\
		return out;

#define POINT_WISE_BINARY(OP) \
		if (a.shape() == b.shape())\
		{\
			auto out = a.SameAs(); \
			T*  __restrict dst = out.ptr(); \
			const T* __restrict srcA = a.ptr(); \
			const T* __restrict srcB = b.ptr(); \
			int64 i = 0; \
			for (int64 l = a.size(); i < l - 4; i += 4) \
			{ \
				{ T a = srcA[i + 0]; T b = srcB[i + 0]; OP; dst[i + 0] = out; }\
				{ T a = srcA[i + 1]; T b = srcB[i + 1]; OP; dst[i + 1] = out; }\
				{ T a = srcA[i + 2]; T b = srcB[i + 2]; OP; dst[i + 2] = out; }\
				{ T a = srcA[i + 3]; T b = srcB[i + 3]; OP; dst[i + 3] = out; }\
			} \
			for (int64 l = a.size(); i < l; ++i) \
			{ \
				{ T a = srcA[i]; T b = srcB[i]; OP; dst[i] = out; }\
			}\
			return out;\
		} \
		else \
		{ \
			auto resultShape = BroadCastShape(a.shape(), b.shape()); \
			auto out = tensor<T, D>::New(resultShape); \
			T*  __restrict dst = out.ptr(); \
			const T* __restrict srcA = a.ptr(); \
			const T* __restrict srcB = b.ptr(); \
			auto resultShapeE = ExpandShape(resultShape); \
			auto aShape = a.shape(); \
			auto bShape = b.shape(); \
			for (int64 n = 0; n < resultShapeE[0]; ++n) \
				for (int64 c = 0; c < resultShapeE[1]; ++c) \
					for (int64 h = 0; h < resultShapeE[2]; ++h) \
						for (int64 w = 0; w < resultShapeE[3]; ++w) \
						{ \
							T a = srcA[ComputeWrappedIndex(n, c, h, w, aShape)];\
							T b = srcB[ComputeWrappedIndex(n, c, h, w, bShape)];\
							OP; dst[ComputeWrappedIndex(n, c, h, w, resultShape)] = out; \
						}\
			return out;\
		}

	template<typename T, int D>
	inline tensor<T, D> LeakyRelu(const tensor<T, D>& in, float alpha)
	{
		float _alpha = alpha - 1.0f;
		POINT_WISE(
			T out = ((v < 0) * _alpha + 1.0f) * v;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> Relu(const tensor<T, D>& in)
	{
		POINT_WISE(
			T out = (v > 0) ? v : T(0);
		)
	}

	namespace details
	{
		template<typename T>
		T tanh_f(T x);

		template<typename T>
		T exp_f(T x);
		
		template<>
		inline float tanh_f<float>(float x)
		{
			return tanhf(x);
		}

		template<>
		inline double tanh_f<double>(double x)
		{
			return tanh(x);
		}

		template<>
		inline float exp_f<float>(float x)
		{
			return expf(x);
		}

		template<>
		inline double exp_f<double>(double x)
		{
			return exp(x);
		}
	}

	template<typename T, int D>
	inline tensor<T, D> Tanh(const tensor<T, D>& in)
	{
		POINT_WISE(
			T out = details::tanh_f(v);
		)
	}

	template<typename T, int D>
	inline tensor<T, D> Exp(const tensor<T, D>& in)
	{
		POINT_WISE(
			T out = details::exp_f(v);
		)
	}

	template<typename T, int D>
	inline tensor<T, D> Neg(const tensor<T, D>& in)
	{
		POINT_WISE(
			T out = -v;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> Mul(const tensor<T, D>& in, T x)
	{
		POINT_WISE(
			T out = v * x;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> operator * (const tensor<T, D>& in, T x)
	{
		return Mul(in, x);
	}

	template<typename T, int D>
	inline tensor<T, D> Mul(const tensor<T, D>& a, const tensor<T, D>& b)
	{
		POINT_WISE_BINARY(
			T out = a * b;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> operator * (const tensor<T, D>& a, const tensor<T, D>& b)
	{
		return Mul(a, b);
	}

	template<typename T, int D>
	inline tensor<T, D> Add(const tensor<T, D>& in, T x)
	{
		POINT_WISE(
			T out = v + x;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> operator + (const tensor<T, D>& in, T x)
	{
		return Add(in, x);
	}

	template<typename T, int D>
	inline tensor<T, D> Add(const tensor<T, D>& a, const tensor<T, D>& b)
	{
		POINT_WISE_BINARY(
			T out = a + b;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> operator + (const tensor<T, D>& a, const tensor<T, D>& b)
	{
		return Add(a, b);
	}

	template<typename T, int D>
	inline tensor<T, D> Sub(const tensor<T, D>& a, const tensor<T, D>& b)
	{
		POINT_WISE_BINARY(
			T out = a - b;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> operator - (const tensor<T, D>& a, const tensor<T, D>& b)
	{
		return Sub(a, b);
	}

	template<typename T, int D>
	inline tensor<T, D> Div(const tensor<T, D>& a, const tensor<T, D>& b)
	{
		POINT_WISE_BINARY(
			T out = a / b;
		)
	}

	template<typename T, int D>
	inline tensor<T, D> operator / (const tensor<T, D>& a, const tensor<T, D>& b)
	{
		return Div(a, b);
	}

	template<int d, typename T, int D>
	inline tensor<T, 2> Flatten(const tensor<T, D>& in)
	{
		return in.Flatten(d);
	}

	template<typename T, int D>
	inline tensor<T, D> Dropout(const tensor<T, D>& in, float x)
	{
		return in;
	}

	template<int axis=-1, typename T, int D>
	inline tensor<T, D> Softmax(tensor<T, D> in)
	{
		T4_ScopeProfiler(Softmax);
		static_assert(axis == -1 || axis < D, "Wrong axis.");
		int _axis = (axis == -1) ? D - 1 : axis;

		tensor<T, D> output = Exp(in);
		int64 elementCount = in.size();
		int64 count = in.shape()[_axis];
		int64 sortInstances = elementCount / count;
		int64 stride = 1;
		if (axis != -1)
		{
			for (int i = _axis + 1; i < D; ++i)
			{
				stride *= in.shape()[i];
			}
		}

		T* dstPtr = output.ptr();

#if T4_USE_OMP
#pragma omp parallel for
#endif
		for (int i = 0; i < sortInstances; ++i)
		{
			T* start = dstPtr + (i / stride) * count * stride + (i % stride);
			T sum = T(0);
			for (int j = 0; j < count; ++j)
			{
				sum += start[j * stride];
			}
			for (int j = 0; j < count; ++j)
			{
				start[j * stride] /= sum;
			}
		}

		return output;
	}

	template<typename T>
	inline tensor<T, 4> BatchNormalization(tensor<T, 4> in, tensor<T, 1> weight, tensor<T, 1> bias, tensor<T, 1> running_mean, tensor<T, 1> running_var, float epsilon = 0.0f)
	{
		T4_ScopeProfiler(BatchNormalization);
		tensor<T, 4> out = in.SameAs();
		T* __restrict bias_ptr = bias.ptr();
		T* __restrict weight_ptr = weight.ptr();
		T* __restrict running_mean_ptr = running_mean.ptr();
		T* __restrict running_var_ptr = running_var.ptr();

		for (int n = 0; n < number(in); ++n)
		{
#if T4_USE_OMP
#pragma omp parallel for
#endif
			for (int c = 0; c < channels(in); ++c)
			{
				tensor<T, 2> sub_in = in.Sub(n, c);
				tensor<T, 2> sub_out = out.Sub(n, c);

				const T* __restrict src = sub_in.ptr();
				T* __restrict dst = sub_out.ptr();

				T mul = weight_ptr[c];
				T add = bias_ptr[c];

				T mean = running_mean_ptr[c];
				T invstd = 1.0f / sqrtf(running_var_ptr[c] + epsilon);

				add -= mean * invstd * mul;
				mul *= invstd;

				int64 i = 0;
				for (int64 l = (int64)sub_in.size(); i < l - 4; i += 4)
				{
					dst[i + 0] = src[i + 0] * mul + add;
					dst[i + 1] = src[i + 1] * mul + add;
					dst[i + 2] = src[i + 2] * mul + add;
					dst[i + 3] = src[i + 3] * mul + add;
				}

				for (int64 l = (int64)sub_in.size(); i < l; ++i)
				{
					dst[i] = src[i] * mul + add;
				}
			}
		}
		return out;
	}

	template<typename T>
	inline void free(T& x)
	{
		x = T();
	}

	template<typename H, typename ...Ts>
	inline void free(H& x, Ts... args)
	{
		x = H();
		free(args...);
	}

	namespace printing
	{
		template<typename T>
		inline bool isfinite(const T& x)
		{
			return true;
		}
		
		template<>
		inline bool isfinite(const float& x)
		{
			return std::isfinite(x);
		}
		
		template<>
		inline bool isfinite(const double& x)
		{
			return std::isfinite(x);
		}

		template<typename T, int D>
		inline int SetupFormat(std::ostream& stream, const tensor<T, D>& tensor)
		{
			constexpr int precision = 4;
			T max = -std::numeric_limits<T>::max();
			T min = std::numeric_limits<T>::max();
			bool has_fractional = false;
			const T* __restrict src = tensor.ptr();
			for (int64 i = 0, l = tensor.size(); i < l; ++i)
			{
				T x = abs(src[i]);
				if (std::isfinite(x))
				{
					max = x > max ? x : max;
					min = x < min ? x : min;
					has_fractional |= x != ceil(x);
				}
			}
			T emin = min != 0 ? floor(log10(min)) + 1 : 1;
			T emax = max != 0 ? floor(log10(max)) + 1 : 1;

			int width = 11;
			stream << std::scientific << std::setprecision(precision);

			if (has_fractional)
			{
				if (emax - emin < 5)
				{
					width = 6 + std::max((int)(emax), 1);
					stream << std::fixed << std::setprecision(precision);
				}
			}
			else
			{
				if (emax < 10)
				{
					width = (int)(emax + 1);
					stream.unsetf(std::ios_base::floatfield);
				}
			}
			return width;
		}
		
		inline void PrintIndent(std::ostream& stream, int indent)
		{
			for (int i = 0; i < indent; i++)
			{
				stream << " ";
			}
		}

		inline void PrintDots(std::ostream& stream, int indent, int level)
		{
			if (level > 1)
			{
				PrintIndent(stream, indent);
			}
			stream << "..., ";
			for (int64_t i = 0; i < level - 1; i++)
			{
				stream << "\n";
			}
		}

		inline void SkipEntries(std::ostream& stream, int& current, int total, int indent, int level)
		{
			if (total > 6 && current == 3)
			{
				PrintDots(stream, indent, level);
				current = total - 3;
			}
		}

		template<int level>
		class Printer
		{
		public:
			template<typename T>
			static void Print(std::ostream& output, const T* data, int indent, int width, const int* shape)
			{
				output << "[";
				size_t stride = 1;
				for (int i = 1; i < level; ++i)
				{
					stride *= shape[i];
				}
				bool nextline = false;
				for (int i = 0; i < *shape; ++i)
				{
					SkipEntries(output, i, *shape, indent + 1, level);
					if (nextline && level > 1)
					{
						PrintIndent(output, indent + 1);
					}
					nextline = false;

					Printer<level - 1>::template Print<T>(output, data + stride * i, indent + 1, width, shape + 1);

					if (i != *shape - 1)
					{
						output << ",";
						for (int l = 1; l < level; ++l)
						{
							output << "\n";
						}
						nextline = true;
					}
				}
				output << "]";
			}
		};

		template<>
		class Printer<1>
		{
		public:
			template<typename T>
			static void Print(std::ostream& output, const T* data, int indent, int width, const int* shape)
			{
				output << "[";
				for (int i = 0; i < *shape; ++i)
				{
					SkipEntries(output, i, *shape, indent + 1, 1);
					output << std::setw(width) << data[i];
					if (i != *shape - 1)
					{
						output << ", ";
					}
				}
				output << "]";
			}
		};

		template<typename T, int D>
		inline void PrintTensor(std::ostream& stream, const tensor<T, D>& tensor)
		{
			int width = SetupFormat(stream, tensor);

			stream << "tensor(";
			int indent = 7;

			if (tensor.size() != 0)
			{
				Printer<D>::template Print<T>(stream, tensor.ptr(), indent, width, tensor.shape().data());
			}

			stream << ")\n";
		}
	}

	template<typename T, int D>
	inline std::ostream& operator << (std::ostream& output, const tensor<T, D>& t)
	{
		printing::PrintTensor(output, t);
		return output;
	}
}
