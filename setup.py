# Copyright 2018 Stanislav Pidhorskyi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="tensor4",
    version="0.0.3",
    author='Stanislav Pidhorskyi',
    author_email='stanislav@podgorskiy.com',

    description="tensor4 - pytorch to C++ convertor using lightweight templated tensor library",
    long_description=long_description,
    long_description_content_type="text/markdown",

	url="https://github.com/podgorskiy/tensor4",

    packages=setuptools.find_packages(),

	classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Programming Language :: C++',
    ],
)
