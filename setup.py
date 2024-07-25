# Copyright 2024 the LlamaFactory team.
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


from setuptools import find_packages, setup


# def get_requires():
#     with open("requirements.txt", encoding="utf-8") as f:
#         file_content = f.read()
#         lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
#         return lines


def main():
    setup(
        name="happycode",
        version="0.0.1",
        author="Yuquan Xie",
        author_email="xieyuquan20016" "@" "gmail.com",
        description="Happy Code for MLLM",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["transformer", "pytorch", "deep learning", "MLLM", "DPO"],
        license="Apache 2.0 License",
        url="https://github.com/xieyuquanxx/HappyCode",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.11.0",
        # install_requires=get_requires(),
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
