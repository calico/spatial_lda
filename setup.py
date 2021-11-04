import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spatial-lda",
    version="0.1.2",
    author="Zhenghao Chen, Vladimir Jojic",
    author_email="zhenghao@calicolabs.com, vjojic@calicolabs.com",
    description="Implementation of the Spatial-LDA model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/calico/spatial_lda/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

