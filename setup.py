import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="private_statistics",
    version="0.0.1",
    author="Sarus",
    author_email="contact@sarus.tech",
    description="Benchmarking quantile methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarus-tech/private-statistics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
