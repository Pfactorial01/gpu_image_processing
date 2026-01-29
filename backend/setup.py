"""
Setup script for GPU Image Filters Python bindings
"""
from setuptools import setup, find_packages

setup(
    name="gpu_image_filters",
    version="0.1.0",
    author="Your Name",
    description="GPU-accelerated image processing filters with Python bindings",
    long_description=open("../README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

