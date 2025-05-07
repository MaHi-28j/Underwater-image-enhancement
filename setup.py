from setuptools import setup, find_packages

setup(
    name="underwater-image-enhancement",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'Pillow>=8.3.1',
        'matplotlib>=3.4.3',
        'streamlit>=1.8.0',
        'opencv-python>=4.5.3',
        'scipy>=1.7.1',
        'scikit-image>=0.18.3',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Enhanced Underwater Image Processing with Multiple Fusion Techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/underwater-image-enhancement",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
