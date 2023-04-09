from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Deep learning package'
LONG_DESCRIPTION = 'A package that allows for the training and evaluation on inputs of convolutional and fully connected neural networks.'

# Setting up
setup(
    name="deeplearningnumpy",
    version=VERSION,
    author="oliver-hamilton (Oliver Hamilton)",
    author_email="<ohamilton0079@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'deep learning', 'neural network', 'convolutional', 'fully connected'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)