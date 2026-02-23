import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peebee",
    version="1.3.1",
    author="Tom Donlon",
    author_email="thomas.donlon@uah.edu",
    description="A python package for the intersection of pulsar accelerations and Galactic structure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thomasdonlon/peebee",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.0',
    install_requires=[
        "numpy",
        "matplotlib",
        "uncertainties",
        "astropy",
        "scipy",
        "gala",
        "galpy"
        ],
)
