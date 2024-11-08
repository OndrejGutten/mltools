from setuptools import setup, find_packages

setup(
    name="mltools",
    version="0.1",
    packages=find_packages(where='./src'),
    package_dir={'': 'src'},
    description="A collection of utility functions for handling data and intermediate operations in MLOps.",
    author="Ondrej Gutten",
    author_email="ondrogutten@gmail.com",
    # url="https://github.com/Merelorn/mlops_utils",
    install_requires=['pandas', 'scikit-learn','mlflow','scipy','numpy'],
)
