from setuptools import setup, find_packages

setup(
    name="my-project",
    version="0.1",
    description="Project description.",
    author="Author name",
    author_email="author@email.com",
    packages=find_packages(),
    install_requires=[
        "torch==1.8.1",
        "torchvision==0.9.1",
        "pytorch-lightning==1.3.5",
        "scikit-image==0.18.1",
        "pandas==1.2.4",
    ],
)
