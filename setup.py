from os import path

import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()


HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, "requirements.txt"), encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="injury",
    version="0.1.0",
    author="Sam Sharpe",
    author_email="ssharpe42y@gmail.com",
    description="Injury data processing and modeling for MLB players",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/ssharpe42/mlb-injury",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    packages=["injury"],
    python_requires=">=3.8",
)
