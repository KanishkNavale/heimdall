from setuptools import setup, find_packages

import re


def get_version() -> str:
    with open("heimdall/__init__.py", "r") as f:
        content = f.read()

    # Use a regular expression to extract the version
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)

    if version_match:
        return version_match.group(1)
    else:
        raise ValueError("Version not found in __init__.py")


# Read the dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="heimdall",
    version=get_version(),
    packages=find_packages(),
    install_requires=requirements,
    author="Kanishk Navale",
    author_email="navalekanishk@gmail.com",
    description="Base package for probabilistic deep learning framework.",
    long_description=open("Readme.md").read(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
