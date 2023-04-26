from setuptools import find_packages, setup
import lasy  # In order to extract the version number

# Get the package requirements from the requirements.txt files
install_requires = []
with open("./requirements.txt") as f:
    install_requires = [line.strip("\n") for line in f.readlines()]
tests_require = []
with open("./tests/requirements.txt") as f:
    tests_require = [line.strip("\n") for line in f.readlines()]

setup(
    name="lasy",
    version=lasy.__version__,
    packages=find_packages("."),
    description="LAser pulse manipulation made eaSY",
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        "tests": tests_require,
    },
)
