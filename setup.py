from setuptools import find_packages, setup

# Get the package requirements from the requirements.txt file
install_requires = []
with open("./requirements.txt") as f:
    install_requires = [line.strip("\n") for line in f.readlines()]

setup(
    name='lasy',
    packages=find_packages('.'),
    description='LAser pulse manipulation made eaSY',
    install_requires=install_requires,
    tests_require=["pytest"],
)
