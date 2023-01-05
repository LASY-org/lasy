from setuptools import find_packages, setup

setup(
    name='lasy',
    packages=find_packages('.'),
    description='LAser pulse manipulation made eaSY',
    install_requires=['openpmd-api', 'scipy', 'numpy'],
    tests_require=["pytest"],
)
