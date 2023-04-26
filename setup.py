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
    python_requires=">=3.7",
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        "tests": tests_require,
    },
    url='https://github.com/LASY-org/lasy',
    project_urls={
        'Documentation': 'https://lasydoc.readthedocs.io',
        'Source': 'https://github.com/LASY-org/lasy',
        'Tracker': 'https://github.com/LASY-org/lasy/issues',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
    ],
    license="BSD-3-Clause-LBNL",
    license_files=["license.txt", "legal.txt"],
)
