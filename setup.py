from setuptools import setup, find_packages


def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()


def license():
    with open('LICENSE.rst') as readme_file:
        return readme_file.read()


setup(
    name='keyfi',
    version='0.0.1',
    description='Key feature identification via dimensionality reduction, unsupervised clustering, and feature correlation',
    long_description=readme(),
    author='Marc Rovira',
    author_email='marrs@kth.se',
    url='https://github.com/marrov/keyfi',
    license=license(),
    packages=find_packages(exclude=('examples'))
)
