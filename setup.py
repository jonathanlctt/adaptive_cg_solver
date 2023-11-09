from setuptools import setup

# Read README and requirements
with open("README.md", encoding="utf8") as f:
    readme = f.read()
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="adacg_solver",
    version='0.0.2',
    license='MIT',
    author="Jonathan Lacotte",
    author_email="<jonathanlacotte@gmail.com>",
    long_description=readme,
    url='https://github.com/jonathanlctt/randomized_ridge_regression_solver',
    install_requires=requirements,
    #packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
)