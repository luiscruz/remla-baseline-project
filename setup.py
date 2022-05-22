from setuptools import find_packages, setup


linter_requires = [
    'pylint==2.12.2',
    'dslinter==2.0.6'
]

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    extras_require={
        'linter': linter_requires
    }
)

