from setuptools import setup, find_packages

with open('requirements.txt') as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

with open('somperf/__version__.py') as fh:
    version = fh.readlines()[-1].split()[-1].strip('\'\'')

setup(
    name='SOMperf',
    version=version,
    description='Self Organizing Maps performance metrics and quality indices',
    author='Florent Forest',
    author_email='florent.forest9@gmail.com',
    packages=find_packages(),
    install_requires=install_reqs,
    url='https://github.com/FlorentF9/SOMperf'
)
