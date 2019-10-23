try: 
    from setuptools.core import setup
except ImportError as e:
    from distutils.core import setup

setup(
    name='mynlp',
    version='0.11',
    description='my nlp toolkit',
    author='Xiaolong Liang',
    author_email='rembern@126.com',
    url='',
    packages=['mynlp', 'mynlp.preprocess'],
    package_data={'mynlp': ['data/*.dat']},
    install_requires=['numpy', 'nltk', 'sklearn', 'gensim'],
    test_suite='tests'
)
