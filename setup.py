import os
from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
with open(os.path.join(version_folder, 'src/version')) as f:
    __version__ = f.read().strip()

install_requires = [
    'accelerate',
    'codetiming',
    'datasets',
    'dill',
    'hydra-core',
    'numpy',
    'pandas',
    'peft',
    'pyarrow>=15.0.0',
    'pybind11',
    'pylatexenc',
    'ray[default]',
    'tensordict',
    'torchdata',
    'transformers',
    'vllm',
    'wandb',
    'datasets',
    'base58',
    'nltk',
    'numpy',
    'langid',
    'openai',
    'peft',
    'PyYAML',
    'rank_bm25',
    'rouge',
    'spacy',
    'tiktoken',
    'torch',
    'tqdm',
    'transformers>=4.40.0',
    'bm25s[core]',
    'fschat',
    'streamlit',
    'chonkie>=0.4.0',
    'gradio>=5.0.0',
    'rouge-chinese',
    'jieba',
    'chardet',
    'pdfplumber',
    'bs4',
    'modelscope',
    'loguru',

    # others
    'sglang',
    'jsonlines',
]

setup(
    name='hiersearch',
    version=__version__,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    url='https://github.com/plageon/HierSearch',
    license='MIT License',
    author='Jiejun Tan',
    author_email='zstanjj@gmail.com',
    description='HierSearch: A Hierarchical Enterprise Deep Search Framework Integrating Local and Web Searches',
    install_requires=install_requires,
    package_data={'': ['**/*.yaml']},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)