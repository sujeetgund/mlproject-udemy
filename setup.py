from setuptools import setup, find_packages


def get_requirements(filepath: str) -> list[str]:
    try: 
        with open(filepath, 'r') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        return []

setup(
    name='ml_project_udemy',
    version='0.1.0',
    author='Sujeet Gund',
    author_email='sujeetgund@gmail.com',
    description='A machine learning project for stock price prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sujeetgund/mlproject-udemy',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
