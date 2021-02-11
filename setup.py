import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='ml_dental_ssd300',
    version="1.0.0",
    author='Stephanie Kao',
    author_email='stephanie.kao5@gmail.com',
    description='Dental Teeth X-ray module',
    long_description=long_description,
    packages=['ssd300'],
    license='see LICENSE.TXT',
    python_requires='>=3.6',
    scripts=['scripts/train_ssd300.py',
             'scripts/eval_ssd300.py',
	     'scripts/deploy_ssd300.py']
)
