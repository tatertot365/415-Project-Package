import setuptools
  
with open("README.md", "r") as fh:
    description = fh.read()
  
setuptools.setup(
    name="IS_415_Project",
    version="0.0.2",
    author="Tate Gillespie",
    author_email="tate.gillespie@gmail.com",
    packages=["IS_415_Project"],
    description="A package for the IS 415 Project",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/tatertot365/415-Project-Package",
    license='MIT',
    python_requires='>=3.8',
    install_requires=[]
)