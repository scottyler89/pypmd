
import setuptools
import glob


with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    # Parse into a clean list, ignore blanks and comments
    install_requires = [
        ln.strip() for ln in fh.read().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]




setuptools.setup(
     name='percent_max_diff',  
     version='0.99.5',
     author="Scott Tyler",
     author_email="scottyler89@gmail.com",
     description="Percent Maximum Difference",
     long_description_content_type="text/markdown",
     long_description=long_description,
     install_requires=install_requires,
     #url="https://scottyler892@bitbucket.org/scottyler892/pyminer",
     packages=setuptools.find_packages(),
     include_package_data=True,
     #package_data={'': ['lib/*','pyminer/*.txt']},
     #scripts = script_list,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU Affero General Public License v3",
         "Operating System :: OS Independent",
     ],
 )
