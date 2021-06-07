# Generating SPHINX documentation
The used file format is reStructuredText. For more information go to the [Sphinx Documenation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
It is recommended to edit rst files with a different editor than pycharm since its preview often shows non-existing errors.

The docs directory contains 6 main parts:
* `index.rst`: It contains a Table of Contents that will link to all other pages of the documentation.
* `conf.py`: The config file is the tool for customization of Sphinx like setting author, templates and project version.
* `Makefile & make.bat`: Interface for building files
* `_build`: Directory which containes ouput files
* `_static`: Containes static files like images
* `_templates`: Contains customized templates (sphinx templates are only onfigured in conf.py)

## Adding Documentation
* Add rst File to source directory
* Add filename to the toctree (listing them in the content of the directive) in index.rst
## Build
* Install Sphinx 
* Generate HTML files using `make html` (in docs directory)


