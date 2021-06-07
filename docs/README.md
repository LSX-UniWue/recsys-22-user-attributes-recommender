# Generating SPHINX documentation
The used file format is reStructuredText. For more information go to the [Sphinx Documenation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
It is recommended to edit rst files with a different editor than pycharm since its preview often shows non-existing errors.
## Adding Documentation
* Add rst File to source directory
* Add filename to the toctree (listing them in the content of the directive) in index.rst
## Build
* Install Sphinx 
* Generate HTML files using `make html` (in docs directory)
