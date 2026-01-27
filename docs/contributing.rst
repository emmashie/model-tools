Contributing
============

We welcome contributions to ROMS Model Tools! This document provides guidelines for contributing.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Submit a pull request

Development Setup
-----------------

Install development dependencies:

.. code-block:: bash

   pip install -r requirements-dev.txt

This includes:

- pytest for testing
- black for code formatting
- flake8 for linting
- sphinx for documentation

Code Style
----------

We follow PEP 8 style guidelines with some exceptions:

- Line length: 100 characters (not 79)
- Use type hints for function signatures
- Use docstrings in NumPy format

Format code with black:

.. code-block:: bash

   black code/ scripts/

Check style with flake8:

.. code-block:: bash

   flake8 code/ scripts/

Documentation
-------------

All public functions and classes must have docstrings following NumPy format:

.. code-block:: python

   def my_function(param1, param2):
       """
       Brief description of function.
       
       Longer description with more details about what the
       function does and how to use it.
       
       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type
           Description of param2
           
       Returns
       -------
       type
           Description of return value
           
       Examples
       --------
       >>> result = my_function(1, 2)
       >>> print(result)
       3
       """
       return param1 + param2

Build documentation locally:

.. code-block:: bash

   cd docs/
   make html
   # View at _build/html/index.html

Testing
-------

Write tests for new functionality in ``tests/`` directory:

.. code-block:: python

   # tests/test_grid.py
   import pytest
   from grid import grid_tools
   
   def test_create_staggered_grids():
       grids = grid_tools.create_staggered_grids(
           30.0, 45.0, -80.0, -65.0, 1.0, 1.0
       )
       assert 'lon_rho' in grids
       assert 'lat_rho' in grids
       # Add more assertions

Run tests:

.. code-block:: bash

   pytest tests/

Pull Request Process
--------------------

1. Update documentation for any changed functionality
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Submit PR with clear description of changes

PR Checklist:

- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Code formatted with black
- [ ] No flake8 warnings
- [ ] CHANGELOG.md updated

Code Review
-----------

All contributions require code review before merging:

- At least one approval from maintainer
- All automated checks passing
- No merge conflicts with main branch

Areas for Contribution
----------------------

We especially welcome contributions in:

- Additional data source support (HYCOM, SODA, etc.)
- Performance optimizations
- Additional grid generation options
- Improved interpolation methods
- Bug fixes and documentation improvements

Reporting Issues
----------------

Report bugs and request features via GitHub Issues:

1. Check if issue already exists
2. Provide minimal reproducible example
3. Include system information (Python version, OS, package versions)
4. Describe expected vs actual behavior

Community Guidelines
--------------------

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what is best for the community

Questions?
----------

- Open a GitHub Discussion for questions
- Email maintainers for sensitive issues
- Join our Slack workspace (link in README)

License
-------

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE.md).
