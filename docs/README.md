# NeuralLayers Documentation

Comprehensive documentation for the NeuralLayers framework using Sphinx.

## 📚 Building the Documentation

### Prerequisites

```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
```

### Build HTML Documentation

```bash
cd docs
make html
```

The documentation will be available at `_build/html/index.html`

### View Documentation

```bash
# Open in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Build Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# EPUB
make epub

# Plain text
make text

# Clean build directory
make clean
```

## 📖 Documentation Structure

```
docs/
├── index.rst                  # Main page
├── installation.rst           # Installation guide
├── quickstart.rst            # Quick start guide
├── conf.py                   # Sphinx configuration
├── Makefile                  # Build automation
│
├── tutorials/                # Step-by-step tutorials
│   ├── index.rst
│   ├── basic_usage.rst
│   └── advanced_training.rst
│
├── examples/                 # Code examples
│   ├── index.rst
│   └── ...
│
├── concepts/                 # Core concepts
│   ├── architecture.rst
│   ├── brain_modules.rst
│   ├── logical_reasoning.rst
│   └── consciousness.rst
│
├── api/                      # API reference
│   ├── logicalbrain_network.rst
│   ├── brain_network.rst
│   └── ...
│
├── advanced/                 # Advanced topics
│   ├── training.rst
│   ├── benchmarking.rst
│   └── deployment.rst
│
└── research/                 # Research documentation
    ├── mathematical_foundations.rst
    ├── biophysical_modeling.rst
    └── consciousness_theory.rst
```

## 🎨 Theme

Using the **Read the Docs** theme (`sphinx_rtd_theme`) for professional appearance.

## 🔧 Configuration

Main configuration in `conf.py`:
- **Extensions**: autodoc, napoleon, viewcode, mathjax, intersphinx
- **Theme**: sphinx_rtd_theme
- **LaTeX support**: For mathematical equations
- **API documentation**: Auto-generated from docstrings

## 📝 Writing Documentation

### reStructuredText (.rst)

Basic syntax:

```rst
Section Title
=============

Subsection
----------

**Bold text** and *italic text*

Code block:

.. code-block:: python

   import neurallayers
   model = UnifiedBrainNetwork()

Links:

* External: `PyTorch <https://pytorch.org/>`_
* Internal: :doc:`quickstart`
* API: :class:`UnifiedBrainNetwork`
```

### Markdown (.md)

Thanks to `myst-parser`, you can also write in Markdown.

## 🚀 Hosting Options

### GitHub Pages

```bash
# Build docs
make html

# Push to gh-pages branch
# GitHub will automatically host at:
# https://biblicalandr0id.github.io/NeuralLayers/
```

### Read the Docs

1. Connect your GitHub repository
2. RTD will automatically build and host
3. Available at: `https://neurallayers.readthedocs.io/`

### Local Server

```bash
cd _build/html
python -m http.server 8000
# Visit http://localhost:8000
```

## 🤝 Contributing

To contribute to documentation:

1. Edit `.rst` files in appropriate directory
2. Build locally to check: `make html`
3. Submit pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## 📚 Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [RTD Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [MyST Parser](https://myst-parser.readthedocs.io/)

---

**Questions?** Open an issue on GitHub!
