# Contextual Dataset Valuation

A research note by Winston Zai Lin answering when a dataset can be valued reproducibly, and why predictive usefulness alone does not determine its market price.

**Status:** Version 1.0, October 22, 2026. Not peer reviewed. Synthetic demonstration and reporting protocol, not a new pricing algorithm or a validated market appraisal. Substantial AI assistance is disclosed in the paper.

- [Read the paper](paper.pdf)
- [Editable manuscript](paper.md)
- [Experiment](experiment.py)
- [Complete results](results.json)
- [Internal review](review.md)

## Reproduce

Use Python 3.10 or newer and NumPy. The recorded environment was Python 3.12.14 with NumPy 2.3.5.

```bash
python experiment.py --output results.json
```

The script runs entirely locally. It does not call model APIs, purchase compute, access email, or download datasets. It evaluates 50 generated instances, 16 coalitions per instance, and three evaluation contexts. Results include source-code hashes and software versions. With the recorded versions, a second execution reproduced the saved JSON byte for byte. Numerical reproducibility on other platforms may vary.

To regenerate the PDF, install ReportLab if needed, then run `python render_paper.py`. Rendering is separate from the experiment.

## Results and boundaries

The ranking of the two specialist datasets reversed across target populations in 42 of 50 synthetic instances. Duplicate suppliers had zero removal value but equal Shapley attribution. Those are illustrations of context and estimand dependence, not estimates of real-market prevalence. No monetary transactions were studied.

This package contains no private correspondence or personal datasets. References and limitations are in the manuscript.
