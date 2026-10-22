# Contextual Dataset Valuation and the Limits of a Universal Price

Winston Zai Lin

Research note and reproducible synthetic experiment | October 22, 2026 | Version 1.0 | Not peer reviewed

## Abstract

Can a digital asset be valued reproducibly? For datasets used in machine learning, the answer depends on the quantity being estimated. Incremental predictive utility can be measured conditional on a learning procedure, existing data, evaluation population, and utility function. A market price cannot be identified from predictive utility alone. This note separates acquisition value, removal value, cooperative attribution, and monetary willingness to pay. Building on established data valuation methods, it specifies a reporting protocol and derives a conservative simultaneous uncertainty bound for a finite family of paired acquisition comparisons. A fully local synthetic experiment evaluates four dataset suppliers, three target populations, and all 16 supplier coalitions across 50 generated instances. The ranking of two specialist datasets reverses between target populations in 42 of 50 instances. With exact deduplication, two identical suppliers have zero removal value but equal, generally nonzero Shapley attribution. These results illustrate dependence on the evaluation context and attribution question; they do not establish a new valuation algorithm, superiority over existing methods, or a dollar price. The practical conclusion is that reproducibility requires a specified context and estimand, while monetary valuation additionally requires economic assumptions and evidence.

## 1 Introduction

A dataset can improve a model, generate revenue, satisfy a contractual requirement, or have no incremental usefulness to an organization that already possesses equivalent information. These are different senses of value. Treating them as interchangeable creates an identification problem before any estimation algorithm is chosen.

Data valuation is an established research area. Data Shapley assigns training-data contributions relative to a model-performance utility [1,2]. Distributional Shapley considers an underlying data distribution [3]. Statistical uncertainty for Data Shapley has also been studied [4]. A credible account must therefore ask what remains unidentified after such measurements, rather than claim that data cannot currently be valued.

This note addresses datasets used for supervised learning. It does not cover cryptocurrency pricing, intellectual-property appraisal generally, personal net worth, or the value of a person. It develops a practical reporting specification and a transparent demonstration using existing estimands. The contribution is a synthesis and auditable worked example, not a claim to have discovered contextual value, Shapley attribution, or concentration inequalities.

The research question is: under which explicit conditions can dataset utility be estimated reproducibly, and which additional assumptions are necessary to interpret it as monetary value? Three distinctions organize the answer. First, utility belongs to an asset-in-context pair. Second, buying an asset and allocating credit among suppliers are different decisions. Third, sampling uncertainty and economic ambiguity require different remedies.

## 2 Related work and scope of contribution

Ghorbani and Zou [1] introduced Data Shapley for allocating predictive contributions to training examples. Jia et al. [2] developed efficient Shapley-based data valuation methods. The present experiment instead uses four suppliers, each providing a batch, so all coalitions can be enumerated exactly. It does not propose a faster approximation or improve these algorithms.

Ghorbani, Kim, and Zou [3] introduced a distributional framework, including statistical properties of valuations under a data distribution. Wu et al. [4] explicitly study uncertainty quantification for Data Shapley through statistical inference. Our acquisition uncertainty bound concerns a different estimand: a paired held-out utility difference for fixed fitted models. It does not replace their distributional Shapley inference.

Agarwal, Dahleh, and Sarkar [5] study a data marketplace and address replication, combinatorial value, and heterogeneous prediction tasks. This work establishes that economic mechanisms and replication are not new concerns. OpenDataVal [6] supplies a common environment for comparing valuation methods and reports task-dependent performance. Consequently, a claim of general empirical superiority would require substantially broader benchmarking than the illustrative experiment here.

The proposed reporting specification brings context, estimand, uncertainty scope, and monetary assumptions into one record. No exhaustive novelty claim is made for that combination. The experiment is deliberately small enough to inspect every coalition and reproduce without paid services.

## 3 Define the context before assigning a number

Let D0 denote data already available to a buyer and D1,...,Dm the candidate supplier batches. Let A be a fully specified training procedure, P the target evaluation distribution, and u(f,z) a utility in [0,1] for fitted predictor f and evaluation example z. For accuracy, u is the indicator of a correct prediction. The context also includes preprocessing, deduplication, seeds or a seed distribution, hyperparameters, and the permitted uses of the data.

For supplier coalition S, define the fitted model and population utility:

    f_S = A(D0 union D_S)
    U_P(S) = E_{Z ~ P}[u(f_S, Z)]

The union symbol here denotes the protocol's specified data-combination operation. Concatenation, weighting, and deduplication can produce different models. In our experiment, exact duplicate feature-label rows are removed before fitting. This is part of the experimental definition, not a universal requirement for valuation.

Three familiar estimands answer different questions:

    Acquisition:  Delta_i = U_P({i}) - U_P(empty)
    Removal:      LOO_i = U_P(N) - U_P(N without i)
    Attribution:  phi_i = sum_{S subset N without i} w(S)
                         * [U_P(S union {i}) - U_P(S)]
    w(S) = |S|! (m - |S| - 1)! / m!

Here N is the supplier set. Acquisition measures the incremental utility of adding one batch to the existing baseline. Removal measures dependence on a supplier when all suppliers are present. Shapley attribution averages marginal contributions over coalition positions. These quantities can disagree without any estimator being incorrect. In particular, acquisition gains need not sum to the gain of acquiring all batches.

A buyer choosing among individual batches should evaluate acquisition differences against the same existing inventory and intended target population. A contributor-payment scheme may choose cooperative attribution, but must additionally specify what economic surplus is distributed and why its fairness axioms are appropriate. A supplier-level Shapley value is not automatically an individual record's value.

## 4 Two identification limits

### 4.1 A universal ordering cannot represent conflicting contexts

Suppose two assets a and b satisfy Delta_a(c1) > Delta_b(c1), but Delta_a(c2) < Delta_b(c2). There is no context-independent scalar s that preserves both strict orderings. The first comparison requires s(a) > s(b), while the second requires s(a) < s(b), a contradiction.

This elementary observation is not a new impossibility theorem about markets. Its scope is precise: a single fixed ordering cannot preserve every contextual utility ordering when a reversal exists. A population-weighted average can still be defined, but its weights become an additional choice and it need not preserve each constituent ordering. Likewise, prices can be observed in a particular market even when other markets would price the same asset differently.

### 4.2 Predictive gain does not identify willingness to pay

Consider a deliberately simple economic model. The buyer expects H independent uses over a stated horizon, receives b monetary units per additional correct outcome, and incurs incremental non-acquisition cost k. Under risk neutrality, constant marginal benefit, and transfer of the measured gain to deployment, the modeled break-even acquisition expenditure is:

    W = H * b * Delta_i - k

This expression is a scenario calculation, not a market-clearing price. The same measured Delta_i is compatible with different H, b, and k, and therefore different W. If Delta_i is positive, choosing two different positive values of b already produces different willingness-to-pay estimates while leaving the predictive experiment unchanged. Predictive observations alone cannot identify these unobserved economic parameters.

For example, a hypothetical gain of 0.02 over 10,000 uses with b = 1 and k = 50 yields W = 150 monetary units. With b = 0.10 and the same other inputs, W = -30. Neither scenario is an empirical valuation in this study. A negative result means the modeled benefit does not cover the modeled non-acquisition cost; it is not a negative observed transaction price.

Actual exchange also depends on alternatives, seller reservation values, bargaining, exclusivity, contracts, and demand. An asset's technical usefulness does not establish ownership or permission to sell it. Monetary valuation should remain unresolved when these assumptions are missing, rather than be filled in with arbitrary coefficients presented as measurements.

## 5 A reproducible reporting protocol

A valuation record should contain the asset identifier and hash; baseline inventory; learning algorithm and software versions; preprocessing and deduplication; target population and sampling design; utility and units; estimand; fitted-model seeds; point estimate; interval and its probability interpretation; sensitivity contexts; and data-use restrictions. Any monetary scenario must separately name its horizon, marginal benefit, costs, and uncertainty.

Freeze candidate assets, preprocessing, and hyperparameters before inspecting the held-out evaluation outcomes. Fit each model using training data only. Evaluate the compared models on the same held-out examples to preserve pairing. Keep an untouched audit set if the results will be used to revise candidate assets or tune the model. Publish a context-specific vector when several deployment populations matter rather than silently average them into a universal score.

### 5.1 A simultaneous held-out acquisition bound

Conditional on the fixed fitted models and training data, let d_ij = u(f_{i}, Z_j) - u(f_empty, Z_j). Then d_ij lies in [-1,1]. For stratified evaluation, let g index independent evaluation strata, n_g their sample sizes, and w_cg the known nonnegative target weights for context c, summing to one. The paired estimator is:

    Delta_hat_ic = sum_g w_cg * mean_j(d_igj)

For a predeclared family of M asset-context contrasts, a weighted Hoeffding inequality [7] and union bound give simultaneous intervals with probability at least 1-alpha:

    epsilon_c = sqrt(2 * log(2*M/alpha) * sum_g(w_cg^2/n_g))
    Delta_ic in [Delta_hat_ic - epsilon_c,
                 Delta_hat_ic + epsilon_c] intersect [-1,1]

Derivation: each weighted summand has range length 2*w_cg/n_g. The sum of squared range lengths is 4*sum_g(w_cg^2/n_g). The two-sided Hoeffding bound is therefore 2*exp(-t^2 / (2*sum_g(w_cg^2/n_g))). Setting this to alpha/M and applying a union bound proves the stated familywise guarantee. Different contrasts may share test observations; the union bound does not require independence between contrasts. Independent examples within the sampling design remain necessary.

The guarantee covers held-out sampling error for the specified fitted models, not training-set variation, unknown context weights, distribution shift, labeling bias, or market-price uncertainty. These intervals are not Shapley confidence intervals. A seed-by-seed acquisition family in our experiment has M = 12 and alpha = 0.05; the guarantee is per generated instance, not simultaneously across all 50 instances. Conservative bounds can be wide, which is preferable to treating uncertainty as zero.

If a minimum worthwhile utility gain tau is specified independently, label a comparison supported when its lower bound exceeds tau, unsupported at that threshold when its upper bound is below tau, and inconclusive otherwise. Such labels inherit only the interval's scope. A positive predictive gain alone does not authorize a purchase or establish an economic return.

## 6 Reproducible synthetic experiment

### 6.1 Design

The experiment generates two subpopulations, g = 0 and g = 1. Features x1,x2,x3 are independent standard normal variables. Labels take values -1 or +1 according to the sign of x1 + 0.5*x3 + e in group 0, and x2 - 0.5*x3 + e in group 1, with independent normal noise e of standard deviation 0.35. Group-specific intercepts and feature interactions give eight regression features. Ridge least squares uses an identity penalty with coefficient 1 on all eight coefficients; predictions are +1 only when the fitted score is positive, and -1 otherwise.

The baseline contains six independently sampled training examples from each group. Supplier A supplies 40 new group-0 examples; B supplies 40 new group-1 examples; A_copy supplies an exact copy of A; Noise supplies 40 examples with random groups and independent fair random labels. Exact duplicate feature-label rows are removed within every coalition before fitting. The same rule applies to every coalition, including the baseline.

We fit all 16 coalitions for each of 50 seeds, numbered 0 through 49. Each instance has 2,000 independently generated held-out examples per group. Contexts use group-0 population weights 0.9, 0.5, and 0.1, called A-heavy, balanced, and B-heavy. The fitted models and held-out examples stay fixed when changing these weights; only the evaluation population mixture changes. No model or hyperparameter is chosen using the held-out labels. This small design was specified before its first run, but it was not externally preregistered.

The provided program requires NumPy and runs locally without network access, paid APIs, downloaded datasets, or GPU services. The recorded run used Python 3.12.14 and NumPy 2.3.5. Enumeration avoids Monte Carlo coalition-approximation error, but held-out sampling error remains.

### 6.2 Results

Table 1 reports acquisition gains in accuracy percentage points, averaged over the 50 generated instances. Parentheses contain the sample standard deviation across instances; they are not confidence intervals or standard errors. These replications generate different assets and baselines, so their dispersion is not the uncertainty of one fixed dataset's value.

| Context | A | B | A_copy | Noise |
|---|---:|---:|---:|---:|
| A-heavy | 9.33 (7.87) | 1.20 (1.01) | 9.33 (7.87) | -19.26 (13.84) |
| Balanced | 5.18 (4.37) | 6.01 (5.05) | 5.18 (4.37) | -17.37 (11.00) |
| B-heavy | 1.04 (0.87) | 10.83 (9.09) | 1.04 (0.87) | -15.47 (15.71) |

The A-versus-B acquisition ranking reverses between A-heavy and B-heavy contexts in 42 of 50 instances (84%). A reversal is counted when the product of the two A-minus-B differences is strictly negative; ties do not count. This is a descriptive frequency for this constructed generator, not an estimate of how often reversals occur in real data markets. The generator intentionally creates specialization, so the result illustrates a mechanism rather than discovering its prevalence.

Table 2 compares three estimands under the balanced context. Values are mean accuracy percentage points across generated instances.

| Supplier | Acquisition | Removal | Shapley attribution |
|---|---:|---:|---:|
| A | 5.18 | 0.00 | 3.97 |
| B | 6.01 | 12.02 | 9.02 |
| A_copy | 5.18 | 0.00 | 3.97 |
| Noise | -17.37 | -3.07 | -8.84 |

Removing either A or A_copy from the full coalition changes no training rows after deduplication, so both removal values are exactly zero. Their Shapley values are equal by symmetry, and their mean attributions are positive. The program checks these identities and the efficiency identity sum_i(phi_i) = U_P(N) - U_P(empty) in every instance and context. The table is not a competition in which one estimand is declared correct: the different numbers answer different questions. Negative values for Noise show that indiscriminately using a harmful batch can reduce this learner's performance.

The simultaneous acquisition interval half-width is approximately 7.12 percentage points in either specialist-heavy context and 5.56 points in the balanced context, for each fixed instance. These widths follow from the declared sampling plan, independent of the observed accuracy values. They must not be attached to the across-instance means in Tables 1 and 2. Instance-specific centers and intervals appear in results.json. The conservative interval can leave useful-looking point estimates inconclusive.

### 6.3 What the experiment does not establish

The experiment contains one synthetic generator, one learner, four suppliers, and no monetary transactions. It does not show that a proposed protocol improves procurement, predict real prices, or outperform OpenDataVal methods. No empirical coverage study is claimed for the intervals; their guarantee follows from the stated conditional assumptions. Hyperparameter sensitivity, real-world distribution shift, and model-family sensitivity remain untested. The duplicated supplier is a deliberate demonstration, not an empirical model of ownership or a replication-resistant payment mechanism.

## 7 Implications for monetary valuation and adoption

The immediate reusable object is a valuation report with explicit conditions. Two teams using the same assets, training procedure, evaluation sample, software, and estimand should reproduce the computation within numerical tolerance. Teams drawing fresh evaluation samples should expect sampling differences and compare estimates with their uncertainty, not demand identical numbers. Reproducible computation and valid deployment inference are distinct goals.

A monetary extension requires evidence linking the utility metric to economic outcomes. The linear scenario in Section 4 is appropriate only when a constant value per additional correct outcome is defensible. If different errors have different consequences, accuracy should be replaced by a prespecified bounded utility or a separately justified loss model. A monotonic transformation of accuracy cannot supply the missing economic information.

Similarly, rewarding data contributors requires a policy choice in addition to a technical measurement. Treating copied data as distinct suppliers can change an allocation even when it adds no information. Identity verification, provenance, contractual rights, and replication-resistant market design cannot be inferred from a predictive utility score. The protocol should report unresolved permission questions rather than convert them into a numeric discount that falsely suggests they have been settled.

A stronger empirical study should use several licensed real datasets and model families, compare acquisition strategies against direct marginal utility, removal, random selection, and relevant cooperative methods, and measure realized utility on an untouched audit set under fixed budgets. Claims about prices require transaction or willingness-to-pay evidence with an explicitly defined market and horizon. Claims about uncertainty under training variation require an additional inferential target and analysis. These are requirements for stronger conclusions, not results already obtained here.

## 8 Conclusion

Dataset utility can be estimated reproducibly when the context, estimand, and uncertainty scope are fixed. The same measurements do not identify a universal asset ranking or a monetary price. In a transparent synthetic example, context changes reverse specialist rankings and redundant suppliers receive different values under removal and cooperative attribution. The practical answer is therefore conditional: measure usefulness in a declared setting; distinguish acquisition from allocation; quantify the uncertainty actually covered by the experiment; and require separate economic evidence before reporting a price.

## Reproducibility and research disclosure

The accompanying experiment.py regenerates results.json with `python experiment.py --output results.json`. The results record all seeds, context weights, software versions, per-instance estimates, and a source-code SHA-256 hash. The scientific experiment uses only generated data and NumPy. The PDF is a presentation of the manuscript, not an additional source of experimental results.

OpenAI Codex was used substantially to develop the exposition, construct the synthetic experiment, execute the code, and conduct an internal mathematical and consistency review. That automated review is not independent peer review. No human-subject data, private email content, or proprietary datasets are used in the research. This version is an openly shared research note; it does not claim journal acceptance or an independently established novel valuation algorithm.

## References

[1] Amirata Ghorbani and James Zou. Data Shapley: Equitable Valuation of Data for Machine Learning. Proceedings of Machine Learning Research 97, 2242-2251, 2019. https://proceedings.mlr.press/v97/ghorbani19c.html

[2] Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gurel, Bo Li, Ce Zhang, Dawn Song, and Costas Spanos. Towards Efficient Data Valuation Based on the Shapley Value. 2019. https://arxiv.org/abs/1902.10275

[3] Amirata Ghorbani, Michael P. Kim, and James Zou. A Distributional Framework for Data Valuation. Proceedings of Machine Learning Research 119, 2020. https://arxiv.org/abs/2002.12334

[4] Mengmeng Wu, Zhihong Liu, Xiang Li, Ruoxi Jia, and Xiangyu Chang. Uncertainty Quantification of Data Shapley via Statistical Inference. 2024. https://arxiv.org/abs/2407.19373

[5] Anish Agarwal, Munther Dahleh, and Tuhin Sarkar. A Marketplace for Data: An Algorithmic Solution. 2018, revised 2019. https://arxiv.org/abs/1805.08125

[6] Kevin Fu Jiang, Weixin Liang, James Zou, and Yongchan Kwon. OpenDataVal: a Unified Benchmark for Data Valuation. NeurIPS Datasets and Benchmarks, 2023. https://arxiv.org/abs/2306.10577

[7] Wassily Hoeffding. Probability Inequalities for Sums of Bounded Random Variables. Journal of the American Statistical Association 58(301), 13-30, 1963. https://doi.org/10.1080/01621459.1963.10500830
