import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from hypergraph_tmx import ArxivHyperGraph

def main(data_file):
    # Load the ArxivHyperGraph object
    graph = ArxivHyperGraph(data_file)

    # Step 1: Extract author counts per paper
    author_counts = [len(authors) for authors in graph.paper_to_authors.values() if len(authors) > 0]
    author_counts = np.array(author_counts)

    # Print descriptive statistics
    print("Author count statistics:")
    print(f"  Total papers:     {len(author_counts)}")
    print(f"  Min authors:      {np.min(author_counts)}")
    print(f"  Max authors:      {np.max(author_counts)}")
    print(f"  Mean authors:     {np.mean(author_counts):.2f}")
    print(f"  Median authors:   {np.median(author_counts)}")
    print(f"  Mode authors:     {stats.mode(author_counts, keepdims=False).mode} (count: {stats.mode(author_counts, keepdims=False).count})")
    print(f"  Std deviation:    {np.std(author_counts):.2f}\n")

    # Step 2: Fit distributions manually
    results = {}

    # --- Poisson ---
    lambda_poisson = np.mean(author_counts)
    ks_stat, ks_p = stats.kstest(author_counts, 'poisson', args=(lambda_poisson,))
    results['poisson'] = {'params': (lambda_poisson,), 'ks_stat': ks_stat, 'ks_p': ks_p}

    # --- Negative Binomial ---
    mean = np.mean(author_counts)
    var = np.var(author_counts)
    if var > mean:
        p_nb = mean / var
        r_nb = mean**2 / (var - mean)
        ks_stat, ks_p = stats.kstest(author_counts, 'nbinom', args=(r_nb, p_nb))
        results['nbinom'] = {'params': (r_nb, p_nb), 'ks_stat': ks_stat, 'ks_p': ks_p}
    else:
        print("Skipping nbinom: variance <= mean, unsuitable for fitting.")

    # --- Log-normal ---
    try:
        params = stats.lognorm.fit(author_counts, floc=0)
        ks_stat, ks_p = stats.kstest(author_counts, 'lognorm', args=params)
        results['lognorm'] = {'params': params, 'ks_stat': ks_stat, 'ks_p': ks_p}
    except Exception as e:
        print(f"Could not fit lognorm: {e}")

    # --- Power-law ---
    try:
        params = stats.powerlaw.fit(author_counts, floc=0)
        ks_stat, ks_p = stats.kstest(author_counts, 'powerlaw', args=params)
        results['powerlaw'] = {'params': params, 'ks_stat': ks_stat, 'ks_p': ks_p}
    except Exception as e:
        print(f"Could not fit powerlaw: {e}")

    # Step 3: Show comparison
    print("\nGoodness-of-fit results (Kolmogorov-Smirnov):")
    for name, result in sorted(results.items(), key=lambda x: x[1]['ks_stat']):
        print(f"{name:<10} KS statistic: {result['ks_stat']:.4f}, p-value: {result['ks_p']:.4f}")

    # Step 4: Visualization
    sns.histplot(author_counts, bins=range(1, max(author_counts)+1), stat="density", kde=False,
                 color='skyblue', label='Empirical', edgecolor='black')

    x = np.arange(1, max(author_counts) + 1)
    for name, result in results.items():
        params = result['params']
        if name in ['poisson']:
            pmf = stats.poisson.pmf(x, *params)
            plt.plot(x, pmf, label=name)
        elif name in ['nbinom']:
            pmf = stats.nbinom.pmf(x, *params)
            plt.plot(x, pmf, label=name)
        else:
            pdf = getattr(stats, name).pdf(x, *params)
            plt.plot(x, pdf, label=name)

    plt.xlabel("Number of authors per paper")
    plt.ylabel("Density")
    plt.title("Distribution fitting for number of authors per paper")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return True 

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_hypergraph.py <input_file>")
        sys.exit(1)
    
    main(sys.argv[1])