from fm_config import *
from scipy.stats import pearsonr
def comp_percbend(x, y, beta=.2):
    """
    Percentage bend correlation (Wilcox 1994).

    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    beta : float
        Bending constant for omega (0 <= beta <= 0.5).

    Returns
    -------
    r : float
        Percentage bend correlation coefficient.
    pval : float
        Two-tailed p-value.

    Notes
    -----
    Code inspired by Matlab code from Cyril Pernet and Guillaume Rousselet.

    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

    .. [2] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606.
    """
    from scipy.stats import t
    X = np.column_stack((x, y))
    nx = X.shape[0]
    M = np.tile(np.median(X, axis=0), nx).reshape(X.shape)
    W = np.sort(np.abs(X - M), axis=0)
    m = int((1 - beta) * nx)
    omega = W[m - 1, :]
    P = (X - M) / omega
    P[np.isinf(P)] = 0
    P[np.isnan(P)] = 0

    # Loop over columns
    a = np.zeros((2, nx))
    for c in [0, 1]:
        psi = P[:, c]
        i1 = np.where(psi < -1)[0].size
        i2 = np.where(psi > 1)[0].size
        s = X[:, c].copy()
        s[np.where(psi < -1)[0]] = 0
        s[np.where(psi > 1)[0]] = 0
        pbos = (np.sum(s) + omega[c] * (i2 - i1)) / (s.size - i1 - i2)
        a[c] = (X[:, c] - pbos) / omega[c]

    # Bend
    a[a <= -1] = -1
    a[a >= 1] = 1

    # Get r, tval and pval
    a, b = a
    r = (a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())
    tval = r * np.sqrt((nx - 2) / (1 - r**2))
    pval = 2 * t.sf(abs(tval), nx - 2)

    coef = np.linalg.pinv(np.concatenate((X[:,0][:,np.newaxis],np.ones((len(X),1))),axis=1)) @ X[:,1]
    intercept = coef[1]
    slope = r / (X[:,0].std() / X[:,1].std())

    return r, pval, slope, intercept

def percbend_corr(x, y, tail='two-sided', nboot=10000):
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == y.ndim == 1, 'x and y must be 1D array.'
    assert x.size == y.size, 'x and y must have the same length.'

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=True)
    nx = x.size

    r, pval, slope, intercept = comp_percbend


def plot_full_skipped_corr(x,y,title):
    from pingouin.utils import _is_sklearn_installed
    _is_sklearn_installed(raise_error=True)
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet
    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))
    # Compute center and distance to center
    center = MinCovDet(random_state=42).fit(X).location_
    B = X - center
    B2 = B**2
    bot = B2.sum(axis=1)
    # Loop over rows
    dis = np.zeros(shape=(nrows, nrows))
    for i in np.arange(nrows):
        if bot[i] != 0:  # Avoid division by zero error
            dis[i, :] = np.linalg.norm(B * B2[i, :] / bot[i], axis=1)
    def idealf(x):
        """Compute the ideal fourths IQR (Wilcox 2012).
        """
        n = len(x)
        j = int(np.floor(n / 4 + 5 / 12))
        y = np.sort(x)
        g = (n / 4) - j + (5 / 12)
        low = (1 - g) * y[j - 1] + g * y[j]
        k = n - j + 1
        up = (1 - g) * y[k - 1] + g * y[k - 2]
        return up - low

    # One can either use the MAD or the IQR (see Wilcox 2012)
    # MAD = mad(dis, axis=1)
    iqr = np.apply_along_axis(idealf, 1, dis)
    thresh = (np.median(dis, axis=1) + gval * iqr)
    outliers = np.apply_along_axis(np.greater, 0, dis, thresh).any(axis=0)

    cloud = X[~outliers]

    rs = np.zeros(10000)
    for i in range(10000):
        _samp = np.random.choice(range(len(cloud)),size=len(cloud))
        rs[i] = pearsonr(cloud[_samp,0],cloud[_samp,1])[0]
    if rs.mean() > 0:
        p = (1 - np.mean(rs > 0)) * 2
    else:
        p = (1 - np.mean(rs < 0)) * 2

    r_pearson, _ = pearsonr(x[~outliers], y[~outliers])
    ci_l, ci_u = np.percentile(rs,[2.5,97.5])
    
    fig, (ax1, ax3) = plt.subplots(2, figsize=(6, 10))
    # plt.subplots_adjust(wspace=0.3)
    sns.despine()

    # Scatter plot and regression lines
    sns.regplot(x[~outliers], y[~outliers], ax=ax1, color='darkcyan')
    ax1.scatter(x[outliers], y[outliers], color='indianred', label='outliers')
    ax1.scatter(x[~outliers], y[~outliers], color='seagreen', label='good')

    sns.distplot(rs, kde=True, ax=ax3, color='steelblue')
    for i in [ci_l,ci_u]:
        ax3.axvline(x=i, color='coral', lw=2)
    ax3.axvline(x=0, color='k', ls='--', lw=1.5)
    ax3.set_xlabel('Correlation coefficient')
    ax3.set_title(
        'Skipped Pearson r = {}\n95% CI = [{}, {}], P = {}'.format(r_pearson.round(2),
                                                           ci_l.round(2),
                                                           ci_u.round(2),
                                                           p.round(4)),
        y=1.05)
    ax1.set_xlim([i*1.2 for i in ax1.get_xlim()])
    ax1.set_title(title)
    # Optimize layout
    plt.tight_layout()

def skipped_corr(x, y, vis=False, ax=None, color='blue'):

    from pingouin.utils import _is_sklearn_installed
    _is_sklearn_installed(raise_error=True)
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet
    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))
    # Compute center and distance to center
    center = MinCovDet(random_state=42).fit(X).location_
    B = X - center
    B2 = B**2
    bot = B2.sum(axis=1)
    # Loop over rows
    dis = np.zeros(shape=(nrows, nrows))
    for i in np.arange(nrows):
        if bot[i] != 0:  # Avoid division by zero error
            dis[i, :] = np.linalg.norm(B * B2[i, :] / bot[i], axis=1)
    def idealf(x):
        """Compute the ideal fourths IQR (Wilcox 2012).
        """
        n = len(x)
        j = int(np.floor(n / 4 + 5 / 12))
        y = np.sort(x)
        g = (n / 4) - j + (5 / 12)
        low = (1 - g) * y[j - 1] + g * y[j]
        k = n - j + 1
        up = (1 - g) * y[k - 1] + g * y[k - 2]
        return up - low

    # One can either use the MAD or the IQR (see Wilcox 2012)
    # MAD = mad(dis, axis=1)
    iqr = np.apply_along_axis(idealf, 1, dis)
    thresh = (np.median(dis, axis=1) + gval * iqr)
    outliers = np.apply_along_axis(np.greater, 0, dis, thresh).any(axis=0)

    cloud = X[~outliers]

    rs = np.zeros(10000)
    for i in range(10000):
        _samp = np.random.choice(range(len(cloud)),size=len(cloud))
        rs[i] = pearsonr(cloud[_samp,0],cloud[_samp,1])[0]
    if rs.mean() > 0:
        p = (1 - np.mean(rs > 0)) * 2
    else:
        p = (1 - np.mean(rs < 0)) * 2
    r_pearson, _ = pearsonr(x[~outliers], y[~outliers])
    ci_l, ci_u = np.percentile(rs,[2.5,97.5])
    
    # Scatter plot and regression lines
    if vis and ax == None:
        fig, ax = plt.subplots()
    if vis:
        sns.regplot(x[~outliers], y[~outliers], ax=ax, color=color, scatter=False)
        ax.scatter(x, y, color=color, edgecolor=color)
    print(
            'Skipped Pearson r = {}\n95% CI = [{}, {}], P = {}'.format(r_pearson.round(2),
                                                           ci_l.round(2),
                                                           ci_u.round(2),
                                                           p.round(4)))
    # Optimize layout

