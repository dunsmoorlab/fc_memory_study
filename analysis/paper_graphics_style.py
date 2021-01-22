import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

sns.set_context('paper')
rcParams['savefig.dpi'] = 900

def p_convert(x):
    if x < .001: return '***'
    elif x < .01: return '**'
    elif x < .05: return '*'
    elif x < .06: return '~'
    else: return ''

def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
def paired_barplot_annotate_brackets(txt, x_tick, height, y_lim, dh=.05, barh=.05, fs=10, maxasterix=None, ax=None):
    """ 
    Annotate barplot with p-values.

    :param txt: string to write or number for generating asterixes
    :param x_tick: center of pair of bars
    :param height: heights of the errors in question
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(txt) is str:
        text = txt
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while txt < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = x_tick-.2, height[0]
    rx, ry = x_tick+.2, height[1]

    ax_y0, ax_y1 = y_lim
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)
def simple_barplot_annotate_brackets(txt, ax, dh=.05, barh=.05, fs=10, maxasterix=None):
    
    lx, rx = ax.get_xticks()
    upper = {line.get_xdata()[0]:line.get_ydata().max() for line in ax.lines}
    ly, ry = upper[lx], upper[rx]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, txt, **kwargs)

def label_bars(txt,ax,dh=0.05,fs=10):
    #use the yerros as the reference since these contain the actual x-values where we want the text
    #for categorical variables, the list order is hue by x-values, so the first clump of a group of 3 is [0,2,4]

    ax_y0, ax_y1 = ax.get_ylim()
    dh *= (ax_y1 - ax_y0)

    x_vals = [line.get_xdata()[0] for line in ax.lines] 
    y_vals = [line.get_ydata().max() + dh for line in ax.lines]

    assert len(txt) == len(x_vals)
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    for i, t in enumerate(txt):
        if t != '':
            ax.text(x_vals[i],y_vals[i],t,**kwargs)



