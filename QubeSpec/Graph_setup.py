def clean_slate_protocol():
    import matplotlib
    matplotlib.rcdefaults()
    matplotlib.rcParams['figure.figsize'] = (5.5,4.5)


def graph_format(Labelsize=12):
    import matplotlib as mpl
    mpl.rcParams['xtick.major.width'] =1.5
    mpl.rcParams['ytick.major.width'] =1.5
    mpl.rcParams['xtick.major.size'] = 5.
    mpl.rcParams['ytick.major.size'] = 5.
    
    mpl.rcParams['xtick.minor.size'] = 3.5
    mpl.rcParams['ytick.minor.size'] = 3.5
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['ytick.minor.width'] = 1.5
    
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    

    mpl.rcParams['xtick.labelsize']= Labelsize
    mpl.rcParams['ytick.labelsize']=Labelsize
   
    mpl.rcParams['axes.linewidth'] = 2.
    
    mpl.rcParams['axes.labelsize']=15

    mpl.rcParams['image.origin']  ='lower'

def graph_format_official(Labelsize=12):
    # coding: utf-8
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if 1==0:
        matplotlib.rcParams['text.usetex'] = True

        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams['text.latex.preamble'] = (r'\\usepackage{tgheros}'
        r'\\usepackage{sansmath}'
        r'\sansmath')  

        matplotlib.rcParams['text.latex.preamble'] = (
        r'\usepackage{siunitx}'
        r'\usepackage{newunicodechar}'
        r'\newunicodechar{☧}{\ChiRo}'
        r'\sisetup{detect-all}'
        r'\usepackage{upgreek}'
        r'\usepackage{textgreek}'
        r'\usepackage{amssymb}'
        r'\usepackage{amsmath}'
        )
    # 14 from mpl_toolkits.axes_grid1 import make_axes_locatable
    # 15 from matplotlib.ticker import MaxNLocator
    #matplotlib.rcParams['mathtext.rm'] = 'serif'
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['xtick.major.width'] = 1.2
    matplotlib.rcParams['xtick.minor.width'] = 1.
    matplotlib.rcParams['xtick.major.size'] = 5.
    matplotlib.rcParams['xtick.minor.size'] = 3.
    matplotlib.rcParams['ytick.major.width'] = 1.2
    matplotlib.rcParams['ytick.minor.width'] = 1.
    matplotlib.rcParams['ytick.major.size'] = 6.
    matplotlib.rcParams['ytick.minor.size'] = 2.5
    matplotlib.rcParams['axes.linewidth'] = 1.2
    matplotlib.rcParams['xtick.direction'] = u'in'
    matplotlib.rcParams['ytick.direction'] = u'in'
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True
    matplotlib.rcParams['xtick.labelsize']= Labelsize
    matplotlib.rcParams['ytick.labelsize']= Labelsize
    #plt.rcParams['ps.useafm'] = True
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42


def move_figure(f, x, y):
    import matplotlib
    '''Move figure's upper left corner to pixel (x, y)'''
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)