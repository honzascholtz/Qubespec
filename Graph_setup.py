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
    

    mpl.rcParams['xtick.labelsize']= Labelsize
    mpl.rcParams['ytick.labelsize']=Labelsize
   
    mpl.rcParams['axes.linewidth'] = 2.
    
    mpl.rcParams['axes.labelsize']=15
    

def move_figure(f, x, y):
    import matplotlib
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)