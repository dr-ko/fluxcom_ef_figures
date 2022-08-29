#!/opt/local/python/Python-2.7.2/bin/python
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import shiftgrid
from matplotlib import ticker

import os

def get_colomap(cmap_nm,bounds__,lowp=0.05,hip=0.95):
    '''
    Get the list of colors from any official colormaps in matplotlib. It returns the number of colors based on the number of items in the bounds. Bounds is a list of boundary for each color.
    '''
    cmap__ = mpl.cm.get_cmap(cmap_nm)
    clist_v=np.linspace(lowp,hip,len(bounds__)-1)
    rgba_ = [cmap__(_cv) for _cv in clist_v]
    return(rgba_)
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def get_ticks(_bounds, nticks=10):
    from math import ceil
    length = float(len(_bounds))
    co_ticks=[]
    for i in range(nticks):
        co_ticks=np.append(co_ticks,_bounds[int(ceil(i * length / nticks))])
    return(co_ticks[1:])
def mk_colo_cont(axcol_,bounds__,cm2,cblw=0.1,cbrt=0,cbfs=9,nticks=10,cbtitle='',col_scale='linear',tick_locs=[],ex_tend='both',cb_or='horizontal',spacing= 'uniform'):
    '''
    Plots the colorbar to the axis given by axcol_. Uses arrows on two sides.
    '''

    axco1=plt.axes(axcol_)
    if col_scale == 'linear':
        cb=mpl.colorbar.ColorbarBase(axco1,cmap=cm2, norm=mpl.colors.BoundaryNorm(bounds__, cm2.N),boundaries=bounds__,orientation = cb_or,drawedges=False,extend=ex_tend,ticks=bounds__[1:-1],spacing=spacing)
        tick_locator = ticker.MaxNLocator(nbins=nticks,min_n_ticks=nticks)
        cb.locator = tick_locator
        cb.update_ticks()
    if col_scale == 'log':
#        cb=mpl.colorbar.Colorbar()#,orientation = 'horizontal',drawedges=False,ticks=bounds__[1:-1])
#        cb=mpl.colorbar.ColorbarBase(axco1,cmap=cm2, norm=mpl.colors.LogNorm(vmin=bounds__.min(),vmax=bounds__.max()), boundaries=bounds__,orientation = 'horizontal',extend='both',drawedges=False,ticks=bounds__[1:-1])

        cb=mpl.colorbar.ColorbarBase(axco1,cmap=cm2, norm=mpl.colors.BoundaryNorm(bounds__, cm2.N), boundaries=bounds__,orientation = cb_or,extend=ex_tend,drawedges=False,ticks=bounds__[1:-1],spacing=spacing)
#        print bounds__,np.log(np.array(tick_locs)),np.exp(np.log(np.array(tick_locs)))
        tick_locs_ori=cb.ax.xaxis.get_ticklocs()
        tick_locs_bn=[]
        for _tl in tick_locs:
            tlInd=np.argmin(np.abs(bounds__[1:-1]-_tl))
            tick_locs_bn=np.append(tick_locs_bn,tick_locs_ori[tlInd])
#        tick_locs_bn=np.log(np.array(tick_locs))/np.log(np.array(tick_locs)).max()
#        tick_locator = ticker.FixedLocator(tick_locs_bn)
#        print cb.ax.xaxis.get_ticklocs()
        cb.ax.xaxis.set_ticks(tick_locs_bn)
        cb.ax.xaxis.set_ticklabels(tick_locs)
#        cb.set_ticklabels(tick_locs,update_ticks=True)
#        cb.ax.set_xticks([cb.vmin + t*(cb.vmax-cb.vmin) for t in cb.ax.get_xticks()])
#        cb.set_ticklabels([np.exp(cb.vmin + t*(cb.vmax-cb.vmin)) for t in cb.ax.get_xticks()])

#        tick_locator = ticker.SymmetricalLogLocator(base=10,linthresh=1,subs=(1,10,100,1000))#,numticks=4)#,subs=(0.25,1,2,4))
    cb.ax.tick_params(labelsize=cbfs,size=2,width=0.3)
#    print cb.ax.xaxis.get_ticklocs()
#    cb.ax.set_xscale(col_scale)
    ##hack the lines of the colorbar to make them white, the same color of background so that the colorbar looks broken.
    cb.outline.set_alpha(0.)
    cb.outline.set_color('white')
    cb.outline.set_linewidth(0*cblw)
    '''
    cb.dividers.set_linewidth(0*cblw)
    cb.dividers.set_alpha(1.0)
    cb.dividers.set_color('white')
    for ll in cb.ax.xaxis.get_ticklines():
        ll.set_alpha(0.)
    '''
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(cbfs)
        t.set_rotation(cbrt)

    for t in cb.ax.get_xticklabels():
        t.set_fontsize(cbfs)
        t.set_rotation(cbrt)
        t.set_y(-0.02)
    if cbtitle != '':
        cb.ax.set_title(cbtitle,fontsize=1.3*cbfs)
    return(cb)
#    cb.update_ticks()

def get_min_coast():
    from mpl_toolkits.basemap import Basemap
    class Basemap(Basemap):
        """ Modify Basemap to use Natural Earth data instead of GSHHG data """
        def drawcoastlines(self,linewidth=1,color='gray'):
            # shapefile = '/Volumes/Kaam/ResearchWork/crescendo/crescendo_tau/analyze/plotData/plot_py3/data/ne_coastline/ne_%sm_coastline' % \
            shapefile = '/home/skoirala/research/crescendo_tau/analyze/plotData/plot_py3/data/ne_coastline/ne_%sm_coastline' % \
                        {'l':110, 'm':50, 'h':10}[self.resolution]
            self.readshapefile(shapefile, 'coastline', linewidth=linewidth,color=color)
    return(Basemap)

def def_map_rob(lonv=0,latv=0,line_w=0.5,labmer=[False,False,False,False],labpar=[False,False,False,False],labfs=4,remLab=False,min_coast=False):
    if min_coast:
        Basemap=get_min_coast()
    else:
        from mpl_toolkits.basemap import Basemap,shiftgrid
    _map = Basemap(projection='robin', resolution = 'l', lon_0=lonv)
    _map.drawcoastlines(linewidth=0.1)
    parallels = np.arange(-90.,90,15.)
    # labels = [left,right,top,bottom]
    _map.drawparallels(parallels,labels=labpar,linewidth=line_w,color='gray',zorder=1,fontsize=labfs)
    meridians = np.arange(0.,360,60.)
    _map.drawmeridians(meridians,labels=labmer,linewidth=line_w,color='gray',zorder=1,fontsize=labfs)
#    _map.drawmapboundary(color=None, linewidth=0)
#    _map.fillcontinents(color='#cccccc',zorder=0)
    return(_map)
def def_map_orth(lonv=0,latv=0,line_w=0.5,labmer=[False,False,False,False],labpar=[False,False,False,False],labfs=4,remLab=False,min_coast=False):
    if min_coast:
        Basemap=get_min_coast()
    else:
        from mpl_toolkits.basemap import Basemap,shiftgrid
    _map = Basemap(projection='ortho', resolution = 'l',lat_0=latv, lon_0=lonv)
    _map.drawcoastlines(linewidth=0.1)
    parallels = np.arange(-90.,90,15.)
    # labels = [left,right,top,bottom]
    _map.drawparallels(parallels,labels=labpar,linewidth=line_w,color='gray',zorder=1,fontsize=labfs)
    meridians = np.arange(0.,360,60.)
    _map.drawmeridians(meridians,labels=labmer,linewidth=line_w,color='gray',zorder=1,fontsize=labfs)
#    _map.drawmapboundary(color=None, linewidth=0)
    return(_map)

def def_map_cyl(lonmin=0,lonmax=360,lonint=60,latmin=-90,latmax=90,latint=30,line_w=0.5,labmer=[False,False,False,False],labpar=[False,False,False,False],labfs=4,remLab=False,min_coast=False):
    if min_coast:
        Basemap=get_min_coast()
    else:
        from mpl_toolkits.basemap import Basemap,shiftgrid

    _map=Basemap( projection ='cyl',  \
                fix_aspect  = True,  \
                 llcrnrlon  = lonmin,  \
                 urcrnrlon  = lonmax,  \
                 llcrnrlat  = latmin,  \
                 urcrnrlat  =  latmax,  \
                 resolution = 'l')    #use i for intermediate, c for coarse
    _map.drawcoastlines(linewidth=line_w, color='gray')
    parallels = np.arange(latmin+latint,latmax,latint)
    _map.drawparallels(parallels,labels=labpar,linewidth=line_w,color='gray',zorder=0,fontsize=labfs)
    meridians = np.arange(lonmin+lonint,lonmax,lonint)
    _map.drawmeridians(meridians,labels=labmer,linewidth=line_w,color='gray',zorder=0,fontsize=labfs)
    # print('here')
    ax=plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right','bottom','left','top']:
            spine.set_position(('outward',0)) # outward by 10 points
            spine.set_linewidth(0.0) # outward by 10 points
            spine.set_linestyle('solid') # outward by 10 points
        else:
            raise ValueError('unknown spine location: %s'%loc)

    if remLab == True:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")
    return(_map)

def def_map_cea(lonmin=0,lonmax=360,lonint=60,latmin=-90,latmax=90,latint=30,line_w=0.5,labmer=[False,False,False,False],labpar=[False,False,False,False],labfs=4,remLab=False,min_coast=False):
    if min_coast:
        Basemap=get_min_coast()
    else:
        from mpl_toolkits.basemap import Basemap,shiftgrid

    _map=Basemap( projection ='cea',  \
                fix_aspect  = True,  \
                 llcrnrlon  = lonmin,  \
                 urcrnrlon  = lonmax,  \
                 llcrnrlat  = latmin,  \
                 urcrnrlat  =  latmax,  \
                 resolution = 'l')    #use i for intermediate, c for coarse
    _map.drawcoastlines(linewidth=line_w, color='gray')
    parallels = np.arange(latmin+latint,latmax,latint)
    _map.drawparallels(parallels,labels=labpar,linewidth=line_w,color='gray',zorder=0,fontsize=labfs)
    meridians = np.arange(lonmin+lonint,lonmax,lonint)
    _map.drawmeridians(meridians,labels=labmer,linewidth=line_w,color='gray',zorder=0,fontsize=labfs)
    # print('here')
    ax=plt.gca()
    for loc, spine in ax.spines.items():
        if loc in ['right','bottom','left','top']:
            spine.set_position(('outward',0)) # outward by 10 points
            spine.set_linewidth(0.0) # outward by 10 points
            spine.set_linestyle('solid') # outward by 10 points
        else:
            raise ValueError('unknown spine location: %s'%loc)

    if remLab == True:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")
    return(_map)