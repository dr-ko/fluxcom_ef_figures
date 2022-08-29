import xarray as xr
import numpy as np
import os,sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import string

# load and import non default functions from plot_py3
sys.path.append(os.path.expanduser('~/pyLib/plot_py3'))
sys.path.append(os.path.expanduser('./plot_py3'))
import plotTools as ptool
import mapTools as mtool

#==================================================================================
# define the functions 
#==================================================================================

def norm_values(_dat0):
    # normalizes the data to the global maximum
    _dat0=np.ma.masked_less(_dat0,0).filled(0)
    _dat=_dat0
    _dat2=abs(np.nanpercentile(np.ma.masked_less_equal(_dat0,0.).compressed(),2))
    _dat98=abs(np.nanmax(_dat0))
    _dat[_dat0 <= _dat2] = _dat2
    _dat[_dat0 >= _dat98] = _dat98
    _dat=_dat/_dat98
    _dat=np.ma.masked_equal(_dat,0).filled(np.nan)
    return(_dat,_dat0)

def abc_to_rgb(A=0.0,B=0.0,C=0.0):
    ''' Map values A, B, C (all in domain [0,1]) to
    suitable red, green, blue values.'''
    return (A,min(B/A,1),min(C/A,1))

def plot_legend(ax,fontsize=9):
    ''' Plots a legend for the colour scheme
    given by abc_to_rgb. Includes some code adapted
    from http://stackoverflow.com/a/6076050/637562'''
    npointsLeg=50j
    basis = np.array([[0.0, 1.0], [-1.5/np.sqrt(3), -0.5],[1.5/np.sqrt(3), -0.5]])
    a, b, c = np.mgrid[0.0:1.0:npointsLeg, 0.0:1.0:npointsLeg, 0.0:1.0:npointsLeg]
    a, b, c = a.flatten(), np.roll(b.flatten(),25), c.flatten()

    abc = np.dstack((a,b,c))[0]
    abc = list(map(lambda x: x/sum(x), abc))

    data = np.dot(abc, basis).real
    data = np.ma.fix_invalid(data, fill_value=-9999.)    
    colours = [abc_to_rgb(A=point[0],B=point[1],C=point[2]) for point in abc]
    colours=list(np.ma.fix_invalid(np.array(colours), fill_value=1.))
    ax.scatter(data[:,0], data[:,1],marker='d',edgecolors='none',facecolors=colours,s=0.4)

    ax.plot([basis[_,0] for _ in list(range(3)) + [0,]],[basis[_,1] for _ in list(range(3)) + [0,]],**{'color':'white','linewidth':3})
    offset = 0.35
    ax.text(basis[0,0]*(1+offset), basis[0,1]*(1+offset), '$H$', horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
    ax.text(basis[1,0]*(1+offset), basis[1,1]*(1+offset), '$\\frac{LE}{Rn}$', horizontalalignment='center',
            verticalalignment='center', fontsize=1.5*fontsize)
    ax.text(basis[2,0]*(1+offset), basis[2,1]*(1+offset), '$LE$', horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
    ax.set_frame_on(False)
    ax.set_xticks(())
    ax.set_yticks(())


filesep='/'
#==================================================================================
# variables and the definition of ranges and colorbar
#==================================================================================

varibs='LE H Rn'.split()
varibs_info={
            'LE':['Latent Heat Energy','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef','jet',[0,10]],\
            'H':['Sensible Heat Flux','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef','jet',[0,8]],\
            'Rn':['Net Radiation','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef','jet',[0,15]],\
            }

exDrive='/media/skoirala/exStore/'

#==================================================================================
# defining the selected points
#==================================================================================

selPoints={'NA':[[-110,40],[-150,40],'North\nAmerica'],'EU':[[40,50],[40,80],'Europe'],'AS':[[81,28],[65,8],'Asia'],'SA':[[-57,-5],[-37,15],'South\nAmerica'],'AF':[[27,-10],[-10,-10],'Africa'],'AU':[[150,-29],[161,-45],'Australia']}
nrows=len(selPoints.keys())+1
ncols=len(varibs)

#==================================================================================
# settings of the plot
#==================================================================================

x0=0.06
y0=0.98
wcolo=0.25
hcolo=0.00925
cb_off_x=0.025
cb_off_y=-0.02158
ax_fs=9
wp=0.5
hp=0.5
xsp=0.01
aspect_data=0.5
ysp=0.02

selPo=list(selPoints.keys())

plt.figure(figsize=(9,8))
prodInd=0

#==================================================================================
# settings of the mask directory
#==================================================================================

maskDir=exDrive+'FLUXCOM_Eflux/data/masks/'

mte_mask_dat=np.fromfile(maskDir+filesep+'mask_rsm_all.hlf',np.float32).reshape(360,720)[0:300,:]
mo_fill_mask=np.ma.getmask(np.ma.masked_equal(mte_mask_dat,0))

#==================================================================================
# Plotting the figures
#==================================================================================

# for prod in ['RS_METEO']:
for prod in ['RS','RS_METEO']:
    #==================================================================================
    # Read the data
    #==================================================================================
    datDir=exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/'+prod+filesep+'/ensembles/common/Global/'
    if prod == 'RS':
        prodN=prod
        fileadd='NONE'
    else:
        prodN='RS+METEO'
        fileadd='ALL'
    
    for _vari in varibs:
        varInd=varibs.index(_vari)
        if _vari == 'Rn':
            dat_=xr.open_dataset(datDir+filesep+_vari+'.'+prod+'.EBC-NONE.MLM-ALL.METEO-'+fileadd+'.720_360.monthly.2001-2013.nc')
        else:
            dat_=xr.open_dataset(datDir+filesep+_vari+'.'+prod+'.EBC-ALL.MLM-ALL.METEO-'+fileadd+'.720_360.monthly.2001-2013.nc')
        dat_m=dat_[_vari]
        datMean=np.mean(dat_m,axis=0)

        dat0,dat1=norm_values(datMean.values[0:300,:])
        vars()['dat_'+str(varInd+1)]=dat0
        vars()['odat_'+str(varInd+1)]=dat1
        lats=dat_['lat']
        lons=dat_['lon']

    # set the RGB values
    datRGB = np.ones((np.shape(dat_1)[0],np.shape(dat_1)[1],4))
    datInv=np.ma.masked_invalid(datMean[0:300,:]).mask
    dat_1[datInv]=1
    dat_2[datInv]=1
    odat_1[datInv]=1
    odat_3[datInv]=1
    datRGB[..., 2] = dat_1
    datRGB[..., 0] = dat_2
    datRGB[..., 1] = odat_1/odat_3
    fill_val=0
    valrange=[0,1]

    #==================================================================================
    # plot the map
    #==================================================================================
    
    # define the plotting axis of the map
    _ax=plt.axes([x0+prodInd*wp+prodInd*xsp,y0-hp,wp,hp])
    if prod == 'RS_METEO':
        _mp=mtool.def_map_cyl(latmin=-60,lonmin=-180,lonmax=180,line_w=0.3,labmer=[False,False,False,True],labpar=[False,True,False,False],labfs=0.7*ax_fs,remLab=True,min_coast=True)

    else:
        _mp=mtool.def_map_cyl(latmin=-60,lonmin=-180,lonmax=180,line_w=0.3,labmer=[False,False,False,True],labpar=[True,False,False,False],labfs=0.7*ax_fs,remLab=True,min_coast=True)
    _mp.imshow(np.ma.masked_less(datRGB,fill_val+0.01),interpolation='none',origin='upper')
    plt.title(prodN,fontsize=ax_fs*1.25,y=1.02)

    # plot the locations of the points with the arrow
    if prodInd == 0:
        for _np in range(len(selPo)):
            lon_=selPoints[selPo[_np]][0][0]
            lat_=selPoints[selPo[_np]][0][1]
            plt.annotate(string.ascii_uppercase[_np],xy=(lon_, lat_),xytext=(selPoints[selPo[_np]][1][0],selPoints[selPo[_np]][1][1]),color='k',fontsize=ax_fs*.90,ha='center',va='center', arrowprops={'arrowstyle':'->','linewidth':0.7},bbox=dict(boxstyle="round",ec=None,fc='#ffff33',linewidth=0,pad=0.1))            #     plt.annotate('+'+string.ascii_uppercase[_np],xy=(lon_, lat_),color='#ffee00',fontsize=ax_fs*.95,ha='left',va='center',weight='bold')
        axC=plt.axes([x0+(prodInd)*wp+(prodInd)*xsp+0.032,y0-hp+0.15,0.08,0.08])
        plot_legend(axC,fontsize=0.7*ax_fs)

    #==================================================================================
    # plot the time series
    #==================================================================================
    
    # settings of the line plots such as distance and height and width
    lppadx=0.061
    lppady=0.04
    lphp=0.14
    lpwp=0.45

    # slice the timeseries for the period of interest
    t1='2001-01'
    t2='2005-12'
    led=xr.open_dataset(datDir+filesep+'LE.'+prod+'.EBC-ALL.MLM-ALL.METEO-'+fileadd+'.720_360.monthly.2001-2013.nc')
    le_m=led['LE'].sel(time=slice(t1,t2))
    le_mad=led['LE_mad'].sel(time=slice(t1,t2))
    hd=xr.open_dataset(datDir+filesep+'H.'+prod+'.EBC-ALL.MLM-ALL.METEO-'+fileadd+'.720_360.monthly.2001-2013.nc')
    h_m=hd['H'].sel(time=slice(t1,t2)).sel(time=slice(t1,t2)).sel(time=slice(t1,t2))
    h_mad=hd['H_mad'].sel(time=slice(t1,t2)).sel(time=slice(t1,t2))
    rnd=xr.open_dataset(datDir+filesep+'Rn.'+prod+'.EBC-NONE.MLM-ALL.METEO-'+fileadd+'.720_360.monthly.2001-2013.nc')
    rn_m=rnd['Rn'].sel(time=slice(t1,t2))
    rn_mad=rnd['Rn_mad'].sel(time=slice(t1,t2))

    # loop through and plot the time series
    for _np in range(len(selPo)):
        lon_=selPoints[selPo[_np]][0][0]
        lat_=selPoints[selPo[_np]][0][1]
        le_po=le_m.sel(lat=lat_,lon=lon_,method='nearest')
        le_mad_po=le_mad.sel(lat=lat_,lon=lon_,method='nearest')
        h_po=h_m.sel(lat=lat_,lon=lon_,method='nearest')
        h_mad_po=h_mad.sel(lat=lat_,lon=lon_,method='nearest')
        rn_po=rn_m.sel(lat=lat_,lon=lon_,method='nearest')
        rn_mad_po=rn_mad.sel(lat=lat_,lon=lon_,method='nearest')
        xdat=np.linspace(2001,2006,len(le_po),endpoint=True)
        tdat=le_po['time']
        _ax=plt.axes([x0+prodInd*wp+prodInd*xsp+lppadx,y0-hp-ysp-(_np)*(lphp+ysp)-lppady,lpwp,lphp])
        ptool.ax_clrXY(axfs=ax_fs)
        p1=_ax.plot(xdat,rn_po,'-',color='#ff8f00',lw=1.0,marker=None)
        plt.fill_between(xdat,rn_po-1.4826*rn_mad_po,rn_po+1.4826*rn_mad_po,color='#ff8f00',alpha=0.3,linewidth=0.)
        p2=_ax.plot(xdat,le_po,'-',color='#039be5',lw=1.0,marker=None)
        plt.fill_between(xdat,le_po-1.4826*le_mad_po,le_po+1.4826*le_mad_po,color='#039be5',alpha=0.3,linewidth=0.)
        p3=_ax.plot(xdat,h_po,'-',color='#e53935',lw=1.0,marker=None)
        plt.fill_between(xdat,h_po-1.4826*h_mad_po,h_po+1.4826*h_mad_po,color='#e53935',alpha=0.3,linewidth=0.)
        if _np != len(selPo)-1:
            ptool.rem_ticklabels(which_ax='x')
        else:
            ptool.rotate_labels(which_ax='x',rot=90,axfs=ax_fs)

        if _np == 0:
            if prodInd == 0:
                plt.title("Energy Fluxes ($\\mathrm{MJ.m^{-2}.d^{-1}}$)",y=0.94,fontsize=ax_fs*0.98)
            else:
                leg=plt.legend([p1[0],p2[0],p3[0]],('Rn','LE','H'),loc=(0.29371,0.94),ncol=3,fontsize=ax_fs*.98,frameon=False,edgecolor=None,handlelength=0.8)
                for line in leg.get_lines():
                    line.set_linewidth(1.2)
        # set the y-axis lables with lat and lon
        if prodInd == 0:
            if lon_ < 0:
                lonStr='%.2f'%(abs(le_po['lon'].values))+'$^\\circ W$'
            elif lon_ > 0:
                lonStr='%.2f'%(abs(le_po['lon'].values))+'$^\\circ E$'
            else:
                lonStr='%d'%(abs(0))
            if lat_ < 0:
                latStr='%.2f'%(abs(le_po['lat'].values))+'$^\\circ S$'
            elif lat_ > 0:
                latStr='%.2f'%(abs(le_po['lat'].values))+'$^\\circ N$'
            else:
                latStr='%d'%(abs(0))
            print(lonStr,le_po['lon'].values)

            plt.ylabel(string.ascii_uppercase[_np]+': '+selPoints[selPo[_np]][2]+'\n('+lonStr+','+latStr+')',fontsize=ax_fs*0.86)
        plt.xlim(2000.9,2006.1)
        plt.ylim(-1,16)


    prodInd=prodInd+1

#==================================================================================
# Save the figure
#==================================================================================
plt.savefig('Figure_03_r1.png',dpi=450,bbox_inches='tight')
plt.savefig('Figure_03_r1.pdf',dpi=450,bbox_inches='tight')

plt.show()
