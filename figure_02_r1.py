import xarray as xr
import numpy as np
import os,sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, InsetPosition,mark_inset

# load and import non default functions from plot_py3
sys.path.append(os.path.expanduser('~/pyLib/plot_py3'))
sys.path.append(os.path.expanduser('./plot_py3'))
import plotTools as ptool
import mapTools as mtool

#==================================================================================
# define the functions 
#==================================================================================

def get_ij(amz,lats,lons):
    # returns the i,j indices of inset regions
    latsij=[np.argmin(abs(lats-amz[2])),np.argmin(abs(lats-amz[3]))]
    lonsij=[np.argmin(abs(lons-amz[0])),np.argmin(abs(lons-amz[1]))]
    ijout=[np.min(lonsij),np.max(lonsij),np.min(latsij),np.max(latsij)]
    return(ijout)

def plot_inset(inr,_dat,_loc1=2,_loc2=4):
    # plots the inset figures
    mark_inset(_ax, axins, loc1=_loc1, loc2=_loc2, lw=0.3,fc="none", ec="k")
    axins.set_xlim(inr[0],inr[1])
    axins.set_ylim(inr[2],inr[3])
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ijout=get_ij(inr,_dat['lat'].values,_dat['lon'].values)
    dat1_Am=_dat.isel(lat=slice(ijout[2],ijout[3]), lon=slice(ijout[0],ijout[1]))
    _mp2=mtool.def_map_cyl(lonmin=inr[0],lonmax=inr[1],latmin=inr[2],latmax=inr[3],line_w=0.10,labmer=[False,False,False,False],labpar=[False,False,False,False],labfs=ax_fs*0.7,remLab=False,min_coast=True)
    _mp2.imshow(np.ma.masked_equal(dat1_Am[_vari],fill_val),interpolation='none',vmin=valrange[0],vmax=valrange[1],cmap=cm_dia,origin='upper')
    return

#==================================================================================
# variables and the definition of ranges and colorbar
#==================================================================================
varibs='Rn LE H'.split()
cbef='viridis'
varibs_info={
            'LE':['Latent Heat Energy','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef',cbef,[0,11]],\
            'H':['Sensible Heat Flux','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef',cbef,[0,5.5]],\
            'Rn':['Net Radiation','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef',cbef,[0,14]],\
            }

#==================================================================================
# path of the data directory
#==================================================================================

exDrive='/media/skoirala/exStore/'
inDir=exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/'


#==================================================================================
# settings of the plot
#==================================================================================

x0=0.06
y0=0.98

wcolo=0.00925
hcolo=0.15
cb_off_x=0.075
cb_off_y=-0.02158
ax_fs=7
wp=1./2-0.05
hp=1./5
xsp=0.06
aspect_data=0.5
ysp=-0.0
lppadx=0.09
lppady=0.12
lphp=hp*0.6
lpwp=wp*0.8
ncolobin=50
#==================================================================================
# defining the zoom in regions
#==================================================================================

amz=[-68,-48,-10,10]
afr=[20,35,-22,-10]
ind=[74,85,22,30]

#==================================================================================
# Plotting the figure
#==================================================================================

plt.figure(figsize=(7,9))
spnInd=1
fill_val=0
for _vari in varibs:
    varInd=varibs.index(_vari)
    # Read the RS+METEO data
    if _vari == 'Rn':
        dat_1d=xr.open_dataset(inDir+'RS_METEO/ensembles/common/Global/'+_vari+'.RS_METEO.EBC-NONE.MLM-ALL.METEO-ALL.720_360.monthly.2001-2013.ltMean.nc')
    else:
        dat_1d=xr.open_dataset(inDir+'RS_METEO/ensembles/common/Global/'+_vari+'.RS_METEO.EBC-ALL.MLM-ALL.METEO-ALL.720_360.monthly.2001-2013.ltMean.nc')
    dat_1=dat_1d[_vari]
    dat_1n=dat_1d[_vari+'_n'][:]
    # Read the RS data
    if _vari == 'Rn':
        dat_2d=xr.open_dataset(inDir+'RS/ensembles/common/Global/'+_vari+'.RS.EBC-NONE.MLM-ALL.METEO-NONE.4320_2160.monthly.2001-2013.ltMean.nc')
    else:
        dat_2d=xr.open_dataset(inDir+'RS/ensembles/common/Global/'+_vari+'.RS.EBC-ALL.MLM-ALL.METEO-NONE.4320_2160.monthly.2001-2013.ltMean.nc')
    dat_2=dat_2d[_vari]
    valrange=varibs_info[_vari][-1]
    # define the bounds of colorbar
    _bounds_dia=np.linspace(min(valrange),max(valrange),ncolobin)

    cbName=varibs_info[_vari][-2]
    clist_=mtool.get_colomap(cbName,_bounds_dia,lowp=0.03,hip=1.0)
    cm_dia = mpl.colors.ListedColormap(clist_)

    #==================================================================================
    # plot RS+METEO
    #==================================================================================
    _ax=plt.axes([x0+wp+xsp+cb_off_x,y0-hp-varInd*hp-varInd*ysp,wp,hp])
    if varInd == 0:
        _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.1,labmer=[False,False,True,False],labpar=[True,False,False,False],labfs=ax_fs*0.7,remLab=False,min_coast=True)
    elif varInd == len(varibs)-1:
        _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.10,labmer=[False,False,False,True],labpar=[True,False,False,False],labfs=ax_fs*0.7,remLab=False,min_coast=True)
    else:
        _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.10,labmer=[False,False,False,False],labpar=[True,False,False,False],labfs=ax_fs*0.7,remLab=False,min_coast=True)
    _mp.imshow(np.ma.masked_equal(dat_1,fill_val),interpolation='none',vmin=valrange[0],vmax=valrange[1],cmap=cm_dia,origin='upper')
    if varInd == 0:
        plt.title('RS+METEO',fontsize=ax_fs,y=1.05)

    ## plot the inset zoom ins
    axins = zoomed_inset_axes(_ax, 4.4, loc=3)
    plot_inset(amz,dat_1d)
    axins = zoomed_inset_axes(_ax, 3.8795, loc=8)
    plot_inset(afr,dat_1d)
    axins = zoomed_inset_axes(_ax, 4.15, loc=4)
    plot_inset(ind,dat_1d,_loc1=1,_loc2=3)

    #==================================================================================
    # plot RS
    #==================================================================================
    _ax=plt.axes([x0+xsp,y0-hp-varInd*hp-varInd*ysp,wp,hp])

    plt.ylabel(varibs_info[_vari][0],fontsize=ax_fs*1.04,x=-0.185)

    if varInd == 0:
        _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.1,labmer=[False,False,True,False],labpar=[False,True,False,False],labfs=ax_fs*0.7,remLab=False,min_coast=True)
    elif varInd == len(varibs)-1:
        _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.10,labmer=[False,False,False,True],labpar=[False,True,False,False],labfs=ax_fs*0.7,remLab=False,min_coast=True)
    else:
        _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.10,labmer=[False,False,False,False],labpar=[False,True,False,False],labfs=ax_fs*0.7,remLab=False,min_coast=True)

    _mp.imshow(np.ma.masked_equal(dat_2,fill_val),interpolation='none',vmin=valrange[0],vmax=valrange[1],cmap=cm_dia,origin='upper')
    if varInd == 0:
        plt.title('RS',fontsize=ax_fs*1.04,y=1.05)

    ## plot the inset zoom ins
    axins = zoomed_inset_axes(_ax, 4.4, loc=3)
    plot_inset(amz,dat_2d)
    axins = zoomed_inset_axes(_ax, 3.8795, loc=8)
    plot_inset(afr,dat_2d)
    axins = zoomed_inset_axes(_ax, 4.15, loc=4)
    plot_inset(ind,dat_2d,_loc1=1,_loc2=3)

    #==================================================================================
    # plot colorbar
    #==================================================================================

    _axcol_dia=[x0+2*(wp+0.8*xsp)+cb_off_x,y0-(varInd+1)*hp-cb_off_y,wcolo,hcolo]
    cb=mtool.mk_colo_cont(_axcol_dia,_bounds_dia,cm_dia,cbfs=ax_fs,cbrt=0,nticks=5,cb_or='vertical')
    if _vari == 'LE':
        cb.set_label(varibs_info[_vari][1]+' | $\\mathrm{W.m^{-2}}$ | $\\mathrm{mm.d^{-1}}$',fontsize=ax_fs,x=0.1)
    clab=cb.ax.get_yticklabels()
    clabw=[]
    for _clab in clab:
        print (_clab.get_text())
        __clab=float(_clab.get_text())/0.0864
        __clabmm=float(_clab.get_text())/2.45
        p1=_clab.get_text().rjust(5)
        p2=str(round(__clab,1)).rjust(5)
        p3=str(round(__clabmm,1)).ljust(5)
        clabw=np.append(clabw,p1+' | '+p2+' | '+p3)
        print (__clab)
    print(clabw,len(_clab.get_text()))
    cb.ax.set_yticklabels(clabw,fontsize=ax_fs,ha='left')

#==================================================================================
# Save the figure
#==================================================================================

plt.savefig('Figure_02_r1.png',dpi=450,bbox_inches='tight')
plt.savefig('Figure_02_r1.pdf',dpi=450,bbox_inches='tight')

plt.show()
