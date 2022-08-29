import sys, os, os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import scipy.stats as scst
import scipy.odr as scodr
import netCDF4 as nc4

# load and import non default functions from plot_py3
sys.path.append(os.path.expanduser('~/pyLib/plot_py3'))
sys.path.append(os.path.expanduser('./plot_py3'))
import plotTools as ptool
import mapTools as mtool
import statTools as sttool

#==================================================================================
# define the functions 
#==================================================================================

def rem_nan(tmp,_fill_val=-9999.):
    # remove invalid values
    whereisNan=np.isnan(tmp)
    tmp[whereisNan]=_fill_val
    whereisNan=np.isinf(tmp)
    tmp[whereisNan]=_fill_val
    return(tmp)

def apply_mask(_dat,_mask_dat,fill_val=-9999.):
    # apply the mask to data
    mask_where=np.ma.getmask(np.ma.masked_less(_mask_dat,1.))
    _dat[mask_where]=fill_val
    return(_dat)

def get_data(_models_dic,_dat_mask,fill_val=-9999.):
    # read the data of each of the model/obs and save it to a dictionary
    all_mod_dat={}
    _models=list(_models_dic.keys())
    mod_dat_min=[]
    mod_dat_max=[]
    t1='2001'
    t2='2005'
    for _md in _models:
        _mdI=_models.index(_md)
        modInfo=_models_dic[_md]
        datfile=modInfo[0]
        datVar=modInfo[1]
        datCorr=modInfo[3]
        if _md =='mte':
            mod_dat_f=nc4.Dataset(datfile)
            le_m=mod_dat_f['LE'][:][:60,:,:].mean(0)
            mod_dat0=le_m*datCorr
        else:
            mod_dat_f=xr.open_dataset(datfile)
            mod_dat0=mod_dat_f[datVar].sel(time=slice(t1,t2)).mean(dim='time').values
            mod_dat0=mod_dat0*datCorr
            _dat_mask_full=np.ones((np.shape(mod_dat0)))*_dat_mask

        mask_where=np.ma.getmask(np.ma.masked_equal(_dat_mask_full,1.))
        mod_dat=rem_nan(mod_dat0)
        mod_dat=apply_mask(mod_dat,_dat_mask)
        all_mod_dat[_md]=mod_dat
        mod_dat_min=np.append(mod_dat_min,np.nanmin(np.ma.masked_equal(mod_dat,fill_val)))
        mod_dat_max=np.append(mod_dat_max,np.nanmax(np.ma.masked_equal(mod_dat,fill_val)))
    return(all_mod_dat,(np.min(mod_dat_min),np.max(mod_dat_max)))

def get_trim_data(_dat1mc,_dat2mc):
    # remove the outliers
    _absDev=np.abs(_dat1mc-_dat2mc)
    _absDevPerc=np.nanpercentile(_absDev,outlierPerc)
    _outMask=np.ma.getmask(np.ma.masked_greater(_absDev,_absDevPerc))
    _dat1mc[_outMask]=-9999.
    _dat2mc[_outMask]=-9999.
    _dat1mco=np.ma.masked_less(_dat1mc,-9990.).compressed()
    _dat2mco=np.ma.masked_less(_dat2mc,-9990.).compressed()
    return(_dat1mco,_dat2mco)

def fLin(B, x):
    '''Linear function y = m*x + b'''
    return B[0]*x + B[1]

def fit_odr(x,y):
    # fit orthogonal (total) least squares
    linearF = scodr.Model(fLin)
    mydata = scodr.Data(x, y)
    slope, intercept, r_value, p_value, std_err = scst.linregress(x, y)
    myodr = scodr.ODR(mydata, linearF, beta0=[slope, intercept])
    myoutput = myodr.run()
    myoutput.pprint()
    return(myoutput.beta[0],myoutput.beta[1])


#==================================================================================
# variables and the definition of ranges and colorbar
#==================================================================================

varibs_info={
            'LE':['Latent Heat Energy','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef','viridis'],\
            'H':['Sensible Heat Flux','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef','viridis'],\
            'Rn':['Net Radiation','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef','viridis'],\
            }

#==================================================================================
# path of the data directory
#==================================================================================

filesep='/'
exDrive='/media/skoirala/exStore/'
tscale_an='monthly'
fill_val=-9999.


#==================================================================================
# settings of the mask directory, reading of the land grid area and so on
#==================================================================================

maskDir=exDrive+'FLUXCOM_Eflux/data/masks/'
mte_mask_dat=np.fromfile(maskDir+filesep+'mask_rsm_common_obs.hlf',np.float32).reshape(360,720)

mo_fill_mask=np.ma.getmask(np.ma.masked_equal(mte_mask_dat,0))

ar_w = nc4.Dataset(maskDir+filesep+'/landfraction/landfraction.720.360.nc')['landfraction'][:]
grArea=np.fromfile(maskDir+filesep+'/grarea_grid.hlf',np.float32).reshape(360,720)
ar_w=(np.ma.masked_equal(ar_w,-9999.).filled(0)/100.)
ar_w[ar_w<0.8] = 0.
mte_mask_dat[ar_w<0.8] = 0.
ar_w=ar_w*grArea*mte_mask_dat



#==================================================================================
# settings of the plot
#==================================================================================
x0=0.02
y0=1.0
wcolo=0.25
cb_off_x=wcolo
cb_off_y=0.07158
ax_fs=3

plotVar='LE'


#==================================================================================
# definitions of the data to be plotted with path, labels, and multipliers for unit
#==================================================================================

dats={
'rs':[exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/RS/ensembles/common/Global/LE.RS.EBC-ALL.MLM-ALL.METEO-NONE.720_360.monthly.2001-2013.nc','LE','RS',1],\
'rsm':[exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/RS_METEO/ensembles/common/Global/LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-ALL.720_360.monthly.2001-2013.nc','LE','RS+METEO',1],\
'mte':[exDrive+'FLUXCOM_Eflux/data/obs/MTE/LE.MTE.ERAI.monthly.2001-2010.nc','LE','MTE',1],\
'gl':[exDrive+'FLUXCOM_Eflux/data/obs/E_gleam_v3.1a/E.gleam.720.360.monthly.2001-2010.nc','E','GLEAM (v3.1a)',2.45],\
'lfe':[exDrive+'FLUXCOM_Eflux/data/obs/lfe_hlf/ET_median/ET_median.all.720.360.204.2001-2005.nc','ET_median','LandFlux-EVAL',2.45],\
}

models='rs rsm mte gl lfe'.split()

#==================================================================================
# MORE settings of the plot based on what is to be plotted
#==================================================================================

nmodels=len(models)

wp=1./nmodels
hp=wp
xsp=0.0
aspect_data=np.shape(mte_mask_dat)[0]/(np.shape(mte_mask_dat)[1]*1.)
ax_fs=7.1
ysp=-0.04
xsp_sca=wp/3*(aspect_data)
ysp_sca=hp/3*(aspect_data)
hcolo=0.055*hp
all_mod_dat,valrange=get_data(dats,mte_mask_dat)
valrange=(0,10)
valrangemd=(0,10)
valrangehb=(0,15)
outlierPerc=98


#==================================================================================
# Plotting the figure
#==================================================================================

fig=plt.figure(figsize=(9,6))
for row_m in range(nmodels):
    row_mod=models[row_m]
    mod_dat_row=all_mod_dat[row_mod]
    mod_dat_row=apply_mask(mod_dat_row,mte_mask_dat)
    for col_m in range(nmodels):
        col_mod=models[col_m]
        mod_dat_col=all_mod_dat[col_mod]
        mod_dat_col=apply_mask(mod_dat_col,mte_mask_dat)

        #==================================================================================
        # Maps along the diagonal
        #==================================================================================

        if row_m == col_m:
            _ax=plt.axes([x0+row_m*wp+row_m*xsp,y0-(col_m*hp+col_m*ysp),wp,hp])
            _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.05,min_coast=True)
            _mp._ax=_ax
            _bounds_dia=np.linspace(min(valrangemd),max(valrangemd),100)


            cbName=varibs_info[plotVar][-1]
            clist_=mtool.get_colomap(cbName,_bounds_dia,lowp=0.03,hip=0.95)
            cm_dia = mpl.colors.ListedColormap(clist_)

            gl_mean=sttool.calc_globmean(mod_dat_row,_fill_val=fill_val)
            gl_mean=np.nansum(np.ma.masked_equal(mod_dat_row,fill_val).filled(0)*ar_w)/2.45*1e-15*365.5
            gl_mean=np.nansum(np.ma.masked_equal(mod_dat_row,fill_val).filled(0)*ar_w)/np.nansum(ar_w)#2.45*1e-15*365.5

            _mp.imshow(np.ma.masked_equal(mod_dat_row,fill_val),interpolation='none',vmin=valrangemd[0],vmax=valrangemd[1],cmap=cm_dia,origin='upper')
            tit_str="$\\mu_{LE}$ = "+str(round(gl_mean,2))+' ('+varibs_info[plotVar][1]+')'
            plt.text(0,-70,tit_str,fontsize=ax_fs*0.939,ha='center',weight='bold')
        
        #==================================================================================
        # Density plots below the diagonal
        #==================================================================================

        if row_m < col_m:
            _ax=plt.axes([x0+row_m*wp+row_m*xsp+xsp_sca,y0-(col_m*hp+col_m*ysp)+ysp_sca,wp*aspect_data,hp*aspect_data])#,sharex=right,sharey=all)
            _ax.hexbin(np.ma.masked_equal(mod_dat_col,fill_val).flatten(),np.ma.masked_equal(mod_dat_row,fill_val).flatten(),bins='log',mincnt=1, gridsize=50, cmap='gist_gray_r',linewidths=0)
            plt.ylim(valrangehb[0],valrangehb[1]*1.05)
            plt.xlim(valrangehb[0],valrangehb[1]*1.05)
            ymin,ymax = plt.ylim()
            xmin,xmax = plt.xlim()
            plt.plot((xmin,xmax),(ymin,ymax),'k',lw=0.1)
            dat1tr,dat2tr=get_trim_data(mod_dat_col.flatten(),mod_dat_row.flatten())

            r,p=sttool.calc_pearson_r(dat1tr,dat2tr)
            tit_str="$R^2$="+str(round(r**2,2))
            slope, intercept, r_value, p_value, std_err = scst.linregress(dat1tr, dat2tr)
            slope,intercept=fit_odr(dat1tr,dat2tr)
            if intercept >= 0:
                eqnTxt="y="+str(round(slope,2))+'x + '+str(round(intercept,2))
            else:
                eqnTxt="y="+str(round(slope,2))+'x - '+str(abs(round(intercept,2)))

            yFitted=intercept + slope*dat1tr
            plt.plot(dat1tr, yFitted, 'r', lw=0.5)
            
            plt.text(0.5,0.93,eqnTxt,color='red',fontsize=ax_fs*0.893,ha='center',weight='bold',transform=plt.gca().transAxes)
            plt.title(tit_str,fontsize=ax_fs*0.89,ma='left',y=1.075,va="top",weight='bold')
            if row_m !=0 and col_m != nmodels-1:
                ptool.ax_clr(axfs=ax_fs)
            elif row_m == 0 and col_m != nmodels-1:
                ptool.ax_clrX(axfs=ax_fs)
            elif col_m == nmodels-1 and row_m !=0 :
                ptool.ax_clrY(axfs=ax_fs)
            if row_m == 0 and col_m == nmodels-1:
                ptool.ax_orig(axfs=ax_fs)
                plt.ylabel('Column',fontsize=ax_fs)
                plt.xlabel('Row',fontsize=ax_fs)
        
        #==================================================================================
        # Difference maps above the diagonal
        #==================================================================================

        if row_m > col_m:
            _ax=plt.axes([x0+row_m*wp+row_m*xsp,y0-(col_m*hp+col_m*ysp),wp,hp])
            _mp=mtool.def_map_cyl(lonmin=-180,lonmax=180,line_w=0.05,min_coast=True)
            _mp._ax=_ax
            _bounds_rat=np.linspace(-3,3,100)
            clist_=mtool.get_colomap('coolwarm',_bounds_rat,lowp=0.05,hip=0.95)
            cm_rat = mpl.colors.ListedColormap(clist_)
            plot_dat=rem_nan(mod_dat_row-mod_dat_col)
            plot_dat=apply_mask(plot_dat,mte_mask_dat)
            _mp.imshow(np.ma.masked_equal(plot_dat,fill_val),interpolation='none',vmin=-3,vmax=3,cmap=cm_rat,origin='upper')
        if row_m == nmodels-1:
            _title_sp=dats[col_mod][2]
            plt.ylabel(_title_sp,fontsize=0.809*ax_fs)
            plt.gca().yaxis.set_label_position("right")
        if col_m == 0:
            _title_sp=dats[row_mod][2]
            plt.title(_title_sp,fontsize=0.809*ax_fs)


t_x=plt.figtext(0.5,0.5,' ',transform=plt.gca().transAxes)

#==================================================================================
# plot colorbar
#==================================================================================

# for the diagonal
x_colo=0.03
y_colo=y0+hp+cb_off_y-0.01
_axcol_dia=[x_colo,y_colo,wcolo,hcolo]
cb_tit=varibs_info[plotVar][0]+' ('+varibs_info[plotVar][1]+')'
mtool.mk_colo_cont(_axcol_dia,_bounds_dia,cm_dia,cbfs=0.95*ax_fs,cbtitle=cb_tit,cbrt=90)

# for the difference
cb_tit='Difference (Column - Row)'
x_colo=0.76
y_colo=y0+hp+cb_off_y-0.01
_axcol_rat=[x_colo,y_colo,wcolo,hcolo]
mtool.mk_colo_cont(_axcol_rat,_bounds_rat,cm_rat,cbfs=0.95*ax_fs,cbrt=90,cbtitle=cb_tit)

#==================================================================================
# Save the figure
#==================================================================================

plt.savefig('Figure_04_r1.png',bbox_inches='tight',bbox_extra_artists=[t_x],dpi=450)
plt.savefig('Figure_04_r1.pdf',bbox_inches='tight',bbox_extra_artists=[t_x],dpi=450)
plt.show()
