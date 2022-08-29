import numpy as np
import os,sys
import matplotlib.pyplot as plt

# load and import non default functions from plot_py3
sys.path.append(os.path.expanduser('~/pyLib/plot_py3'))
sys.path.append(os.path.expanduser('./plot_py3'))
import plotTools as ptool

#==================================================================================
# define the functions 
#==================================================================================

def get_file_list(mainDir,varName):
    # get the list of all files of all members of RS and RS+METEO runs    
    fileList={}
    coloList=[]
    for prd in ['RS','RS_METEO']:
        fileListprd=[]
        checkDir=mainDir+filesep+prd+filesep+'members'+filesep+'common'+filesep+_regn+filesep
        print(checkDir)
        for root, dirs, files in os.walk(checkDir):
            for file in files:
                if file.startswith('commonLandSeaMask.'+_regn+"."+varName) and file.endswith(".txt") and 'v6' not in file:
                    infile=os.path.join(root, file)
                    fileListprd=np.append(fileListprd,infile)
                    if 'RS_METEO' in infile:
                        coloList=np.append(coloList,rsmcolo)
                    else:
                        coloList=np.append(coloList,rscolo)
        fileList[prd]=fileListprd
    return(fileList,coloList)
def get_data(fileList):
    # get the data from the list of all files of all members of RS and RS+METEO runs    
    datOutD={}
    for _prd in list(fileList.keys()):
        fileListprd=fileList[_prd]
        datOutD[_prd]={}
        for _conti in range(7):
            datTmp=np.zeros((60,len(fileListprd)))
            fileInd=0
            for infile in fileListprd:
                dat=np.loadtxt(infile,skiprows=3,delimiter=',')
                mod_dat=dat[:60,2+_conti]
                datTmp[:,fileInd]=mod_dat
                fileInd=fileInd+1
            datOutD[_prd][str(_conti+1)]=datTmp
    return(datOutD)


#==================================================================================
# variables and the definition of ranges and colorbar
#==================================================================================

cbef='nipy_spectral_r'
varibs_info={
            'LE':['Latent Heat Energy','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef',cbef],\
            'H':['Sensible Heat Flux','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef',cbef,[0,6.5]],\
            'Rn':['Net Radiation','$\\mathrm{MJ.m^{-2}.d^{-1}}$','ef',cbef,[0,15]]
}

#==================================================================================
# define the region separation and the color to use for RS and RS+METEO runs
#==================================================================================

_regn='Continent'
rscolo='#16A085'
rsmcolo='#E67E22'

basis_regions={'1':'Asia','2':'North\nAmerica','3':'Europe','4':'Africa','5':'South\nAmerica','6':'Oceania','7':'Global'}
bas_area=[43307083044792,22981349108360,9445355470882,29474205380853,17469766849152,7906990341944,130584750195983]
selPoints=basis_regions
nrows=len(selPoints.keys())+1

#==================================================================================
# path of the data directory
#==================================================================================

filesep='/'
exDrive='/media/skoirala/exStore/'
mainDir=exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/'#RS-METEO/EnergyFluxes/'


#==================================================================================
# definitions of the data to be plotted with path, labels, and multipliers for unit
#==================================================================================

dats_LE={
'rs':['','LE','RS',1,'#16A085'],\
'rsm':['','LE','RS_METEO',1,'#E67E22'],\
'mte':[exDrive+'FLUXCOM_Eflux/data/obs/MTE/LE.MTE.ERAI.monthly.2001-2010.commonLandSeaMask.'+_regn+'.txt','LE','MTE',1,'blue'],\
'gl':[exDrive+'FLUXCOM_Eflux/data/obs/E_gleam_v3.1a/E.gleam.720.360.monthly.2001-2010.commonLandSeaMask.'+_regn+'.txt','E','GLEAM (v3.1a)',2.45,'red'],\
'lfe':[exDrive+'FLUXCOM_Eflux/data/obs/lfe_hlf/ET_median/ET_median.all.720.360.204.2001-2005.commonLandSeaMask.'+_regn+'.txt','ET_median','LandFlux-EVAL',2.45,'k'],\

}
dats_Rn={
'rs':['','LE','RS',1,'#16A085'],\
'rsm':['','LE','RS_METEO',1,'#E67E22'],\
'crs':[exDrive+'FLUXCOM_Eflux/data/obs/Rn_CERES/Rn.CERES.Ed4A.720.360.monthly.2001-2010.commonLandSeaMask.'+_regn+'.txt','Rn','CERES (Ed4A)',0.0864,'blue'],\
'srb':[exDrive+'FLUXCOM_Eflux/data/obs/Rn_SRB/NETrad.SRB.rel3_1.720.360.monthly.2001-2007.commonLandSeaMask.'+_regn+'.txt','NETrad','SRB (Rel. 3.1)',1,'red'],\
}

mods_Rn='crs srb'.split()
mods_LE='mte gl lfe'.split()

#==================================================================================
# settings of the plot
#==================================================================================

x0=0.06
y0=0.98

wcolo=0.25
hcolo=0.00925
cb_off_x=0.085
cb_off_y=0.02158
ax_fs=8.5
wp=0.44
hp=0.2
xsp=0.05
aspect_data=0.5
ysp=0.02

selPo=list(selPoints.keys())
plt.figure(figsize=(7,8))
prodInd=0
t1='2001-01'
t2='2010-12'
contOrd=[7,1,2,3,4,5,6]
contOrd=[7,4,1,3,6,2,5]


#==================================================================================
# Plotting the figure
#==================================================================================

for prod in ['LE','Rn']: # loop through the variables (columns)
    dats=vars()['dats_'+prod]
    models=vars()['mods_'+prod]
    fileList,coloList=get_file_list(mainDir,prod)
    all_mod_dat=get_data(fileList)
    lppadx=0.091
    lppady=0.12
    lphp=0.10

    lpwp=0.45
    # loop through the continents (rows)
    for _np in range(len(selPo)):
        _ax=plt.axes([x0+prodInd*wp+prodInd*xsp+lppadx,y0-hp-ysp-(_np)*(lphp+ysp)-lppady,lpwp,lphp])
        ptool.ax_clrXY(axfs=ax_fs)
        if prodInd == 0:
            plt.ylim(0,11)
            plt.ylabel(selPoints[str(contOrd[_np])],fontsize=ax_fs*0.95)
        else:
            plt.ylim(-1,16)

        modInd=1
        modList=[]
        # loop through the rs and rsm runs because they need to have the shading around the lines
        for _mod in ['rs','rsm']:
            modInfo=dats[_mod]
            modVar=modInfo[1]
            modName=modInfo[2]
            if _mod == 'rsm':
                labName='RS+METEO'
            else:
                labName=modName
            modCorr=modInfo[3]
            modColo=modInfo[4]
            le_dat=all_mod_dat[modName][str(contOrd[_np])]
            print(_np+1,np.shape(le_dat))
            le_med=np.nanmedian(le_dat,1)
            le_dev=le_dat
            for _cl in range(np.shape(le_dat)[1]):
                le_dev[:,_cl]=abs(le_dat[:,_cl]-le_med)
            le_mad=1.4826*np.median(le_dev,1)
            xdat=np.linspace(2001,2006,len(le_med),endpoint=True)
            vars()['p'+str(modInd)]=_ax.plot(xdat,le_med,'-',color=modColo,lw=0.8,marker=None,label=labName)
            if _mod == 'rs' or _mod == 'rsm':
                plt.fill_between(xdat,le_med-le_mad,le_med+le_mad,color=modColo,alpha=0.3,linewidth=0.)
            modList=np.append(modList,labName)

            modInd=modInd+1
        # loop through all the models to draw the lines
        for _mod in models:
            modInfo=dats[_mod]
            modFile=modInfo[0]
            print(modFile)
            m_dat=np.loadtxt(modFile,skiprows=3,delimiter=',')
            mod_d=m_dat[:60,1+contOrd[_np]]
            modVar=modInfo[1]
            modName=modInfo[2]
            modCorr=modInfo[3]
            modColo=modInfo[4]
            modList=np.append(modList,modName)
            le_m=mod_d
            # print (_mod)
            if _mod == 'mte' or _mod == 'gl' or _mod == 'crs' or _mod == 'lfe':

                le_po=le_m
            else:
                le_po=le_m
            ptool.rotate_labels(which_ax='x',rot=00,axfs=ax_fs)
            plt.xlim(2000.9,2006.1)
            vars()['p'+str(modInd)]=_ax.plot(xdat,le_po*modCorr,'-',color=modColo,lw=0.8,marker=None,label=modName)
            modInd=modInd+1
        if _np != len(selPo)-1:
            ptool.rem_ticklabels(which_ax='x')
        else:
            ptool.rotate_labels(which_ax='x',rot=90,axfs=ax_fs)
        # plot the legend
        if _np == 0:
            plt.title(varibs_info[prod][0]+' ('+varibs_info[prod][1]+')',fontsize=ax_fs*0.95,y=0.97009)
            if prod == 'LE':
                leg=plt.legend([p1[0],p2[0],p3[0],p4[0],p5[0]],modList,loc=(0.0178171,0.738297),ncol=5,fontsize=ax_fs*0.705,edgecolor=None,handlelength=1.1,fancybox=True,frameon=False,columnspacing=0.34297)
            if prod == 'Rn':
                leg=plt.legend([p1[0],p2[0],p3[0],p4[0]],modList,loc=(0.1042571,0.738297),ncol=4,fontsize=ax_fs*0.705,edgecolor=None,handlelength=1.1,frameon=False,fancybox=True,columnspacing=0.34297)
            for line in leg.get_lines():
                line.set_linewidth(1.2)
    prodInd=prodInd+1

#==================================================================================
# Save the figure
#==================================================================================

plt.savefig('Figure_06_r1.png',dpi=450,bbox_inches='tight')
plt.savefig('Figure_06_r1.pdf',dpi=450,bbox_inches='tight')
#plt.show()
