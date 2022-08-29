import numpy as np
import os,sys
import matplotlib.pyplot as plt

# load and import non default functions from plot_py3
sys.path.append(os.path.expanduser('./plot_py3'))
sys.path.append(os.path.expanduser('~/pyLib/plot_py3'))
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
        for root, dirs, files in os.walk(checkDir):
            for file in files:
                if file.startswith(_regn+"."+varName) and file.endswith(".txt") and 'v6' not in file and 'commonLandSeaMask.' not in file:
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
        print('FUCKKKKK',_prd,fileListprd,len(fileListprd))
        datOutD[_prd]={}
        for _conti in range(7):
            datTmp=np.zeros((len(fileListprd)))
            fileInd=0
            for infile in fileListprd:
                dat=np.loadtxt(infile,skiprows=3,delimiter=',')
                mod_dat=dat[:,2+_conti]
                datTmp[fileInd]=np.nanmean(mod_dat)
                fileInd=fileInd+1
            datOutD[_prd][str(_conti+1)]=datTmp
    return(datOutD)

def get_plotData(_dat,_whatto):
    # get the data for what to plot as bars    
    if _whatto == 'mean':
        __odat=np.nanmean(_dat)
    if _whatto == 'median':
        __odat=np.round(np.nanmedian(_dat),2)
    if _whatto == 'mad':
        __odat=np.nanmedian(abs(_dat-np.nanmedian(_dat)))
    if _whatto == 'rstd':
        __odat=np.round(1.4826*np.nanmedian(abs(_dat-np.nanmedian(_dat))),2)
        # __odat=1.4826*np.nanmedian(abs(_dat-np.nanmedian(_dat)))
    if _whatto == 'uncmedian':
        __odat_rstd=np.round(1.4826*np.nanmedian(abs(_dat-np.nanmedian(_dat))),2)
        __odat_med=np.round(np.nanmedian(_dat),2)
        __odat=np.round(__odat_rstd*100./__odat_med,2)
    if _whatto == 'uncmean':
        __odat=1.4826*np.nanmedian(abs(_dat-np.nanmedian(_dat)))*100./np.nanmean(_dat)
    return(__odat)

def get_ff_rstd_imb(f_dat_rn,f_dat_le,f_dat_h):
    # get the energy budget imbalance    
    odat=[]
    for rr in range(len(f_dat_rn)):
        r_d=f_dat_rn[rr]
        for ll in range(len(f_dat_le)):
            l_d=f_dat_le[ll]
            for hh in range(len(f_dat_h)):
                h_d=f_dat_h[hh]
                imba=r_d-l_d-h_d
                odat=np.append(odat,imba)
    rstdd=1.4826*np.nanmedian(abs(odat-np.nanmedian(odat)))
    return(rstdd)


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
basis_regions={'1':'Asia','2':'North America','3':'Europe','4':'Africa','5':'South America','6':'Oceania','7':'Global'}
bas_area=[43307083044792,22981349108360,9445355470882,29474205380853,17469766849152,7906990341944,130584750195983]
selPoints=basis_regions
nrows=len(selPoints.keys())+1
selPo=list(selPoints.keys())

#==================================================================================
# path of the data directory
#==================================================================================

filesep='/'
exDrive='/media/skoirala/exStore/'
mainDir=exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/'#RS-METEO/EnergyFluxes/'


#==================================================================================
# getting the data of LE, H, and Rn
#==================================================================================

for prod in ['LE','Rn','H']:
    fileList,coloList=get_file_list(mainDir,prod)
    vars()['all_mod_dat_'+prod]=get_data(fileList)

#==================================================================================
# settings of the plot
#==================================================================================

ax_fs=9.5

barwidth=1
wpb=0.10
hpb=0.14
x0=0.09
x1=0.5
x2=0.85
y1=0.26
y2=0.83
xsp=0.05
lppadx=0.061
lppady=0.12
lphp=0.2

lpwp=0.45

contOrd=[7,4,1,3,6,2,5]
xdat=[1,2,3,4,6,7,8,9]

coloList=['#ff8f00','#039be5','#e53935','#777777','#ff8f00','#039be5','#e53935','#777777']


#==================================================================================
# Plotting the figure
#==================================================================================

plt.figure(figsize=(9,8))
prodInd=0


#==================================================================================
# Plotting the lower row for the uncertainty
#==================================================================================
y0=-0.01

for _np in range(len(selPo)): # loop over continents 

    # get the data of the ranges of uncertainty for the bars
    plotMet='uncmedian'
    dat_rn_rs=get_plotData(all_mod_dat_Rn['RS'][str(contOrd[_np])],plotMet)
    dat_rn_rsm=get_plotData(all_mod_dat_Rn['RS_METEO'][str(contOrd[_np])],plotMet)
    dat_le_rs=get_plotData(all_mod_dat_LE['RS'][str(contOrd[_np])],plotMet)
    dat_le_rsm=get_plotData(all_mod_dat_LE['RS_METEO'][str(contOrd[_np])],plotMet)
    dat_h_rs=get_plotData(all_mod_dat_H['RS'][str(contOrd[_np])],plotMet)
    dat_h_rsm=get_plotData(all_mod_dat_H['RS_METEO'][str(contOrd[_np])],plotMet)

    # get the data of the means of uncertainty for the bars
    plotMet='median'
    m_dat_rn_rs=get_plotData(all_mod_dat_Rn['RS'][str(contOrd[_np])],plotMet)
    m_dat_rn_rsm=get_plotData(all_mod_dat_Rn['RS_METEO'][str(contOrd[_np])],plotMet)
    m_dat_le_rs=get_plotData(all_mod_dat_LE['RS'][str(contOrd[_np])],plotMet)
    m_dat_le_rsm=get_plotData(all_mod_dat_LE['RS_METEO'][str(contOrd[_np])],plotMet)
    m_dat_h_rs=get_plotData(all_mod_dat_H['RS'][str(contOrd[_np])],plotMet)
    m_dat_h_rsm=get_plotData(all_mod_dat_H['RS_METEO'][str(contOrd[_np])],plotMet)

    # calculate the imbalance
    datImb_rs=(m_dat_rn_rs-m_dat_le_rs-m_dat_h_rs)*100./m_dat_rn_rs
    datImb_rsm=(m_dat_rn_rsm-m_dat_le_rsm-m_dat_h_rsm)*100./m_dat_rn_rsm
    datPlt=np.array([dat_rn_rsm,dat_le_rsm,dat_h_rsm,datImb_rsm,dat_rn_rs,dat_le_rs,dat_h_rs,datImb_rs])

    # define the axes and plot data on it
    _ax=plt.axes([x0+_np*(wpb+xsp),y0,wpb,hpb])
    ptool.ax_clrXY(axfs=ax_fs)
    bc = plt.barh(xdat,datPlt,height=barwidth,color=coloList,edgecolor='white',linewidth=0.,alpha=0.86)#, 63, normed=1, facecolor='green', alpha=0.75)
    plt.xlim(-5,30)
    _ax.set_yticks([2.5,7.5])
    plt.axvline(x=0,lw=0.3,color='k')
    datErr=np.zeros(np.shape(datPlt))

    # get and write the text on the figure
    for x in range(len(xdat)):
        xval=xdat[x]
        yval=abs(datPlt[x])+datErr[x]+0.5
        datColo=coloList[x]
        if datErr[x] != 0.:
            dattxt=str(round(datPlt[x],2))+'$\\pm$'+str(round(datErr[x],2))
        else:
            if abs(datPlt[x]) < 0.001:
                dattxt=str(round(datPlt[x],3))
            else:
                dattxt=str(round(datPlt[x],2))
                print(dattxt,datPlt[x])
        plt.text(datPlt.max()+0.5, xval,dattxt,color=datColo,rotation=0,ha='left',va='center',fontsize=ax_fs*0.8647)
    plt.text(datPlt.max()+0.5, xval,dattxt,color=datColo,rotation=0,ha='left',va='center',fontsize=ax_fs*0.8647)

    # set the y-axis labels
    if _np == 0:
        _ax.set_yticklabels(['RS+\nMETEO', 'RS'],fontsize=0.77*ax_fs,ma='center',va='center',rotation=0)
        h=plt.ylabel('b',weight='bold',fontsize=ax_fs,rotation=0)
        h.set_rotation(0) 
        # plt.ylabel('b)',fontsize=ax_fs) 
    else:
        _ax.set_yticklabels(['', ''],fontsize=0.77*ax_fs,va='center',rotation=90)

    # set the x-axis label or plot the legend
    if _np == 0:

        _ax.legend((bc[0], bc[1],bc[2],bc[3]), ('Rn', 'LE','H','Imb'),loc=(0.08,1.095),ncol=4,frameon=False,fontsize=ax_fs,edgecolor=None,handlelength=0.9,fancybox=True,columnspacing=0.34297)
    if _np == 3:
        plt.xlabel('Uncertainty\n'+'(%)',fontsize=ax_fs) 

#==================================================================================
# Plotting the upper row for the mean fluxes
#==================================================================================

y0=0.21
for _np in range(len(selPo)):

    # get the data of the mean for the bars
    plotMet='median'
    dat_rn_rs=get_plotData(all_mod_dat_Rn['RS'][str(contOrd[_np])],plotMet)
    dat_rn_rsm=get_plotData(all_mod_dat_Rn['RS_METEO'][str(contOrd[_np])],plotMet)
    dat_le_rs=get_plotData(all_mod_dat_LE['RS'][str(contOrd[_np])],plotMet)
    dat_le_rsm=get_plotData(all_mod_dat_LE['RS_METEO'][str(contOrd[_np])],plotMet)
    dat_h_rs=get_plotData(all_mod_dat_H['RS'][str(contOrd[_np])],plotMet)
    dat_h_rsm=get_plotData(all_mod_dat_H['RS_METEO'][str(contOrd[_np])],plotMet)

    datImb_rs=dat_rn_rs-dat_le_rs-dat_h_rs
    datImb_rsm=dat_rn_rsm-dat_le_rsm-dat_h_rsm
    datPlt=np.array([dat_rn_rsm,dat_le_rsm,dat_h_rsm,datImb_rsm,dat_rn_rs,dat_le_rs,dat_h_rs,datImb_rs])

    # get the data of the robust standard deviation for the bars
    plotMet='rstd'
    r_dat_rn_rs=get_plotData(all_mod_dat_Rn['RS'][str(contOrd[_np])],plotMet)
    r_dat_rn_rsm=get_plotData(all_mod_dat_Rn['RS_METEO'][str(contOrd[_np])],plotMet)
    r_dat_le_rs=get_plotData(all_mod_dat_LE['RS'][str(contOrd[_np])],plotMet)
    r_dat_le_rsm=get_plotData(all_mod_dat_LE['RS_METEO'][str(contOrd[_np])],plotMet)
    r_dat_h_rs=get_plotData(all_mod_dat_H['RS'][str(contOrd[_np])],plotMet)
    r_dat_h_rsm=get_plotData(all_mod_dat_H['RS_METEO'][str(contOrd[_np])],plotMet)

    # create the errors array using robust standard deviation
    datErr=np.array([r_dat_rn_rsm,r_dat_le_rsm,r_dat_h_rsm,0,r_dat_rn_rs,r_dat_le_rs,r_dat_h_rs,0])

    # define the axes and plot data on it
    _ax=plt.axes([x0+_np*(wpb+xsp),y0,wpb,hpb])
    ptool.ax_clrXY(axfs=ax_fs)
    bc = plt.barh(xdat,datPlt,height=barwidth,color=coloList,edgecolor='white',linewidth=0.,alpha=0.86,xerr=datErr,error_kw=dict(color=coloList, lw=0.6, capsize=0, capthick=0))#, 63, normed=1, facecolor='green', alpha=0.75)
    connector, cap_lines, (vertical_lines,) = bc.errorbar.lines
    vertical_lines.set_color(coloList)
    plt.title(selPoints[str(contOrd[_np])],fontsize=ax_fs)
    _ax.set_yticks([2.5,7.5])

    # set the y-axis labels
    if _np == 0:
        _ax.set_yticklabels(['RS+\nMETEO', 'RS'],fontsize=0.77*ax_fs,ma='center',va='center',rotation=0)
        h=plt.ylabel('a',weight='bold',fontsize=ax_fs,rotation=0)
        h.set_rotation(0) 
    else:
        _ax.set_yticklabels(['', ''],fontsize=0.77*ax_fs,va='center',rotation=90)   
    plt.axvline(x=0,lw=0.3,color='k')
    plt.xlim(-1,17)

    # get and write the text on the figure
    for x in range(len(xdat)):
        xval=xdat[x]
        yval=abs(datPlt[x])+datErr[x]+0.5
        datColo=coloList[x]
        if datErr[x] != 0.:
            dattxt=str(round(datPlt[x],2))+'$\\pm$'+str(round(datErr[x],2))
        else:

            if abs(datPlt[x]) < 0.001:
                dattxt=str(round(datPlt[x],3))
            else:
                dattxt=str(round(datPlt[x],2))
                print(dattxt,datPlt[x])
        plt.text(datPlt.max()+0.5, xval,dattxt,color=datColo,rotation=0,ha='left',va='center',fontsize=ax_fs*0.8647)
    if _np == 3:
        plt.xlabel('Mean Energy Fluxes\n'+'('+varibs_info['LE'][1]+')',fontsize=ax_fs) 

#==================================================================================
# Save the figure
#==================================================================================

plt.savefig('Figure_07_r1.png',dpi=450,bbox_inches='tight')
plt.savefig('Figure_07_r1.pdf',dpi=450,bbox_inches='tight')
