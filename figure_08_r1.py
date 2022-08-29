import numpy as np
import os,sys
import matplotlib.pyplot as plt
from matplotlib import colors

# load and import non default functions from plot_py3
sys.path.append(os.path.expanduser('~/pyLib/plot_py3'))
sys.path.append(os.path.expanduser('./plot_py3'))
import plotTools as ptool
import mapTools as mtool

#==================================================================================
# define the functions 
#==================================================================================

class BoundaryNorm(colors.Normalize):
    # normalize the boundaries of the map
    def __init__(self, boundaries):
        self.vmin = boundaries[0]
        self.vmax = boundaries[-1]
        self.boundaries = boundaries
        self.N = len(self.boundaries)

    def __call__(self, x, clip=False):
        x = np.asarray(x)
        ret = np.zeros(x.shape, dtype=np.int)
        for i, b in enumerate(self.boundaries):
            ret[np.greater_equal(x, b)] = i
        ret[np.less(x, self.vmin)] = -1
        ret = np.ma.asarray(ret / float(self.N-1))
        return ret

def get_file_list(mainDir):
    # get the list of all files of all members of RS and RS+METEO runs    
    fileList=[]
    coloList=[]
    for prd in ['RS','RS_METEO']:
        checkDir=mainDir+filesep+prd+filesep+'members'+filesep+'common'+filesep+_regn+filesep
        for root, dirs, files in os.walk(checkDir):
            for file in files:
                if file.startswith(_regn+".LE") and file.endswith(".txt") and 'v6' not in file  and 'commonLandSeaMask.' not in file:
                    infile=os.path.join(root, file)
                    fileList=np.append(fileList,infile)
                    if 'RS_METEO' in infile:
                        coloList=np.append(coloList,rsmcolo)
                    else:
                        coloList=np.append(coloList,rscolo)
        print('File List',prd,fileList,len(fileList))
    return(fileList,coloList)

def get_data(fileList):
    # get the data from the list of all files of all members of RS and RS+METEO runs    
    datOut=np.zeros((len(fileList),len(basis_regions)))
    Ind=0
    for infile in fileList:
        dat=np.loadtxt(infile,skiprows=3,delimiter=',')
        mod_dat=dat[:,2:]
        datOut[Ind,:]=mod_dat.mean(0)
        Ind=Ind+1
    return(datOut)

def plot_bar(regnID):
    # plot the box and whisker chart
    datInd=regnID-1
    ptool.ax_clrXY(axfs=ax_fs*1.15)
    datBas=datAll[:,datInd]*bas_area[datInd]/2.45*1e-15*365.5
    bplot = plt.boxplot([datBas[coloList==rscolo],datBas[coloList==rsmcolo]],sym='',widths=0.3054,patch_artist=True,notch=True)#,alpha=0.85,linewidth=0.,edgecolor=None)#, 63, normed=1, facecolor='green', alpha=0.75)
    colors = [rscolo, rsmcolo]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_linewidth(0.)
    for patch, color in zip(bplot['medians'], colors):
        patch.set_color(color)
        patch.set_linewidth(1.5)
        # patch.set_line('green')
    whiscolors = [rscolo, rscolo,rsmcolo,rsmcolo]
    for line, color in zip(bplot['whiskers'], whiscolors):
        line.set_color(color)
        line.set_linewidth(0.9)
    for line, color in zip(bplot['fliers'], whiscolors):
        line.set_color(color)
        line.set_linewidth(0.9)
    for line, color in zip(bplot['caps'], whiscolors):
        line.set_color(color)
        line.set_linewidth(0.9)


    if basis_regions[str(regnID)] == 'Global':
        plt.ylabel(varibs_info['LE'][0]+'\n('+varibs_info['LE'][1]+' |\n$mm.yr^{-1}$'+')',x=-0.185,fontsize=ax_fs*1.25)
        t_x=plt.figtext(-2.5,0.5,'b',fontsize=ax_fs*1.85,weight='bold',transform=plt.gca().transAxes)
    plt.title(basis_regions[str(regnID)],fontsize=ax_fs*1.27)
    clab=plt.gca().get_yticks()[:-1]
    plt.gca().set_xticklabels(['RS\n(27)','RS+\nMETEO\n(36)'],fontsize=ax_fs)
    # plt.gca().text(133.5,-24,basis_regions['6t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))

    textw=[]
    for _clab in clab:
        print (_clab)
        area_r=bas_area[datInd]
        text_mm=_clab*1e15/(area_r)
        textw=np.append(textw,str(int(_clab))+' | '+str(int(text_mm)))
    plt.yticks(clab,textw,fontsize=ax_fs*1.219)
    return


#==================================================================================
# variables and the definition of ranges and colorbar
#==================================================================================

cbef='nipy_spectral_r'
varibs_info={
            'LE':['Evapotranspiration','$\\mathrm{10^3\ km^3.yr^{-1}}$','ef',cbef],\
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
bas_area=[38807195300896,20645828565200,9380828200832,21518730902059,17281271912192,7901544260408,115535399141587]


#==================================================================================
# path of the data directory
#==================================================================================

filesep='/'
exDrive='/media/skoirala/exStore/'
mainDir=exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/'

#==================================================================================
# get all the data of individual members and ensembles
#==================================================================================

fileList,coloList=get_file_list(mainDir)
datAll=get_data(fileList)

ens_rsm=np.loadtxt(exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/RS_METEO/ensembles/common/'+_regn+filesep+_regn+'.LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-ALL.720_360.monthly.2001-2013.txt',skiprows=3,delimiter=',')[:,2:].mean(0)
ens_rsm_rn=np.loadtxt(exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/RS_METEO/ensembles/common/'+_regn+filesep+_regn+'.Rn.RS_METEO.EBC-NONE.MLM-ALL.METEO-ALL.720_360.monthly.2001-2013.txt',skiprows=3,delimiter=',')[:,2:].mean(0)
ens_rs=np.loadtxt(exDrive+'FLUXCOM_Eflux/analysis_eFlux_paper_iter4_201903/data.local/tEnergyFluxes/RS/ensembles/common/'+_regn+filesep+_regn+'.LE.RS.EBC-ALL.MLM-ALL.METEO-NONE.720_360.monthly.2001-2013.txt',skiprows=3,delimiter=',')[:,2:].mean(0)


#==================================================================================
# read the data for the map of the continents
#==================================================================================

fill_val=0.
maskDir=exDrive+'FLUXCOM_Eflux/data/masks/'
conti=np.fromfile(maskDir+filesep+'continent_halfdegree_720_360.byte',np.int8).reshape(360,720)
conti=conti+1
conti=np.ma.masked_equal(conti,7).filled(6)
conti=np.ma.masked_greater(conti,6).filled(0)
mte_mask_dat=np.fromfile(maskDir+filesep+'mask_rsm_all.hlf',np.float32).reshape(360,720)
mo_fill_mask=np.ma.getmask(np.ma.masked_equal(mte_mask_dat,0))
conti[mo_fill_mask]=0



#==================================================================================
# settings of the plot
#==================================================================================
x0=0.06
y0=0.98
ax_fs=6
wp=0.95
hp=1
valrange=[1,7]
coloListp=["white","#f7dc6f", '#82e0aa','#ccd1d1','#7fb3d5',"#c39bd3",'#ec7063']
cm2 = colors.ListedColormap(coloListp) 
bounds2 = np.array([-0.5,0.5, 1.5,2.5,3.5,4.5,5.5,6.5])


#==================================================================================
# Plotting the figure
#==================================================================================

plt.figure(figsize=(9,6),facecolor='#bbbbbb')

#==================================================================================
# Plotting the continental map
#==================================================================================

_ax=plt.axes([x0,y0-hp,wp,hp])
_mp=mtool.def_map_cyl(latmin=-60,lonmin=-180,lonmax=180,line_w=0.2,labmer=[False,False,True,False],labpar=[True,True,False,False],labfs=ax_fs,remLab=True,min_coast=True)
_mp.imshow(np.ma.masked_equal(conti[0:300,:],fill_val), cmap=cm2, norm=BoundaryNorm(bounds2),interpolation='none',origin='upper')
plt.title('a',fontsize=ax_fs*1.85,weight='bold',x=-0.065,y=0.5)
# Getting the data of the continental scale fluxes

relContribRS=np.zeros((7))
relContribRSM=np.zeros((7))
for con in range(1,8):
    relContribRS[con-1]=(1.*ens_rs[con-1]*bas_area[con-1])
    relContribRSM[con-1]=(1.*ens_rsm[con-1]*bas_area[con-1])
    contrib=(1.*ens_rs[con-1]*bas_area[con-1])/(1.*ens_rs[-1]*bas_area[-1])*100.
    datInd=con-1
    datBas=datAll[:,datInd]*bas_area[datInd]/2.45*1e-15*365.5
    print(datBas)
    dat_rsm=datBas[coloList==rsmcolo]
    dat_rs=datBas[coloList==rscolo]
    levol=(1.*ens_rs[con-1]*bas_area[con-1])/2.45*1e-15*365.5
    contribm=(1.*ens_rsm[con-1]*bas_area[con-1])/(1.*ens_rsm[-1]*bas_area[-1])*100.
    levolm=(1.*ens_rsm[con-1]*bas_area[con-1])/2.45*1e-15*365.5
    rsstd=1.4826*np.nanmedian(abs(dat_rs-np.nanmedian(dat_rs)))
    rsmstd=1.4826*np.nanmedian(abs(dat_rsm-np.nanmedian(dat_rsm)))
    bastext=basis_regions[str(con)]+'\n'+'RS: '+str(round(levol,1))+'$\\pm$'+str(round(rsstd,1))+'\nRS+M: '+str(round(levolm,1))+'$\\pm$'+str(round(rsmstd,1))+''
    basis_regions[str(con)+'t']=bastext

# Plotting (writing) the text of the continental scale fluxes over the map

plt.gca().text(105,59,basis_regions['1t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))
plt.gca().text(-100,45,basis_regions['2t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))
plt.gca().text(40,58,basis_regions['3t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))
plt.gca().text(27,9,basis_regions['4t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))
plt.gca().text(-59,-8,basis_regions['5t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))
plt.gca().text(133.5,-24,basis_regions['6t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))
plt.gca().text(0,-50,basis_regions['7t'],color='k',ha='center',va='center',fontsize=ax_fs*1.239,bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0.))


#==================================================================================
# Plotting the pie/donut chart
#==================================================================================

# settings of the pie chart
hpie=0.13
wpie=0.195
xpie=0.07

# defining axes and plotting for RS
prs=plt.axes([xpie+0.02,0.37,hpie,wpie])
prs.pie(relContribRS[:-1], autopct='%1.1f%%',colors=coloListp[1:], startangle=90,textprops={'fontsize':ax_fs*1.1692},wedgeprops={'edgecolor':'white','linewidth':0.9,'width':0.5},pctdistance=1.3)
plt.title('RS',fontsize=ax_fs*1.25,y=0.435,color=rscolo,va='center',bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0))


# defining axes and plotting for RS
prsm=plt.axes([xpie+0.13,0.18,hpie,wpie])
prsm.pie(relContribRSM[:-1], autopct='%1.1f%%',colors=coloListp[1:], startangle=90,textprops={'fontsize':ax_fs*1.1692},wedgeprops={'edgecolor':'white','linewidth':0.9,'width':0.5},pctdistance=1.3)
plt.title('RS+\nMETEO',fontsize=ax_fs*1.25,y=0.45,color=rsmcolo,va='center',bbox=dict(boxstyle="round",linewidth=0,fc='w',alpha=0))


#==================================================================================
# Plotting the box plots
#==================================================================================

# settings of the box plots

barwidth=0.5
wpb=0.0558
hpb=0.15
x0=0.13
x1=0.5
x2=0.85
y0=-0.03
y1=0.26
y2=0.83
xsp=0.083

regns=[7,1,2,3,4,5,6]
regns=[7,4,1,3,6,2,5]

# plotting of the box plots

for rr in range(len(regns)): # looping over continents
    axins = plt.axes([x0+rr*(wpb+xsp),y0,wpb,hpb])
    plot_bar(regns[rr])

#==================================================================================
# Save the figure
#==================================================================================
plt.savefig('Figure_08_r1.png',dpi=450,bbox_inches='tight')
plt.savefig('Figure_08_r1.pdf',dpi=450,bbox_inches='tight')
# plt.show()
