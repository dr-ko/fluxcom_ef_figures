#!/usr/local/bin/python
import numpy as np
import scipy.stats as scst
def calc_pearson_r(_dat1,_dat2,_fill_val=-9999.):
    _mask_1=get_mask_01(_dat1,_fill_val)
    _mask_2=get_mask_01(_dat2,_fill_val)
    _com_mask=_mask_1*_mask_2
    _data_mask=np.ma.getmask(np.ma.masked_equal(_com_mask,1))
    __dat1=_dat1[_data_mask].flatten()
    __dat2=_dat2[_data_mask].flatten()
    _r,_p=scst.pearsonr(np.ma.masked_equal(__dat1,_fill_val),np.ma.masked_equal(__dat2,_fill_val))
    print (_r,_p)
    return(_r,_p)

def calc_spearman_r(_dat1,_dat2,_fill_val=-9999.):
    _mask_1=get_mask_01(_dat1,_fill_val)
    _mask_2=get_mask_01(_dat2,_fill_val)
    _com_mask=_mask_1*_mask_2
    _data_mask=np.ma.getmask(np.ma.masked_equal(_com_mask,1))
    __dat1=_dat1[_data_mask].flatten()
    __dat2=_dat2[_data_mask].flatten()
    _r,_p=scst.spearmanr(np.ma.masked_equal(__dat1,_fill_val),np.ma.masked_equal(__dat2,_fill_val))
    print (_r,_p)
    return(_r,_p)

def calc_globmean(_dat,_fill_val=-9999.):
    gl_mean=np.nanmean(np.ma.masked_equal(clean_nan(_dat),_fill_val))
    return(gl_mean)

def calc_mean_relative_bias(_dat1,_dat2,_fill_val=-9999.):
    _mask=np.ma.getmask(np.ma.masked_equal(clean_nan(_dat1),_fill_val))
    m_r_bias_gr=clean_nan(_dat1)/clean_nan(_dat2)
    m_r_bias_gr[_mask]=_fill_val
    m_r_bias_gr=clean_nan(m_r_bias_gr)
    m_r_bias=calc_globmean(m_r_bias_gr,_fill_val)
    return(m_r_bias)
def get_mask_01(_dat,_fill_val=-9999.):
    m_mask = np.ma.masked_equal(np.ma.masked_not_equal(_dat,_fill_val).filled(1),_fill_val).filled(0)
    return(m_mask)
def calc_global_relative_bias(_dat1,_dat2,_fill_val=-9999.):
    gl_mean_1=calc_globmean(_dat1,_fill_val)
    gl_mean_2=calc_globmean(_dat2,_fill_val)
    g_r_bias=gl_mean_1/gl_mean_2
    return(g_r_bias)

def clean_nan(_tmp,_fill_val=-9999.):
    whereisNan=np.isnan(_tmp)
    _tmp[whereisNan]=_fill_val
    whereisInf=np.isinf(_tmp)
    _tmp[whereisInf]=_fill_val
    return(_tmp)
