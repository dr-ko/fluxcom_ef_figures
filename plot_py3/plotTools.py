#!/usr/local/bin/python
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib
from matplotlib import *
import numpy as np
def rem_axLine(rem_list=['top','right'],axlw=0.4):
    ax=plt.gca()
    for loc, spine in ax.spines.items():
        if loc in rem_list:
            spine.set_position(('outward',0)) # outward by 10 points
            spine.set_linewidth(0.)
        else:
            spine.set_linewidth(axlw)
    return

def set_ax_font(axfs=11):
    locs,labels = yticks()
    setp(labels,fontsize=axfs)
    locs,labels = xticks()
    setp(labels,fontsize=axfs)
    labels = ax.get_xticklabels()
    return
def rotate_labels(which_ax='both',rot=0,axfs=6):
    ax=plt.gca()
    if which_ax == 'x' or which_ax=='both':
        locs,labels = plt.xticks()
        plt.setp(labels,rotation=rot,fontsize=axfs)
    if which_ax == 'y' or which_ax=='both':
        locs,labels = plt.yticks()
        plt.setp(labels,rotation=rot,fontsize=axfs)
    return

def rem_ticks(which_ax='both'):
    ax=plt.gca()
    if which_ax == 'x' or which_ax=='both':
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position("none")
    if which_ax == 'y' or which_ax=='both':
        ax.set_yticklabels([])
        ax.yaxis.set_ticks_position("none")
    return
def rem_ticklabels(which_ax='both'):
    ax=plt.gca()
    if which_ax == 'x' or which_ax=='both':
        ax.set_xticklabels([])
    if which_ax == 'y' or which_ax=='both':
        ax.set_yticklabels([])
    return
def put_ticks(nticks=5,which_ax='both',axlw=0.3):
    ticksfmt=plt.FormatStrFormatter('%.1f')
    ax=plt.gca()
    if which_ax == 'x':
        ax.xaxis.set_ticks_position('bottom')
        lines = ax.get_xticklines()
        labels = ax.get_xticklabels()
        for line in lines:
            line.set_marker(matplotlib.lines.TICKDOWN)
        for label in labels:
            label.set_y(-0.02)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))
    #........................................
    if which_ax == 'y':
        ax.yaxis.set_ticks_position('left')
        lines = ax.get_yticklines()
        labels = ax.get_yticklabels()
        for line in lines:
            line.set_marker(matplotlib.lines.TICKLEFT)
            line.set_linewidth(axlw)
        '''
        for label in labels:
            label.set_x(-0.02)
        '''
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))
    if which_ax=='both':
        ax.yaxis.set_ticks_position('left')
        lines = ax.get_yticklines()
        labels = ax.get_yticklabels()
        for line in lines:
            line.set_marker(matplotlib.lines.TICKLEFT)
            line.set_linewidth(axlw)
        '''
        for label in labels:
            label.set_x(-0.02)
        '''
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))

        ax.xaxis.set_ticks_position('bottom')
        lines = ax.get_xticklines()
        labels = ax.get_xticklabels()
        for line in lines:
            line.set_marker(matplotlib.lines.TICKDOWN)
        '''
        for label in labels:
            label.set_y(-0.02)
        '''
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nticks,min_n_ticks=nticks))

def clean_nan(tmp):
    whereisNan=numpy.isnan(tmp)
    tmp[whereisNan]=-9999.
    whereisInf=numpy.isinf(tmp)
    tmp[whereisInf]=-9999.
    return(tmp)

def ax_clr(axfs=7):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right','left','bottom'])
    rem_ticks(which_ax='both')
def ax_clrX(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right','bottom'])
    rem_ticks(which_ax='x')
    ax.tick_params(axis='y', labelsize=axfs)
    put_ticks(which_ax='y',axlw=0.3,nticks=nticks)
def ax_clrY(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right','left'])
    rem_ticks(which_ax='y')
    put_ticks(which_ax='x',axlw=0.3,nticks=nticks)
    ax.tick_params(axis='x', labelsize=axfs)

def ax_clrXY(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right'])
#    rem_ticks(which_ax='y')
    put_ticks(which_ax='y',axlw=0.3,nticks=nticks)
    ax.tick_params(axis='both', labelsize=axfs)

def ax_orig(axfs=7,axlw=0.3,nticks=3):
    ax=plt.gca()
    rem_axLine(rem_list=['top','right'],axlw=axlw)
    put_ticks(which_ax='both',axlw=0.3,nticks=nticks)
    ax.tick_params(axis='both',labelsize=axfs)
