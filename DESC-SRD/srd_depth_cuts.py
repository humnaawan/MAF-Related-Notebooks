########################################################################################################################
# The goal here is to find the footprint for extragalactic science, for both 1yr and 10yrs. This is achieved by looking
# the coadded depth in each of the six bands, after accounting for MW dust extinction and restricting attention to only
# those areas where there's data for all-6 bands and where the depth is > 0 for all the bands. Then, within this
# all-band footprint, we look at the median 5sigma point source depth, the std in this depth and the sky area coverage
# after i-band depth cuts (i.e., looking at regions where i_depth > mag_cut).
#
# This script prints out stats (in a Markdown table format) for various cases, e.g. no cut, all-band, depth cuts.
# 
# Also, although we dont want an EBV cut, we want our depth cut to be such that an EBV cut (say discarding regions with
# EBV>0.2-0.3) does not throw away a significant portion of the area. To ensure this, this script plots out the EBV
# histogram for each of the cuts.
# 
# Also, to ensure that our depth cuts are effectively discarding the high extinction pixels, we plot out the histogram
# of the galactic latitude for each of the depth cuts too.
# 
# For 1yr data, the median depths are renormalized to match the depth achieved with 10% WFD visits since 1yrin minion1016
# doesn't contain 10% of total visits. This issue looked at in "minion1016 - 1yr, 10yr depth visit comparison.ipynb".
#
# The coadded depth data was saved from a previous run; implemented RepulsiveRandomFieldPerVisit dithers on minion1016.
# Here we look at the data without border masking. The EBV map is loaded from MAF; it isn't using the dithered poinints
# but the effect on the histograms should be minimal; adding the right dithers would require setting up mafContrib which
# could be painful with JupyterLab.
# 
## Based on the EBV limit we'd like and the depth variation, we finalize the i-band depth limit for 1yr to be i=24.5
## and i=26.0 for 10yrs. For these, we plot the galactic latitude and EBV histograms, alonside skymaps that show the 
## survey footprint in each of the six bands.
#
# Initial code was built in v0_DESC-ScienceReq_1,10yrOptimization-renormalized1yr.ipynb Switced to .py script for ease.
#
# Humna Awan; humna.awan@rutgers.edu
#
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import os
import healpy as hp
import copy
from collections import OrderedDict
import lsst.sims.maf.metricBundles as metricBundles   # need MAF installed; for reading in coadd data
# for galactic latitude histogram
from astropy.coordinates import SkyCoord
from astropy import units as u
# for EBV map from MAF
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.maps as maps

########################################################################################################################
# set some things up
data_dir = '/global/homes/a/awan/LSST/output'   # where coadd depth maps are
dbpath = '/global/cscratch1/sd/awan/dbs_old_unzipped/minion_1016_sqlite_new_dithers.db'
latLonDeg = False # know that ra, dec in minion1016 are in radians; need for EBV map read-in from MAF

orderBand = ['u', 'g', 'r', 'i', 'z', 'y']
nside = 256
mag_cuts = [24.0, 24.5, 24.7, 25.0, 25.3, 25.5, 26.0, 26.5]   # depth cuts to look after

print('data_dir: %s'%data_dir)
print('nside: %s\n'%nside)
    
########################################################################################################################
# read in the data: with dust extinction: 1yr, 10yr
dir_paths = {}
dir_paths['1yr'] = '%s/Depth_Data_1yr_minion1016/'%data_dir
dir_paths['1yr'] += 'coaddM5Analysis_nside256_withDustExtinction_minion1016_1yrCut_directory/'
dir_paths['10yr'] = '%s/Depth_Data_10yr_minion1016/'%data_dir
dir_paths['10yr'] += '/coaddM5Analysis_nside256_withDustExtinction_minion1016_fullSurveyPeriod_directory/'

# filenames for npz files without border masking
filenames= {}
filenames['1yr'] = [f for f in os.listdir(dir_paths['1yr']) if any([f.endswith('.npz') and not f.startswith('._')])]
filenames['10yr'] = [f for f in os.listdir(dir_paths['10yr']) if any([f.endswith('.npz') and \
                                                                      not f.startswith('._') and \
                                                                      not f.__contains__('_masked_')])]
print('filenames:\n%s\n'%(filenames, ))
    
# get the files and put the data in a bundle
print('## Reading in the data ... \n')
data_bundle = OrderedDict()
for yr_cut in filenames.keys():
    for ith in range(len(filenames[yr_cut])): #[::-1]:
        filename= [f for f in filenames[yr_cut] if f.__contains__('_%s_'%orderBand[ith])][0]
        
        band = filename.split('_Rep')[0].split('unmasked_')[1] # get the band from the filename
        mB = metricBundles.createEmptyMetricBundle()
        mB.read('%s/%s'%(dir_paths[yr_cut], filename))
        data_bundle['%s_%s'%(yr_cut, band)]= mB
        
########################################################################################################################
# check the improvement factor between 1, 10yr data
print('\n## Calculating improvement in fluxes between 1yr and 10yrs ... ')
allImprovs = []
for band in orderBand:
    yr_cut = '1yr_%s'%band
    in_survey_positive = np.where((data_bundle[yr_cut].metricValues.mask == False) & \
                                 (data_bundle[yr_cut].metricValues.data > 0))[0]
    one_yr_med = np.median(data_bundle[yr_cut].metricValues.data[in_survey_positive])
    
    yr_cut = '10yr_%s'%band
    in_survey_positive = np.where((data_bundle[yr_cut].metricValues.mask == False) & \
                                 (data_bundle[yr_cut].metricValues.data > 0))[0]
    ten_yr_med = np.median(data_bundle[yr_cut].metricValues.data[in_survey_positive])
    
    one_yr_flux = 10**(-one_yr_med/2.5)
    ten_yr_flux = 10**(-ten_yr_med/2.5)
    print('%s-band: improvement factor in flux: %s'%(band, one_yr_flux/ten_yr_flux))
    allImprovs.append(one_yr_flux/ten_yr_flux)
                                
print('Wanted improvement factor over ten years: %s'%np.sqrt(10.))
print('Mean improvement factor across ugrizy: %s'%np.mean(allImprovs))

# The improvement from 1-10yr is too good -- 1yr in minion1016 strongly prefers DDFs so 1yr defined by the number
# of the nights doesnt have 10% of total visits (has <7%). For now, renormalize the 1yr depth s.t. median matches
# that after 10% of 10-year WFD observations.
# Based on "coaddM5 analysis - minion1016 1yr with 10% WFD visits.ipynb"
wanted1yr_medianDepth= {'g': 25.377165833786055, 'i': 24.910057884620223, 'r': 25.565945074516804,
                        'u': 23.795160853950424, 'y': 23.315667199085482, 'z': 24.002597276614527}

print('\n## Renormalizing 1yr depth data ... ')
inSurveyIndex = {}
for key in data_bundle.keys():
    inSurveyIndex[key] = np.where(data_bundle[key].metricValues.mask == False)[0]
    if key.__contains__('1yr'):
        band = key.split('1yr_')[1]
        print(band)
        medDepth = np.median(data_bundle[key].metricValues.data[inSurveyIndex[key]])
        print('Median depth as read: %s'%np.median(data_bundle[key].metricValues.data[inSurveyIndex[key]]))
        delm = wanted1yr_medianDepth[band]-medDepth
        print('m_wanted-m_current = %s'%delm)
        data_bundle[key].metricValues.data[:] += delm
        print('Renormalized map. \nNew median: %s\n'%np.median(data_bundle[key].metricValues.data[inSurveyIndex[key]]))

# re-check the improvement factor between 1, 10yr data
print('## Re-calculating improvement in fluxes between 1yr and 10yrs after the renormalizing ... ')
allImprovs = []
for band in orderBand:
    yr_cut = '1yr_%s'%band
    in_survey_positive = np.where((data_bundle[yr_cut].metricValues.mask == False) & \
                                  (data_bundle[yr_cut].metricValues.data > 0))[0]
    one_yr_med = np.median(data_bundle[yr_cut].metricValues.data[in_survey_positive])
    
    yr_cut = '10yr_%s'%band
    in_survey_positive = np.where((data_bundle[yr_cut].metricValues.mask == False) & \
                                  (data_bundle[yr_cut].metricValues.data > 0))[0]
    ten_yr_med = np.median(data_bundle[yr_cut].metricValues.data[in_survey_positive])
    
    one_yr_flux = 10**(-one_yr_med/2.5)
    ten_yr_flux = 10**(-ten_yr_med/2.5)
    print('%s-band: improvement factor in flux: %s'%(band, one_yr_flux/ten_yr_flux))
    allImprovs.append(one_yr_flux/ten_yr_flux)
                                
print('\nWanted improvement factor over ten years: %s'%np.sqrt(10.))
print('Mean improvement factor across ugrizy: %s'%np.mean(allImprovs))

########################################################################################################################
########################################################################################################################
# calculate some stats
areaPerPixel= hp.pixelfunc.nside2pixarea(nside=nside, degrees=True)

def calc_stats(bundle, index, allBandInds=False, return_stuff=False):
    # index must have the same keys as bundle
    if (bundle.keys()!=index.keys()) and not allBandInds:
        raise ValueError('index must have the same keys as bundle:\n%s\n%s'%(bundle.keys(), index.keys()))
        
    if return_stuff and allBandInds: 
        stuff_to_return = {}
        for key in ['5$\sigma$ Depth: Median', '5$\sigma$ Depth: Std', 'Area (deg2)']:
            stuff_to_return[key] = {}
        
    header, sep, med_depth, std_depth, area = '| ', '| ---- ', '| 5$\sigma$ Depth: Median ', '| 5$\sigma$ Depth: Std ', '| Area (deg2) '
    yr = None
    for key in bundle:
        if yr is None: yr = key.split('yr')[0]+'yr'
        
        current_yr = key.split('yr')[0]+'yr'
        if current_yr!=yr:
            print('%s\n%s\n%s\n%s\n%s\n'%(header, sep, med_depth, std_depth, area))
            header, sep, med_depth, std_depth, area = '| ', '| ---- ', '| 5$\sigma$ Depth: Median ', '| 5$\sigma$ Depth: Std ', '| Area (deg2) '
            yr = current_yr
        
        if allBandInds: index_key = current_yr
        else: index_key = key
            
        med = np.nanmedian(bundle[key].metricValues.data[index[index_key]])
        std = np.nanstd(bundle[key].metricValues.data[index[index_key]])
        sarea = (len(index[index_key])*areaPerPixel)
        
        if return_stuff and allBandInds:
            stuff_to_return['5$\sigma$ Depth: Median'][key] = med
            stuff_to_return['5$\sigma$ Depth: Std'][key] = std
            stuff_to_return['Area (deg2)'][index_key] = sarea
            
        header += '| %s '%key
        sep += '| ---- '
        med_depth += '| %.2f '%med
        std_depth += '| %.2f '%std
        area += '| %.2f '%sarea
        
    print('%s\n%s\n%s\n%s\n%s\n'%(header, sep, med_depth, std_depth, area))
    
    if return_stuff: return stuff_to_return
########################################################################################################################
########################################################################################################################
# Calculate stats in the survey region (unmasked; no constraints on depth, i.e., even have negative depths rn).
print('\n#### Stats: no constraints on depth, i.e., even have negative depths')
calc_stats(bundle=data_bundle, index=inSurveyIndex, allBandInds=False)

########################################################################################################################
# Find the area common to all-6 bands with depths>0 in all.
allBandPixels = {}  # dictionary for pixels that are common in all six bands with depth>0.

for key in data_bundle:
    index = np.where((data_bundle[key].metricValues.mask == False) & \
                     (data_bundle[key].metricValues.data > 0))[0]
    # save the indices
    yrTag = key.split('yr')[0]+'yr'
    if yrTag not in allBandPixels.keys():
        allBandPixels[yrTag]= index  
    else:
        allBandPixels[yrTag]= list(set(allBandPixels[yrTag]).intersection(index))

########################################################################################################################
# Calculate the stats for the all-band footprint
print('\n#### Stats: considering area common to all-6 bands with depths>0 in all')
calc_stats(bundle=data_bundle, index=allBandPixels, allBandInds=True)

########################################################################################################################
# implement depth cuts in i-band and save the pixel numbers
iCutPixels = {}
# run over different cuts
for mag_cut in mag_cuts:
    if mag_cut not in iCutPixels.keys():
        iCutPixels[mag_cut] = {}
        
    for yrTag in allBandPixels:
        if yrTag not in iCutPixels[mag_cut].keys():
            iCutPixels[mag_cut][yrTag] = {}
        
        # find the pixels satisfying the iBand cut.
        iBandCutInd = np.where((data_bundle['%s_i'%yrTag].metricValues.data[allBandPixels[yrTag]]>=mag_cut))[0]
        iCutPixels[mag_cut][yrTag] = np.array(allBandPixels[yrTag])[iBandCutInd] # store

########################################################################################################################
# Calculate the stats in the survey region (unmasked; no constraints on depth, i.e., even have negative depths rn).
dat_keys = ['Area (deg2)', '5$\sigma$ Depth: Median', '5$\sigma$ Depth: Std']

########################################################################################################################
# plots for area and depth variations as a funtion of mag cuts
# need to create a list of all the stats for cleaner plots.
stats_allmags = {}
for mag_cut in mag_cuts:
    print('\n#### Stats: i>%s in area common to all six bands with depths>0 in all'%mag_cut)
    stats = calc_stats(bundle=data_bundle, index=iCutPixels[mag_cut],
                       allBandInds=True, return_stuff=True) # area: 1yr, 10yr; depth stuff: 1yr_band, 10yr_band
    for dat_key in dat_keys:
        if dat_key not in stats_allmags: stats_allmags[dat_key] = {}
        for key in stats[dat_key].keys():
            if key.__contains__('_'):  # need to separate 1yr, 10yr from 1yr_<band>, 10yr_<band>
                sp = key.split('_')
                yr, band = sp[0], sp[1]
            else:
                yr, band = key, None
                
            if band is None: # all-band keys: 1yr, 10yr
                if yr not in stats_allmags[dat_key]:
                    stats_allmags[dat_key][yr] = []
            else: # need to account for the band for each of 1yr, 10yr
                if yr not in stats_allmags[dat_key]:
                    stats_allmags[dat_key][yr] = {}
                if band not in  stats_allmags[dat_key][yr]:
                     stats_allmags[dat_key][yr][band] = []
            
            if band is not None:
                stats_allmags[dat_key][yr][band].append(stats[dat_key][key])
            else:
                stats_allmags[dat_key][yr].append(stats[dat_key][yr])

print('\n## Plotting area and depth variations for mag_cuts: %s ...'%mag_cuts)
colors = ['m', 'g', 'b', 'r', 'k', 'c']
fontsize = 14
# plot
plt.clf()
fig, axes = plt.subplots(2,3)
fig.subplots_adjust(wspace=.2, hspace=.3)

for i, dat_key in enumerate(dat_keys):
    if dat_key.__contains__('Depth'):  # band-specific statistic
        for j, band in enumerate(stats_allmags[dat_key]['1yr'].keys()):
            # plot 1yr data for this statistic
            axes[0, i].plot(mag_cuts, stats_allmags[dat_key]['1yr'][band], 'o-',
                            color=colors[j], label='%s-band'%band)
            # plot 10yr data for this statistic
            axes[1, i].plot(mag_cuts, stats_allmags[dat_key]['10yr'][band], 'o-',
                            color=colors[j], label='%s-band'%band)
    else:
        # plot 1yr area
        axes[0, i].plot(mag_cuts, stats_allmags[dat_key]['1yr'], 'o-')
        # plot 10yr area
        axes[1, i].plot(mag_cuts, stats_allmags[dat_key]['10yr'], 'o-')

for row in [0, 1]:
    axes[row, 0].ticklabel_format(style='sci',scilimits=(-3,4), axis='y')  # area
    axes[row, 2].legend(loc='upper right', ncol=2, fontsize=fontsize-2)
    for col in [0, 1, 2]:
        axes[row, col].set_ylabel(dat_keys[col], fontsize=fontsize)    
        axes[row, col].tick_params(axis='x', labelsize=fontsize-2)
        axes[row, col].tick_params(axis='y', labelsize=fontsize-2)
        
axes[0,1].set_title('1yr', fontsize=fontsize)
axes[1,1].set_title('10yr', fontsize=fontsize)
axes[1, 1].set_xlabel('i-band cut (i>?) (in all-band footprint with all depth > 0)', fontsize=fontsize)

fig.set_size_inches(20,10)
plt.show()

########################################################################################################################
# plot galactic latitude and EBV histograms for different cuts
print('\n## Plotting galactic latitude and EBV histograms for mag_cuts: %s'%mag_cuts)
# import EBV map from MAF
opsdb = db.OpsimDatabase(dbpath)
simdata = opsdb.fetchMetricData(['fieldRA', 'fieldDec'], sqlconstraint=None)  # ideally: use the RepRandomFPV Stacker
dustmap = maps.DustMap(nside=nside)
slicer = slicers.HealpixSlicer(latLonDeg=latLonDeg, nside=nside)
slicer.setupSlicer(simdata)
result = dustmap.run(slicer.slicePoints)
ebv_map = result['ebv']

# histograms: latitude, extinction
nBins_b = 150
nBins_ebv = 40
colors = ['m', 'g', 'b', 'r', 'c', 'y']

# plot
plt.clf()
fig, axes = plt.subplots(2,2)
fig.subplots_adjust(wspace=0.2, hspace=0.3)

max_counts = 0  # for EBV histogram; needed for plotting constant EBV lines
for i, yr in enumerate(['1yr', '10yr']):
    linestyle = 'solid'
    # first plot for no-cut
    # plot the galactic latitude histogram
    lon, lat = hp.pix2ang(nside=nside, ipix=allBandPixels[yr], lonlat=True)
    c = SkyCoord(ra= lon*u.degree, dec=lat*u.degree)
    axes[i, 0].hist(c.galactic.b.degree, label='no cut', bins=nBins_b,
                    histtype='step', lw=2, color='k')
    
    # plot the EBV histogram
    cts, _, _ = axes[i, 1].hist(np.log10(ebv_map[allBandPixels[yr]]), label='no cut',
                                bins=nBins_ebv, histtype='step', lw=2, color='k')
    max_counts = max(max_counts, max(cts))
    
    # now loop over mag cuts
    for j, mag_cut in enumerate(iCutPixels):
        if j>=len(colors): linestyle = 'dashed'  # out of colors so change linestyle
            
        # plot the galactic latitude histogram
        lon, lat= hp.pix2ang(nside=nside, ipix=iCutPixels[mag_cut][yr], lonlat= True)
        c = SkyCoord(ra= lon*u.degree, dec=lat*u.degree)
        axes[i, 0].hist(c.galactic.b.degree, label= 'i>%s'%mag_cut,
                        bins=nBins_b, histtype='step', lw=2,
                        color=colors[j%len(colors)], linestyle=linestyle)
        
        # plot the EBV histogram
        cts, _, _ = axes[i, 1].hist(np.log10(ebv_map[iCutPixels[mag_cut][yr]]), label= 'i>%s'%mag_cut,
                                    bins=nBins_ebv, histtype='step', lw=2,
                                    color=colors[j%len(colors)], linestyle=linestyle)
        max_counts = max(max_counts, max(cts))
    
for row in [0, 1]:
    x = np.arange(0,max_counts,10)
    for ebv in [0.2, 0.3]:
        axes[row, 1].plot(np.log10([ebv]*len(x)), x, '-.', label='EBV: %s'%ebv)
        
    axes[row, 0].set_xlabel('Galactic Latitude (deg)', fontsize=fontsize)
    axes[row, 1].set_xlabel(r'log$_{10}$ E(B-V)', fontsize=fontsize) 
    axes[row, 1].set_ylim(0,max_counts)
    
    for col in [0, 1]:
        axes[0, col].set_title('1yr', fontsize=fontsize)
        axes[1, col].set_title('10yr', fontsize=fontsize)
        axes[row, col].legend(loc='upper right', fontsize=fontsize-2)
        axes[row, col].set_ylabel('Pixel Counts', fontsize=fontsize)    
        axes[row, col].tick_params(axis='x', labelsize=fontsize-2)
        axes[row, col].tick_params(axis='y', labelsize=fontsize-2)
        
plt.suptitle('allBand footprint; all depths>0', fontsize=fontsize)
fig.set_size_inches(20,10)
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Finalized cuts
########################################################################################################################
print('#################################################################################################################')
print('#################################################################################################################')
print('#################################################################################################################')
print('#################################################################################################################')
# chosen mag cuts: 24.5 for 1yr, 26.0 for 10yr
chosen_cuts = {'1yr': [24.5], '10yr': [26.0]}  # need lists since will loop over things
print('Chosen cuts: %s'%chosen_cuts)

# histogram latitude, extinction
nBins_b = 150
nBins_ebv = 40
colors = ['m', 'g', 'b', 'r', 'c', 'y']

plt.clf()
fig, axes = plt.subplots(2,2)
fig.subplots_adjust(wspace=0.2, hspace=0.3)

max_counts = 0
for i, yr in enumerate(['1yr', '10yr']):
    linestyle = 'solid'
    # first plot for no-cut
    # plot the galactic latitude histogram
    lon, lat = hp.pix2ang(nside=nside, ipix=allBandPixels[yr], lonlat=True)
    c = SkyCoord(ra= lon*u.degree, dec=lat*u.degree)
    axes[i, 0].hist(c.galactic.b.degree, label='no cut', bins=nBins_b, histtype='step', lw=2, color='k')
    
    # ----------
    # print out area for which EBV>0.2, 0.3
    tot = areaPerPixel*len(allBandPixels[yr])
    print('\n%s: no depth cut\nTotal area (allBand footprint; all depths>0): %.2f deg2'%(yr, tot))
    for lim in [0.2, 0.3]:
        area_here = len(np.where(ebv_map[allBandPixels[yr]]>lim)[0])*areaPerPixel
        print('EBV>%s: Area: %.2f deg2 (%.2f%% of total)'%(lim, area_here, 100.*area_here/tot))
    # ----------
    
    # plot the EBV histogram
    cts, _, _ = axes[i, 1].hist(np.log10(ebv_map[allBandPixels[yr]]), label='no depth cut',
                                bins=nBins_ebv, histtype='step', lw=2, color='k')
    max_counts = max(max_counts, max(cts))
    
    for j, mag_cut in enumerate(chosen_cuts[yr]):
        # plot the galactic latitude histogram
        lon, lat= hp.pix2ang(nside=nside, ipix=iCutPixels[mag_cut][yr], lonlat= True)
        c = SkyCoord(ra= lon*u.degree, dec=lat*u.degree)

        axes[i, 0].hist(c.galactic.b.degree, label= 'i>%s'%mag_cut,
                        bins=nBins_b, histtype='step', lw=2, color=colors[j%len(colors)])
        
        # ----------
        # print out area for which EBV>0.2, 0.3
        tot = areaPerPixel*len(iCutPixels[mag_cut][yr])
        print('\n%s:\nTotal area: i>%s: allBand footprint; all depths>0: %.2f deg2'%(yr, mag_cut, tot))
        for lim in [0.2, 0.3]:
            area_here = len(np.where(ebv_map[iCutPixels[mag_cut][yr]]>lim)[0])*areaPerPixel
            print('EBV>%s: Area: %.2f deg2 (%.2f%% of total)'%(lim, area_here, 100.*area_here/tot))
        # ----------
    
        if j>=len(colors): linestyle = 'dashed' # out of colors so change linestyle
        # plot the EBV histogram
        cts, _, _ = axes[i, 1].hist(np.log10(ebv_map[iCutPixels[mag_cut][yr]]),
                                    label= 'i>%s'%(mag_cut),
                                    bins=nBins_ebv, histtype='step', lw=2,
                                    color=colors[j%len(colors)], linestyle=linestyle)
        max_counts = max(max_counts, max(cts))
    
for row in [0, 1]:
    x = np.arange(0,max_counts,10)
    for ebv in [0.2, 0.3]: # add lines for constant EBV
        axes[row, 1].plot(np.log10([ebv]*len(x)), x, '-.', label='EBV: %s'%ebv)
    axes[row, 0].set_xlabel('Galactic Latitude (deg)', fontsize=fontsize)
    axes[row, 1].set_xlabel(r'log$_{10}$ E(B-V)', fontsize=fontsize) 
    axes[row, 1].set_ylim(0,max_counts)
    
    for col in [0, 1]:
        axes[0, col].set_title('1yr', fontsize=fontsize)
        axes[1, col].set_title('10yr', fontsize=fontsize)
        axes[row, col].legend(loc='upper right', fontsize=fontsize-2)
        axes[row, col].set_ylabel('Pixel Counts', fontsize=fontsize)    
        axes[row, col].tick_params(axis='x', labelsize=fontsize-2)
        axes[row, col].tick_params(axis='y', labelsize=fontsize-2)
plt.suptitle('allBand footprint; all depths>0', fontsize=fontsize)
fig.set_size_inches(20,10)
plt.show()

########################################################################################################################
# plot skymaps for each band before and after the depth cut
nTicks = 5
for band in orderBand:
    for yr in ['1yr', '10yr']:
        plt.clf()
        fig, axes= plt.subplots(1,2)
        temp = copy.deepcopy(data_bundle['%s_%s'%(yr, band)])  # need to copy since will change the mask
        
        # retain the all-band footprint only
        temp.metricValues.mask = True
        temp.metricValues.mask[allBandPixels[yr]] = False
        
        # figure out the color range
        inSurveyIndex = np.where(temp.metricValues.mask == False)[0]
        median = np.median(temp.metricValues.data[inSurveyIndex])
        stddev = np.std(temp.metricValues.data[inSurveyIndex])
        colorMin = median-2.5*stddev
        colorMax = median+2.5*stddev
        increment = (colorMax-colorMin)/float(nTicks)
        ticks = np.arange(colorMin+increment, colorMax, increment)
        
        # plot the no-cut skymap
        plt.axes(axes[0])
        hp.mollview(temp.metricValues.filled(temp.slicer.badval), 
                    flip='astro', rot=(0,0,0) , hold= True,
                    min=colorMin, max=colorMax,
                    title= '%s: No Depth Cut'%yr, cbar=False)
        hp.graticule(dpar=20, dmer=20, verbose=False)
        
        # plot the mag-cut skymap
        temp.metricValues.mask = True
        temp.metricValues.mask[iCutPixels[chosen_cuts[yr][0]][yr]] = False
        
        plt.axes(axes[1])
        hp.mollview(temp.metricValues.filled(temp.slicer.badval), 
                    flip='astro', rot=(0,0,0) , hold= True,
                    min=colorMin, max=colorMax,
                    title= '%s: i>%s'%(yr, chosen_cuts[yr][0]), cbar=False)
        hp.graticule(dpar=20, dmer=20, verbose=False)
            
        # add a color bar
        im = plt.gca().get_images()[0]
        cbaxes = fig.add_axes([0.25, 0.38, 0.5, 0.01]) # [left, bottom, width, height]
        cb = plt.colorbar(im,  orientation='horizontal',
                          ticks=ticks, format='%.2f', cax=cbaxes) 
        cb.set_label('%s-band depth\n(all-band footprint with all depth > 0)'%(band), fontsize=14)
        cb.ax.tick_params(labelsize=14)
        
        fig.set_size_inches(18,18)
        plt.show()