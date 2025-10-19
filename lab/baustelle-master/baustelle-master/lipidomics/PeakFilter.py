import re

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance

from lipidomics.utils import mz_delta


def qc_rsd_ratio(data):
    # filter out all the qc tables and then calculate the rsd
    meanColName = colName + '_mean'
    rsdColName = colName + '_RSD'
    # Calculate and insert in the input dataframe the mean and RSD of QC
    # samples (row-wise).
    data[meanColName] = data.iloc[:, firstIndex : lastIndex].mean(axis=1)
    # pandas.DataFrame.std() returns sample standard deviation over
    # requested axis, so we need to change the degrees of freedom (ddof)
    # to 0 to get the population standard deviation.
    data[rsdColName] = data.iloc[:, firstIndex : lastIndex].std(axis=1, ddof=0)\
                       * 100.0 / data[meanColName]
    # Return the ratio of QC samples with RSD less than QCRSD's lower
    # part to QC samples with RSD less than QCRSD's upper part.
    lowerRSDCount = (data[rsdColName] < parameters['QCRSD'][0]).sum()
    upperRSDCount = (data[rsdColName] < parameters['QCRSD'][1]).sum()
    return round(lowerRSDCount / float(upperRSDCount) * 100, 1)


def cluster_by_mz(data):
    auxData = pd.DataFrame(
        {'mzDiffNextFrame': data['mz'].shift(-1) - data['mz']})
    auxData['mzClusterSectionID'] = np.nan
    # Calculate the cluster section ID for each m/z
    numRowsData = len(data)
    sectionBegin = 0
    # Minimum amount of m/z that will belong to the same cluster section
    sectionMinSize = 49
    clusterSectionID = 1
    while (numRowsData - sectionBegin) >= sectionMinSize:
        sectionEnd = sectionBegin + sectionMinSize
        while sectionEnd < (numRowsData - 1):
            currentDelta = mz_delta(data.loc[sectionEnd, 'mz'], 0.0005, 4.0)
            nextDelta = mz_delta(data.loc[sectionEnd + 1, 'mz'], 0.0005, 4.0)
            if auxData.iloc[sectionEnd, 0] > (currentDelta + nextDelta):
                break
            sectionEnd += 1
        sectionEnd += 1
        auxData.iloc[sectionBegin: sectionEnd, 1] = clusterSectionID
        clusterSectionID += 1
        sectionBegin = sectionEnd
    if sectionBegin < numRowsData:
        auxData.iloc[sectionBegin: numRowsData, 1] = clusterSectionID
    else:
        # The last cluster section ID was not used so get the total
        # number of IDs assigned
        clusterSectionID -= 1
    # Add a column to dataframe where the mass cluster IDs will be saved
    data['mzClusterID'] = np.nan
    currentMaxClusterID = 0
    for sectionID in range(1, clusterSectionID + 1):
        sectionRows = auxData.iloc[:, 1] == sectionID
        # Copy the masses in the current cluster into a list of single
        # item lists (one per mass)
        vectorMZ = data.loc[sectionRows, 'mz'].values.reshape((-1, 1))
        if len(vectorMZ) == 1:
            # Give the next cluster ID to the item and move to next
            # cluster section
            currentMaxClusterID += 1
            data.loc[sectionRows, 'mzClusterID'] = currentMaxClusterID
        else:
            # Perform hierarchical clustering:
            # Get maximum m/z error in current cluster (based on maximum
            # m/z). This will be the cut off for hierarchical clustering.
            maxMZ = data.loc[sectionRows, 'mz'].max()
            currentMaxMZError = 2 * mz_delta(maxMZ, 0.0005,4.0)
            # Calculate distance between every mass in cluster section
            mzDistMatrix = distance.pdist(vectorMZ)
            # Calculate linkage
            mzLinkage = hierarchy.complete(mzDistMatrix)
            # Return a list of flat cluster IDs for each mass, shifting
            # the numbers by the last assigned cluster ID
            mzClusters = hierarchy.fcluster(mzLinkage, currentMaxMZError,
                                            'distance') + currentMaxClusterID
            # Add this information to the dataframe
            data.loc[sectionRows, 'mzClusterID'] = mzClusters
            # Increment the current cluster ID by the number of unique
            # clusters in the current mass section
            currentMaxClusterID += len(set(mzClusters))
    # Renumber Cluster IDs based on their appearance in the dataframe
    clusterIDs = data['mzClusterID'].values
    id = 1
    numRowsData = len(data)
    for index in range(0, numRowsData - 1):
        clusterIDs[index] = id
        if clusterIDs[index] != clusterIDs[index + 1]:
            id += 1
    clusterIDs[numRowsData - 1] = id


def cluster_by_features(data):
    # Re-sort dataframe ready for feature clustering
    data.sort_values(by=['mzClusterID', 'rt', 'mz'], inplace=True,
                     kind='mergesort')
    # Reset index
    data.reset_index(inplace=True, drop=True)
    # Create a new dataframe with auxiliary information:
    # "TimeDiff": retention time difference between current and next
    #     frames
    auxData = pd.DataFrame(
        {'TimeDiff': data['rt'].shift(-1) - data['rt']})
    # Assign a feature cluster ID to each cluster of contiguous
    # ions within the same mass cluster where each member is separated
    # by a retention time difference of less than 'maxRTDiffAdjFrame'
    data['FeatureClusterID'] = np.nan
    timeDiffs = auxData['TimeDiff'].values
    mzClusterIDs = data['mzClusterID'].values
    featureClusterIDs = data['FeatureClusterID'].values
    id = 1
    numRowsData = len(data)
    for index in range(0, numRowsData - 1):
        featureClusterIDs[index] = id
        if ((mzClusterIDs[index] != mzClusterIDs[index + 1])
                or (timeDiffs[index] > 0.3)):
            id += 1
    featureClusterIDs[numRowsData - 1] = id


def correct_retention_time(data, parameters, means=False):
    data = data.groupby(['FeatureClusterID']).apply(
            __process_feature__, parameters=parameters, means=means)
    if (not means):
        # Drop empty frames (if any)
        data.drop_empty_frames('Empty frames after Retention Time Correction',
                               parameters)


def __process_feature__(featureCluster, parameters, means):
    if len(featureCluster) == 1:
        return featureCluster
    if (means):
        # The sample means for the feature cluster
        tmpData = featureCluster.iloc[:, -parameters['numSamples'] : ].copy()
        # Get the index of frames with at least 1 column with a non-zero
        # intensity
        nonZeroIndices = np.where(tmpData.sum(axis=1) > 0)[0]
        if nonZeroIndices.size > 1:
            # Get array of retention times (RT)
            rtArray = featureCluster[parameters['rtCol']].values
            # Get an array of the time difference to next frame
            rtDiff = np.roll(rtArray[nonZeroIndices], -1) \
                    - rtArray[nonZeroIndices]
            # Get the array of intensities for the frames with at least
            # 1 column with a non-zero intensity
            intensity = tmpData.values[nonZeroIndices]
            __process_sample__(intensity, rtDiff, parameters,
                               parameters['numSamples'])
            # Replace old values with the new ones
            tmpData.values[nonZeroIndices] = intensity
            featureCluster.iloc[:, -parameters['numSamples'] : ] = tmpData
    else:
        firstSampleIndex = parameters['firstSampleIndex'] - 1
        lastSampleIndex = firstSampleIndex + (parameters['numSamples']
                                              * parameters['numTechReps'])
        # Get array of RTs
        rtArray = featureCluster[parameters['rtCol']].values
        # Loop through each set of replicates per sample
        for firstIndex in range(firstSampleIndex, lastSampleIndex,
                                parameters['numTechReps']):
            lastIndex = firstIndex + parameters['numTechReps']
            tmpData = featureCluster.iloc[:, firstIndex : lastIndex].copy()
            # Get the index of frames with at least 1 replicate with a
            # non-zero intensity
            nonZeroIndices = np.where(tmpData.sum(axis=1) > 0)[0]
            if (nonZeroIndices.size > 1):
                # Get an array of the time difference to next frame
                rtDiff = np.roll(rtArray[nonZeroIndices], -1) \
                        - rtArray[nonZeroIndices]
                # Get the array of intensities for the frames with at least
                # 1 replicate with a non-zero intensity
                intensity = tmpData.values[nonZeroIndices]
                __process_sample__(intensity, rtDiff, parameters,
                                   parameters['numTechReps'])
                # Replace old values with the new ones
                tmpData.values[nonZeroIndices] = intensity
                featureCluster.iloc[:, firstIndex : lastIndex] = tmpData
    return featureCluster


def __process_sample__(intensity, rtDiff, parameters, repsPerGroup):

    while True:
        # Copy 'intensity' array to check later if it has been modified
        oldIntensity = np.copy(intensity)
        # Number of frames and replicates in the given feature cluster
        numRows, numCols = intensity.shape
        for rep in range(0, numCols):
            for row in range(0, numRows):
                if (intensity[row][rep] != 0):
                    continue
                # Require at least half non-zero intensity values
                elif ((2 * np.count_nonzero(intensity[row]))
                      >= repsPerGroup):
                    # Adjacent frame (row -/+ 1) intensity values
                    adjFrameValues = [0, 0]
                    if ((row > 0) and (intensity[row - 1][rep] != 0)
                        and (rtDiff[row - 1]
                             < parameters['maxRTDiffAdjFrame'])):
                        # The frame above has a non-zero intensity and
                        # is within the allowed retention time (RT)
                        # threshold
                        adjFrameValues[0] = intensity[row - 1][rep]
                    if ((row < (numRows - 1)) and (intensity[row + 1][rep] != 0)
                        and (rtDiff[row] < parameters['maxRTDiffAdjFrame'])):
                        # The frame below has a non-zero intensity and
                        # is within the allowed RT threshold
                        adjFrameValues[1] = intensity[row + 1][rep]
                    if (any(adjFrameValues)):
                        # Save the contiguous frame (if any) where to
                        # swap the intensity values
                        swapIndex = 0
                        # At least one contiguous intensity is greater
                        # than zero. Get mean and standard deviation of
                        # current frame (non-zero values).
                        repMean = intensity[row][np.nonzero(
                                intensity[row])[0]].mean()
                        repStdDev = intensity[row][np.nonzero(
                                intensity[row])[0]].std()
                        # Calculate the maximum standard deviation
                        stDev = parameters['intensityStDev'] * repStdDev
                        if ((adjFrameValues[0] != 0)
                            and (adjFrameValues[0] >= repMean - stDev)
                            and (adjFrameValues[0] <= repMean + stDev)):
                            if ((2 * np.count_nonzero(intensity[row - 1]))
                                < repsPerGroup):
                                swapIndex = -1
                            elif ((2 * np.count_nonzero(intensity[row - 1]))
                                  == repsPerGroup):
                                prevFrameMean = intensity[row - 1][
                                        np.nonzero(intensity[row - 1])[0]
                                        ].mean()
                                if (repMean >= prevFrameMean):
                                    swapIndex = -1
                        if ((adjFrameValues[1] != 0)
                            and (adjFrameValues[1] >= repMean - stDev)
                            and (adjFrameValues[1] <= repMean + stDev)):
                            # If 'swapIndex' is not 0, swap with the
                            # closest intensity value to the mean of the
                            # current frame
                            if ((swapIndex == 0)
                                or ((swapIndex != 0)
                                    and (abs(repMean - adjFrameValues[1])
                                         < abs(repMean - adjFrameValues[0])))):
                                nextNonZeroReps = np.count_nonzero(
                                        intensity[row + 1])
                                if ((2 * nextNonZeroReps) < repsPerGroup):
                                    swapIndex = 1
                                elif ((2 * nextNonZeroReps) == repsPerGroup):
                                    nextFrameMean = intensity[row + 1][
                                            np.nonzero(intensity[row + 1])[0]
                                            ].mean()
                                    if (repMean >= nextFrameMean):
                                        swapIndex = 1
                        if (swapIndex != 0):
                            # Swap with the chosen contiguous frame
                            intensity[row][rep] = \
                                    intensity[row + swapIndex][rep]
                            intensity[row + swapIndex][rep] = 0
        # Repeat the process until no more modifications are performed
        if (np.array_equal(intensity, oldIntensity)):
            break


ISO_OFFSET = 1.003354838


def remove_isotopes(data, parameters):
    mzCol = parameters['mzCol']
    rtCol = parameters['rtCol']
    # Calculate the location of sample columns based on the current
    # state of the dataframe (before adding isotope annotation)
    firstSampleCol = len(data.columns) - parameters['numSamples']
    lastSampleCol = len(data.columns)
    for i in range(firstSampleCol, lastSampleCol):
        # Create an array from 'data' with m/z, retention time, the
        # samples' intensity mean and index per row
        array = numpy.stack((data[mzCol].values, data[rtCol].values,
                             data.iloc[:, i], data.iloc[:, 0].values), axis=-1)
        tagArray = _detect_sample_isotopes(array, parameters)
        # Set the intensity of the sample detected isotopes to 0
        colName = data.columns[i]
        isoColName = colName + '_isotopes'
        data.insert(len(data.columns), isoColName, tagArray)
        if (parameters['removeIsotopes']):
            data.loc[data[isoColName].str.contains('M\+'), colName] = 0.0
    if (parameters['removeIsotopes']):
        # Drop empty frames, i.e. isotope frames found in every sample
        data.drop_empty_frames(
                'Isotope removal (isotopes found in every sample)', parameters,
                True)


def _detect_sample_isotopes(array, parameters):
    """Return an array with the tagged parents and their corresponding
    isotopes in the same order as in the given sample array.
    Keyword Arguments:
        array      -- array with m/z, retention time (RT), sample's
                      intensity mean and index of the original dataframe
        parameters -- LipidFinder's PeakFilter parameters instance
    """
    # Get the corresponding symbol for the polarity of the data (+ or -)
    polSign = '+' if (parameters['polarity'].lower() == 'positive') else '-'
    # Create an array of empty strings that will contain the tagged
    # parents and their corresponding isotopes
    tagArray = numpy.full(len(array), '', dtype=object)
    # Loop over each m/z to search for isotopes
    isotopesIndex = set()
    for index in range(0, len(array)):
        # Skip if frame has already been identified as an isotope
        if (array[index, 3] in isotopesIndex):
            continue
        for isoPeak in range(1, parameters['numIsotopes'] + 1):
            parentMZ = array[index, 0]
            tagID = int(array[index, 3])
            # Get the first and last indexes of the frames that are
            # within the first isotope m/z range for the current analyte
            isotopeMZ = parentMZ + ISO_OFFSET * isoPeak
            minMZ, maxMZ = mz_tol_range(isotopeMZ, parameters['mzFixedError'],
                                        parameters['mzPPMError'])
            mzMatches = numpy.searchsorted(array[:, 0], [minMZ, maxMZ])
            if (mzMatches[0] == mzMatches[1]):
                # Have not found any analyte with an isotope-like m/z
                if (isoPeak == 1):
                    # The first isotope must exists to search for others
                    break
                else:
                    continue
            # Filter m/z matches with the same RT as the parent
            parentRT = array[index, 1]
            minRT, maxRT = rt_tol_range(parentRT,
                                        parameters['maxRTDiffAdjFrame'])
            rtMatches = numpy.where(
                    (array[mzMatches[0] : mzMatches[1], 1] >= minRT)
                    & (array[mzMatches[0] : mzMatches[1], 1] <= maxRT))[0]
            if (len(rtMatches) == 0):
                # No candidates are within the same RT
                if (isoPeak == 1):
                    # The first isotope must exists to search for others
                    break
                else:
                    continue
            # Resultant indexes are based on the previous search
            rtMatches += mzMatches[0]
            # Filter the candidate isotopes by intensity
            parentInten = array[index, 2]
            # The intensity range coefficients vary depending on the
            # isotope number
            if (isoPeak == 1):
                # Get an estimated maximum number of C in the molecule
                numC = round(parentMZ / 12)
                # Calculate isotopic distribution based on polynomial
                # expansion
                baseIntensity = parentInten * (numC ** 1.3) * 0.002
                minIntensity = baseIntensity * parameters['isoIntensityCoef'][0]
                maxIntensity = baseIntensity * parameters['isoIntensityCoef'][1]
            elif (isoPeak == 2):
                # Get an estimated maximum number of C in the molecule
                numC = round(parentMZ / 12)
                # Calculate isotopic distribution based on polynomial
                # expansion
                baseIntensity = parentInten * (numC ** 1.7) * 0.0001
                minIntensity = baseIntensity * parameters['isoIntensityCoef'][0]
                maxIntensity = baseIntensity * parameters['isoIntensityCoef'][1]
            else:
                # Calculate isotopic distribution with the same formula
                # as CAMERA (from XCMS)
                minIntensity = parentInten * float('1e-{0}'.format(isoPeak + 2))
                maxIntensity = parentInten * 2
            isotopes = numpy.where((array[rtMatches, 2] >= minIntensity)
                                   & (array[rtMatches, 2] <= maxIntensity))[0]
            if (len(isotopes) == 0):
                # No candidates have an intensity within expected range
                if (isoPeak == 1):
                    # The first isotope must exists to search for others
                    break
                else:
                    continue
            # Resultant indexes are based on the previous search
            isotopes += rtMatches[0]
            # Tag the analyte as isotope and save its index to avoid
            # checking it as parent of other analytes
            tagArray[isotopes] = '[{0}][M+{1}]{2}'.format(tagID, isoPeak,
                                                          polSign)
            isotopesIndex.update(array[isotopes, 3])
        else:
            # Tag the analyte as parent
            tagArray[index] = '[{0}][M]{1}'.format(tagID, polSign)
    return tagArray