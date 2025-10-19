import logging
import os
import re

import matplotlib
import requests
import time
import warnings
import numpy
import pandas as pd
from matplotlib import pyplot
from requests_toolbelt.multipart.encoder import MultipartEncoder
from .utils import print_progress_bar

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
LIPIDMAPS_URL = 'http://lipidmaps-dev.babraham.ac.uk/tools/ms/py_bulk_search.php'
BATCH_SIZE = 500


def create_summary(data, dst=''):
    mzList = data['Input Mass'].unique().tolist()
    summary = pd.DataFrame(columns=list(data))
    for name, group in data.groupby(['Input Mass', 'rt']):
        categories = group['Category'].value_counts()
        if categories.empty:
            summary = summary.append(group, ignore_index=True)
        else:
            bestCategory = categories.index[0]
            # Keep only those rows for the most frequent category
            subgroup = group[group['Category'] == bestCategory]
            if bestCategory == 'other metabolites':
                if (len(categories) == 1) or (categories[0] > categories[1]):
                    summary = summary.append(subgroup.head(1),
                                             ignore_index=True)
                    continue
                else:
                    bestCategory = categories.index[1]
                    subgroup = group[group['Category'] == bestCategory]
            mainClass = subgroup['Main Class'].value_counts().index[0]
            summary = summary.append(
                subgroup[subgroup['Main Class'] == mainClass].head(1),
                ignore_index=True)
    # Create XLSX file with the summary putative profiling in 'dst'
    fileName = 'mssearch_{0}_summary.tab'.format('COMP_DB')
    summary.to_csv(os.path.join(dst, fileName), index=False, sep='\t')


CATEGORIES = ['unknown', 'sterol lipids', 'sphingolipids', 'saccharolipids',
              'prenol lipids', 'polyketides', 'other metabolites',
              'glycerophospholipids', 'glycerolipids', 'fatty acyls']


def category_scatterplot(data, dst):
    # Choose the color palette to assign to each lipid category

    colors = ['#111111', '#ffC0cb', '#1e90ff', '#00ffff', '#ffd700',
              '#ff1493', '#ff8c00', '#32cd32', '#ff0000', '#9370db']
    markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    widths = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data['Category'].fillna('unknown', inplace=True)

    catData = _get_main_categories(data)
    # Configure the plot style, layout and parameters
    matplotlib.style.use('seaborn-paper')
    ax = pyplot.subplot(111)
    # Calculate the ceiling of the 102% of the maximum retention time
    maxRT = numpy.amax(catData['rt'].values)
    maxX = numpy.ceil(1.02 * maxRT)
    # Calculate the ceiling of the 110% of the maximum m/z
    maxMZ = numpy.amax(catData['rt'].values)
    maxY = numpy.ceil(1.1 * maxMZ)
    # Set range of X and Y axes
    pyplot.xlim([0, maxX])
    pyplot.ylim([0, maxY])
    # Set label text of X and Y axes
    pyplot.xlabel('Retention time (min)')
    pyplot.ylabel('m/z', fontstyle='italic')
    # Set a readable and color-blind friendly marker cycle
    ax.set_prop_cycle(
        cycler('marker', markers) + cycler('color', colors) + \
        cycler('markersize', sizes) + cycler('markeredgewidth', widths))
    # Load each category to the plot
    for i, category in enumerate(CATEGORIES):
        catMatches = catData.loc[catData['Category'].str.lower() == category]
        if len(catMatches) == 0:
            # Skip to the next marker so each category will always have
            # the same one, allowing an ease comparison between plots
            next(ax._get_lines.prop_cycler)
        else:
            pyplot.plot(catMatches['rt'], catMatches['Input Mass'], linestyle='None',
                        markeredgecolor='#666666',
                        label=string.capwords(category))
    # Get handles and labels for legend
    handles, labels = ax.get_legend_handles_labels()
    numCats = len(labels)
    # Change the axes position of the plot to leave some room under it
    # to display the legend based on its number of rows for an up to
    # 5-column layout
    box = ax.get_position()
    numRows = numpy.ceil(numCats / 5.0)
    yShift = numRows * 0.05
    ax.set_position([box.x0, box.y0 + box.height * yShift, box.width,
                     box.height * (1.0 - yShift)])
    # Reverse labels in legend to sort them alphabetically, locate the
    # legend below the plot and display the categories in 1 or 2 rows
    # and up to 5 columns
    numCols = int(numpy.ceil(numCats / numRows))
    ax.legend(handles[::-1], labels[::-1], loc='upper center',
              bbox_to_anchor=(0.5, -0.12), fancybox=False, shadow=False,
              ncol=numCols, numpoints=1)
    # Adjust plot parameters to fit it into the figure area
    pyplot.tight_layout()
    # Save the figure into the selected file format and close it
    figName = 'category_scatterplot_{0}.{1}'.format(
        'COMP_DB', 'png')
    figPath = os.path.join(dst, figName)
    pyplot.savefig(figPath, dpi=600, bbox_inches='tight')
    pyplot.close()


def _get_main_categories(data, defaultCat='other metabolites'):
    # Check the default category is within the expected values
    if defaultCat not in CATEGORIES:
        raise ValueError("'defaultCat' must be one of {0}".format(CATEGORIES))
    # Get count the number of matches per m/z, RT and category
    categoryCounts = pd.DataFrame(
        {'Count': data.groupby(['Input Mass', 'rt', 'Category'],
                               sort=True).size()})
    categoryCounts.reset_index(inplace=True)
    # Group the category counts by m/z and RT
    groupedData = categoryCounts.groupby(['Input Mass', 'rt'])
    # Set up a new dataframe to save the most frequent categories
    catData = pd.DataFrame()
    for name, group in groupedData:
        # Append the row with the most frequent category for each m/z
        # and RT (excluding "Count" column)
        if len(group) == 1:
            catData = catData.append(group[['Input Mass', 'rt', 'Category']],
                                     ignore_index=True)
        else:
            # When there are two or more categories, pick the
            # non-default most frequent one
            index = group.loc[group['Category'].str.lower() != defaultCat,
                              'Count'].idxmax()
            catData = catData.append(
                group.loc[index, ['Input Mass', 'rt', 'Category']],
                ignore_index=True)
    return catData


def bulk_structure_search(data, dst=''):
    logFilePath = 'mssearch.log'
    if dst:
        logFilePath = os.path.join(dst, logFilePath)
    # Create logger and its file handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logFilePath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Write initial information in log file
    logger.info('Starting MS Search on %s. Input dataframe has %d rows.',
                'COMP_DB', len(data.index))
    # Start progress bar
    progress = 0
    print_progress_bar(progress, 100, prefix='MSSearch progress:')
    # Get the list of unique m/z values from 'data'
    mzList = data['mz'].unique().tolist()
    numMZ = len(mzList)
    logger.info('%d unique m/z values found.', numMZ)
    targetAdducts = ['[M.OAc]-', '[M-H]-']
    targetAdducts = [x[x.find('[') + 1: x.find(']')] for x in targetAdducts]
    targetAdducts = ','.join(targetAdducts)
    matches = pd.DataFrame()
    # Calculate progress increment for each batch
    increment = 63.0 / numpy.ceil(float(numMZ) / BATCH_SIZE)
    for start in range(0, numMZ, BATCH_SIZE):
        mzBatch = mzList[start: start + BATCH_SIZE]
        # Get a string with one m/z per line (text file alike)
        mzStr = os.linesep.join(map(str, mzBatch))
        tolerance = mzBatch[-1] * (4.0 / 1e6)
        categories = ["Fatty Acyls [FA]", "Glycerolipids [GL]", "Glycerophospholipids [GP]", "Sphingolipids [SP]",
                      "Sterol Lipids [ST]", "Prenol Lipids [PR]",
                      "Saccharolipids [SL]", "Polyketides [PK]"]
        mpData = MultipartEncoder(
            fields={'CHOICE': 'COMP_DB', 'sort': 'DELTA',
                    'file': ('file', StringIO(mzStr), 'text/plain'),
                    'tol': str(tolerance), 'ion': targetAdducts,
                    'even': '1',
                    'category': ','.join(categories)})

        response = requests.post(LIPIDMAPS_URL, data=mpData, headers={'Content-Type': mpData.content_type})
        if len(response.text) == 0:
            # Update progress bar
            progress += increment
            print_progress_bar(progress, 100, prefix='MSSearch progress:')
            continue
        # Process the response to create a dataframe
        batchMatches = pd.read_csv(StringIO(response.text), sep='\t',
                                   engine='python', index_col=False)
        if batchMatches.empty:
            # Update progress bar
            progress += increment
            print_progress_bar(progress, 100, prefix='MSSearch progress:')
            continue
        # Join all the information already gathered
        matches = matches.append(batchMatches, ignore_index=True)
        # Update progress bar
        progress += increment
        print_progress_bar(progress, 100, prefix='MSSearch progress:')
    matches['Input Mass'] = matches['Input Mass'].apply(round, ndigits=6)
    # Calculate the delta PPM of each row and add it to the dataframe
    dPPM = abs(matches['Input Mass'] - matches['Matched MZ']) * 1e6 / matches['Input Mass']
    matches.insert(2, 'Delta_PPM', dPPM)
    matches.insert(3, 'rt', 0.0)
    matches.insert(4, 'Polarity', '')
    # Calculate progress increment for each batch
    increment = 33.0 / numpy.ceil(len(data) / float(BATCH_SIZE))
    # Create result dataframe with all the columns in that dataframe
    colNames = [x for x in list(data) if x not in ['', 'rt', 'Polarity']]
    extraCols = []
    for column in colNames:
        if column not in list(matches):
            extraCols.append(column)
        else:
            # Keep all columns from source dataset, adding prefix "src_"
            # if that column name is already in the dataframe
            extraCols.append('src_' + column)
            data.rename(columns={column: 'src_' + column}, inplace=True)
    result = pd.DataFrame(columns=list(matches) + extraCols)
    # Ensure the polarity column contains only strings so the
    # conditional test in the next loop works as expected
    data['Polarity'].replace(numpy.nan, '', regex=True, inplace=True)
    # For those m/z values with more than one RT, the whole set of
    # matches is replicated for every RT
    for index, row in data.iterrows():
        mzMatches = matches.loc[matches['Input Mass'] == row['Input Mass']]
        # Remove positive adduct matches for m/z found in negative mode,
        # and negative adduct matches for m/z found in positive mode
        if row['Polarity'].lower().startswith('n'):
            mzMatches = mzMatches.loc[mzMatches['Adduct'].str[-1] != '+']
        elif row['Polarity'].lower().startswith('p'):
            mzMatches = mzMatches.loc[mzMatches['Adduct'].str[-1] != '-']
        if mzMatches.empty:
            # Unmatched m/z from 'data'
            mzMatches = mzMatches.append(row[['Input Mass', 'rt', 'Polarity']],
                                         ignore_index=True)
        else:
            # Copy RT and polarity values to each matched m/z
            mzMatches['rt'] = row['rt']
            mzMatches['Polarity'] = row['Polarity']
        # Copy the extra columns (if any) to each matched m/z
        for col in extraCols:
            mzMatches[col] = row[col]
        result = result.append(mzMatches, ignore_index=True)
        if (index + 1) % BATCH_SIZE == 0:
            # Update progress bar
            progress += increment
            print_progress_bar(progress, 100, prefix='MSSearch progress:')
    result.sort_values(['Input Mass', 'Delta_PPM', 'Matched MZ'], inplace=True,
                       kind='mergesort')
    outPath = os.path.join(
        dst, 'mssearch_{0}.tab'.format('COMP_DB'))
    result.to_csv(outPath, index=False, sep='\t')
    create_summary(result, dst)
    # Update progress bar
    print_progress_bar(98, 100, prefix='MSSearch progress:')
    # Generate the category scatter plot of the most common lipid
    # category per m/z and RT

    category_scatterplot(result, dst)
    # Update progress bar
    print_progress_bar(100, 100, prefix='MSSearch progress:')
    # Write the final information in log file and close handler
    matches = result[result['Category'].notna()]
    logger.info('MS Search completed. %d matches found for %d m/z values.\n',
                len(matches), len(matches['Input Mass'].unique()))
    handler.close()
    logger.removeHandler(handler)
