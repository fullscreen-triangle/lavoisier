import glob
import logging
import os

import pandas as pd


class LDataFrame(pd.DataFrame):

    def __init__(self, src, resolution=6):

        if not os.path.isdir(src):
            data = self._read_file(src)
        else:
            # Create a list of the input files in the source folder (in
            # alphabetical order)
            fileList = sorted(glob.iglob(os.path.join(src, '*.*')))
            if len(fileList) == 0:
                raise FileNotFoundError("No files found in '{0}'".format(src))
            data = self._read_file(fileList[0])
            if len(fileList) > 1:
                # Sort first dataframe by RT
                data.sort_values(['rt'], inplace=True, kind='mergesort')
                timeCol = 'minute'
                data = data.assign(minute=data['rt'].astype(int))
                data = data[data[timeCol] != data.iloc[-1][timeCol]]
                for index, filePath in enumerate(fileList[1:], start=1):
                    chunk = self._read_file(filePath)
                    # Sort next chunk dataframe by RT
                    chunk.sort_values(['mz'], inplace=True, kind='mergesort')
                    # Append "minute" column to the dataframe with the
                    # integer part of the float values of its RT column
                    chunk = chunk.assign(minute=chunk['rt'].astype(int))
                    # Remove the frames of the first minute
                    chunk = chunk[chunk[timeCol] != chunk.iloc[0][timeCol]]
                    if index < (len(fileList) - 1):
                        chunk = chunk[chunk[timeCol] != chunk.iloc[-1][timeCol]]
                    overlap = pd.DataFrame(
                        {'data': data.groupby(timeCol).size(),
                         'chunk': chunk.groupby(timeCol).size()}
                    ).fillna(0)
                    # Keep the minutes where the number of frames in the
                    # next chunk is higher than in the current dataframe
                    overlap = overlap[overlap['chunk'] > overlap['data']]
                    minutesToReplace = overlap.index.tolist()
                    if minutesToReplace:
                        # Remove the dataframe frames to be replaced
                        data = data[~data[timeCol].isin(minutesToReplace)]
                        # Append chunk frames preserving the column
                        # order of the main dataframe
                        data = data.append(
                            chunk[chunk[timeCol].isin(minutesToReplace)],
                            ignore_index=True
                        )[data.columns.tolist()]
                # Drop "minute" column as it will be no longer necessary

        # Rename first column if no name was given in the input file(s)
        data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        data.sort_values(['mz', 'rt'], inplace=True, kind='mergesort')
        data.reset_index(drop=True, inplace=True)
        # Adjust m/z column values to the machine's maximum float
        # resolution
        data['mz'] = data['mz'].apply(round, ndigits=resolution)
        super(LDataFrame, self).__init__(data=data)
        self.src = src
        self._resolution = resolution

    def drop_empty_frames(self, module, parameters, means=False):
        """Remove empty frames from the dataframe and reset the index.
        An empty frame is a row for which every sample replicate or
        sample mean has a zero intensity.
        Keyword Arguments:
            module     -- module name to write in the logging file
            parameters -- LipidFinder's parameters instance (can be for
                          any module)
            means      -- check sample means instead of each sample
                          replicate? [default: False]
        """
        if means:
            meanColIndexes = [i for i, col in enumerate(self.columns)
                              if col.endswith('_mean')]
            if parameters['numSolventReps'] > 0:
                # The first mean column is for the solvents
                firstIndex = meanColIndexes[1]
            else:
                firstIndex = meanColIndexes[0]
            lastIndex = meanColIndexes[-1] + 1
        else:
            firstIndex = parameters['firstSampleIndex'] - 1
            lastIndex = firstIndex + (parameters['numSamples'] * parameters['numTechReps'])
        # Get the indices of all empty frames
        emptyFrames = self.iloc[:, firstIndex: lastIndex].eq(0).all(axis=1)
        indices = self[emptyFrames].index.tolist()
        if indices:
            # Drop empty frames and reset the index
            self.drop(module, labels=indices, axis=0, inplace=True)
            self.reset_index(drop=True, inplace=True)

    @staticmethod
    def _read_file(src):
        extension = os.path.splitext(src)[1].lower()[1:]
        # Load file based on its extension
        if extension == 'tab':
            data = pd.read_csv(src, sep='\t', float_precision='high')
        elif extension == 'tsv':
            data = pd.read_csv(src, sep='\t', float_precision='high')
        else:
            raise IOError(("Unknown file extension '{0}'. Expected: csv, tsv, "
                           "xls, xlsx").format(extension))

        data['rt'] = data['rt'].apply(lambda x: round(x / 60.0, 2))
        return data
