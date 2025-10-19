import os
import threading
from typing import Optional
from lipidomics.LDataFrame import LDataFrame
from lipidomics.utils import extract_zip_files, extraction


class Conversion:

    def __int__(self, srcpath: str, vendor: Optional[str] = None,
                experiment_mode: Optional[str] = None, dda_top: Optional[int] = None,
                instrument: Optional[str] = None):
        self.srcpath = srcpath

        if vendor is None:
            vendor = "thermo"
        self.vendor = vendor

        if experiment_mode is None:
            experiment_mode = "LC-MS"
        self.experiment_mode = experiment_mode

        if instrument is None:
            instrument = "orbitrap"
        self.instrument = instrument

        if dda_top is None:
            dda_top = 5
        self.dda_top = dda_top

    def converter(self):
        raw_files = [os.path.join(self.srcpath, file) for file in os.listdir(self.srcpath) if file.endswith(".mzML")]
        threads = []
        for raw_file in raw_files:
            thread = threading.Thread(target=self.conversion_process,
                                      args=(raw_file, self.srcpath),
                                      daemon=True)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        data = LDataFrame(self.srcpath)
        return data

    def conversion_process(self, raw_file):
        extraction(raw_file, self.srcpath)


