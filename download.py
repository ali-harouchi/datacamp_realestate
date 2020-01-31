from __future__ import print_function
from google_drive_downloader import GoogleDriveDownloader as gdd

import os
from shutil import copyfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


URLBASE = 'https://drive.google.com/uc?export=download&id'
URLS = ['1fyT1h8WzMvzb2r25u2c-iv0UL874QEGe','1urrukqPsHRvMBA0n279SdnluWgBQQ8qg','19cO2mHbdcQlCJgxpDZG7A00hgRt6sntF']
DATA = ['real_estate_train.csv.gz','real_estate_test.csv.gz','dvf-2014-2018.csv.gz']





def main(output_dir='data'):
    filenames = DATA
    urls = URLS
    full_urls = [URLBASE.format(url) for url in URLS]
    
    for url, filename in zip(urls, filenames):
        print("Downloading from {} ...".format(URLBASE+str(url)))
        gdd.download_file_from_google_drive(file_id=url,
                                        dest_path='./data/'+filename)
        print("=> File saved as {}".format(filename))

    # copy awards data to submission file
    if os.path.exists(os.path.join('submissions', 'starting_kit')):
        copyfile(
            os.path.join('data', DATA[0]),
            os.path.join('submissions', 'starting_kit', DATA[0])
        )


if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()