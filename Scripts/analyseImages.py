import logging
import click
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from src.datalayer.analyseImages import AnalyzeDataSet
from src.models.convNet import Vgg19


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    #Set logger
    logger = logging.getLogger(__name__)


    #Datalayer
    path1 = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\Train\TrainImages'
    getDataSetAttributes = AnalyzeDataSet(path=path1)





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()