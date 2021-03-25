import logging
import click
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from src.datalayer.CreateDataloaders import CreateDataloaders
from src.models.trainModel import trainInpainting
from src.models.testModel import testInpainting
from src.models.produceTestResults import testResultsInpainting
from src.models.convNet import Vgg19, Vgg11
import torch

@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    #Set logger
    logger = logging.getLogger(__name__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    batch_Size = 32

    #Datalayer
    DatLayer = CreateDataloaders(batch_size=batch_Size)
    trainloader, testLoader, valLoader, classes = DatLayer.getDataloaders()

    path = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\src'
    modelPath = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\data\finishedModels\Vgg19.pth'

    model = Vgg19(batch_size=batch_Size)
    model.load_state_dict(torch.load(modelPath))

    vggNet = model.to(device)
    #Produce test results
    testResultsClass  = testResultsInpainting(testLoader,vggNet,classes)
    testResultsClass.runTest()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()