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
import torchvision
from torch import nn


@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    #Set logger
    logger = logging.getLogger(__name__)

    #Create model
    batch_Size = 32
    #vggNet = Vgg19(batch_size=batch_Size)
    densenet = torchvision.models.densenet121(pretrained=True)
    densenet.classifier = nn.Linear(1024,29)
    #vggNet = Vgg11(batch_size=batch_Size)
    model = densenet
    #Datalayer
    DatLayer = CreateDataloaders(batch_size=batch_Size)
    trainloader, testLoader, valLoader, classes = DatLayer.getDataloaders()
    path = r'C:\Users\Morten From\PycharmProjects\KaggleChallengeCVML2021\src'

    #Training
    trainingClass = trainInpainting(trainloader,model, path)
    model = trainingClass.traingan()

    #Testing
    testClass = testInpainting(valLoader, model, classes)
    testClass.runTest()

    #Produce test results
    testResultsClass  = testResultsInpainting(testLoader,model,classes)
    testResultsClass.runTest()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()