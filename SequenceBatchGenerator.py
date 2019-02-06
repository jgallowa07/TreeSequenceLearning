'''
Author: Jared Galloway.
'''

from imports import *
from TreesDirHelpers import *
from EncodeTreeSequence import *

class SequenceBatchGenerator(keras.utils.Sequence):

    '''
    This class, SequenceBatchGenerator, extends keras.utils.Sequence.
    So as to multithread the batch preparation in tandum with network training
    for maximum effeciency on the hardware provided.

    This specific batch generator is created for learning on tree sequence visualizations

    It also offers a range of data prepping heuristics as well as normalizing
    the targets.

    def __getitem__(self, idx):

    def __data_generation(self, batchTreeIndices):

    '''

    #Initialize the member variables which largely determine the data prepping heuristics
    #in addition to the .trees directory containing the data from which to generate the batches
    def __init__(self,
            treesDirectory,
            targetNormalization = 'zscore',
            batchSize=64,
            maxLen=None,
            frameWidth=0,
            width = None,
            seperateTimes = False,
            posPadVal = 0,
            shuffleExamples = True,
            ):

        self.treesDirectory = treesDirectory
        self.targetNormalization = targetNormalization
        infoFilename = os.path.join(self.treesDirectory,"info.p")
        self.infoDir = pickle.load(open(infoFilename,"rb"))
        if(targetNormalization != None):
            self.normalizedTargets = self.normalizeTargets()
        self.batch_size = batchSize
        self.maxLen = maxLen
        self.frameWidth = frameWidth
        self.width = width
        self.seperateTimes = seperateTimes
        self.realLinePos = realLinePos
        self.indices = np.arange(self.infoDir["numReps"])
        self.shuffleExamples = shuffleExamples

        if(shuffleExamples):
            np.random.shuffle(self.indices)

    def normalizeTargets(self):

        '''
        We want to normalize all targets.
        '''

        norm = self.targetNormalization
        nTargets = copy.deepcopy(self.infoDir['rho'])
        if(norm == 'zscore'):
            tar_mean = np.mean(nTargets,axis=0)
            tar_sd = np.std(nTargets,axis=0)
            nTargets -= tar_mean
            nTargets = np.divide(nTargets,tar_sd,out=np.zeros_like(nTargets),where=tar_sd!=0)

        return nTargets

    def on_epoch_end(self):

        if(self.shuffleExamples):
            np.random.shuffle(self.indices)

    def __len__(self):

        return int(np.floor(self.infoDir["numReps"]/self.batch_size))

    def __getitem__(self, idx):

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = self.__data_generation(indices)
        return X,y

    def __data_generation(self, batchTreeIndices):

        respectiveNormalizedTargets = [[t] for t in self.normalizedTargets[batchTreeIndices]]
        targets = np.array(respectiveNormalizedTargets)

        Encodings = []
        Times = []

        for treeIndex in batchTreeIndices:
            treeFilepath = os.path.join(self.treesDirectory,str(treeIndex) + ".trees")
            treeSequence = msp.load(treeFilepath)
            dts = DiscretiseTreeSequence(ts=treeSequence)
            Encodings.append(EncodeTree_F64(ts=dts,width=self.width).astype(np.uint8))

        if(self.seperateTimes)
            Times.append(np.array([node.time for node in treeSequence.nodes()],dtype='float32'))
            return [Encodings.Times],targets
        else:
            return Encodings,targets
                        




