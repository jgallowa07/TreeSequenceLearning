
'''
Authors: Jared Galloway, Jeff Adrion 

A collection of helper functions for a .trees directory
'''

from imports import *

def zscoreTargets(self):
    norm = self.targetNormalization
    nTargets = copy.deepcopy(self.infoDir['y'])
    if(norm == 'zscore'):
        tar_mean = np.mean(nTargets,axis=0)
        tar_sd = np.std(nTargets,axis=0)
        nTargets -= tar_mean
        nTargets = np.divide(nTargets,tar_sd,out=np.zeros_like(nTargets),where=tar_sd!=0)


#-------------------------------------------------------------------------------------------

def runModels(ModelFuncPointer,
            ModelName,
            TrainDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            outputNetwork=None,
            gpuID = 0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    if(resultsFile == None):

        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    x,y = TrainGenerator.__getitem__(0)
    model = ModelFuncPointer(x,y)

    callbacks = [EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5)]

    history = model.fit_generator(TrainGenerator,
        steps_per_epoch= epochSteps,
        epochs=numEpochs,
        validation_data=ValidationGenerator,
        validation_steps=validationSteps,
        #callbacks=callbacks,
        use_multiprocessing=True,
        workers = 4
        )

    x,y = TestGenerator.__getitem__(0)
    predictions = model.predict(x)

    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(predictions)
    history.history['Y_test'] = np.array(y)
    history.history['name'] = ModelName

    if(outputNetwork != None):
        model.save(outputNetwork)

    print("results written to: ",resultsFile)
    pickle.dump(history.history, open( resultsFile, "wb" ))

    return None

#-------------------------------------------------------------------------------------------

def indicesGenerator(batchSize,numReps):
    '''
    Generate indices randomly from range (0,numReps) in batches of size batchSize
    without replacement.

    This is for the batch generator to randomly choose trees from a directory
    but make sure
    '''
    availableIndices = np.arange(numReps)
    np.random.shuffle(availableIndices)
    ci = 0
    while 1:
        if((ci+batchSize) > numReps):
            ci = 0
            np.random.shuffle(availableIndices)
        batchIndices = availableIndices[ci:ci+batchSize]
        ci = ci+batchSize

        yield batchIndices

#-------------------------------------------------------------------------------------------

def getHapsPosLabels(direc,simulator,shuffle=False):
    '''
    loops through a trees directory created by the data generator class
    and returns the repsective genotype matrices, positions, and labels
    '''
    haps = []
    positions = []
    infoFilename = os.path.join(direc,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))
    labels = infoDict["y"]

    #how many trees files are in this directory.
    li = os.listdir(direc)
    numReps = len(li) - 1   #minus one for the 'info.p' file

    if(simulator=='msprime'):
        for i in range(numReps):
            filename = str(i) + ".trees"
            filepath = os.path.join(direc,filename)
            treeSequence = msp.load(filepath)
            haps.append(treeSequence.genotype_matrix())
            positions.append(np.array([s.position for s in treeSequence.sites()]))

    else:
        for i in range(numReps):
            filename = str(i) + ".trees"
            filepath = os.path.join(direc,filename)
            treeSequence = pyslim.load(filepath)
            haps.append(treeSequence.genotype_matrix())
            positions.append(np.array([s.position for s in treeSequence.sites()]))

    haps = np.array(haps)
    positions = np.array(positions)

    return haps,positions,labels

#-------------------------------------------------------------------------------------------

def simplifyTreeSequenceOnSubSampleSet_stub(ts,numSamples):
    '''
    This function should take in a tree sequence, generate
    a subset the size of numSamples, and return the tree sequence simplified on
    that subset of individuals
    '''

    ts = ts.simplify() #is this neccessary
    inds = [ind.id for ind in ts.individuals()]
    sample_subset = np.sort(np.random.choice(inds,sample_size,replace=False))
    sample_nodes = []
    for i in sample_subset:
        ind = ts.individual(i)
        sample_nodes.append(ind.nodes[0])
        sample_nodes.append(ind.nodes[1])

    ts = ts.simplify(sample_nodes)

    return ts

#-------------------------------------------------------------------------------------------

def shuffleIndividuals(x):
    t = np.arange(x.shape[1])
    np.random.shuffle(t)
    return x[:,t]

#-------------------------------------------------------------------------------------------

def sort_min_diff(amat):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''

    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

#-------------------------------------------------------------------------------------------

def pad_HapsPos(haplotypes,positions,maxSNPs=None,frameWidth=0,center=False):
    '''
    pads the haplotype and positions tensors
    to be uniform with the largest tensor
    '''

    haps = haplotypes
    pos = positions

    #Normalize the shape of all haplotype vectors with padding
    for i in range(len(haps)):
        numSNPs = haps[i].shape[0]
        paddingLen = maxSNPs - numSNPs
        if(center):
            prior = paddingLen // 2
            post = paddingLen - prior
            haps[i] = np.pad(haps[i],((prior,post),(0,0)),"constant",constant_values=2.0)
            pos[i] = np.pad(pos[i],(prior,post),"constant",constant_values=-1.0)

        else:
            haps[i] = np.pad(haps[i],((0,paddingLen),(0,0)),"constant",constant_values=2.0)
            pos[i] = np.pad(pos[i],(0,paddingLen),"constant",constant_values=-1.0)

    haps = np.array(haps,dtype='float32')
    pos = np.array(pos,dtype='float32')

    if(frameWidth):
        fw = frameWidth
        haps = np.pad(haps,((0,0),(fw,fw),(fw,fw)),"constant",constant_values=2.0)
        pos = np.pad(pos,((0,0),(fw,fw)),"constant",constant_values=-1.0)

    return haps,pos

#-------------------------------------------------------------------------------------------


def mutateTrees(treesDirec,outputDirec,muLow,muHigh,numMutsPerTree=1,simulator="msprime"):
    '''
    read in .trees files from treesDirec, mutate that tree numMuts seperate times
    using a mutation rate pulled from a uniform dirstribution between muLow and muHigh

    also, re-write the labels file to reflect.
    '''
    if(numMutsPerTree > 1):
        assert(treesDirec != outputDirec)

    if not os.path.exists(outputDirec):
        print("directory '",outputDirec,"' does not exist, creating it")
        os.makedirs(outputDirec)

    infoFilename = os.path.join(treesDirec,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))
    labels = infoDict["y"]

    newLabels = []
    newMaxSegSites = 0

    #how many trees files are in this directory.
    li = os.listdir(treesDirec)
    numReps = len(li) - 1   #minus one for the 'labels.txt' file

    if(simulator=='msprime'):
        for i in range(numReps):
            filename = str(i) + ".trees"
            filepath = os.path.join(treesDirec,filename)
            treeSequence = msp.load(filepath)
            blankTreeSequence = msp.mutate(treeSequence,0)
            rho = labels[i]
            for mut in range(numMuts):
                simNum = (i*numMuts) + mut
                simFileName = os.path.join(outputDirec,str(simNum)+".trees")
                mutationRate = np.random.uniform(muLow,muHigh)
                mutatedTreeSequence = msp.mutate(blankTreeSequence,mutationRate)
                mutatedTreeSequence.dump(simFileName)
                newMaxSegSites = max(newMaxSegSites,mutatedTreeSequence.num_sites)
                newLabels.append(rho)

    else:
        for i in range(numReps):
            filename = str(i) + ".trees"
            filepath = os.path.join(treesDirec,filename)
            treeSequence = pyslim.load(filepath)
            blankTreeSequence = msp.mutate(treeSequence,0)
            rho = labels[i]
            for mut in range(numMuts):
                simNum = (i*numMuts) + mut
                simFileName = os.path.join(outputDirec,str(simNum)+".trees")
                mutationRate = np.random.uniform(muLow,muHigh)
                mutatedTreeSequence = msp.mutate(blankTreeSequence,mutationRate)
                mutatedTreeSequence.dump(simFileName)
                newMaxSegSites = max(newMaxSegSites,mutatedTreeSequence.num_sites)
                newLabels.append(rho)

    infoCopy = copy.deepcopy(infoDict)
    infoCopy["maxSegSites"] = newMaxSeqSites
    if(numMutsPerTree > 1):
        infoCopy["y"] = np.array(newLabels,dtype="float32")
        infoCopy["numReps"] = numReps * numMuts
    outInfoFilename = os.path.join(outputDirec,"info.p")
    pickle.dump(infocopy,open(outInfoFilename,"wb"))

    return None

#-------------------------------------------------------------------------------------------

def printDirInfo(treesDirec):
    "print out the info nicely"
    infoFilename = os.path.join(treesDirec,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))

    table = pt()
    keys = []
    values = []
    for key in infoDict:
        if(key == "y"):
            keys.append("num targets")
            values.append(len(infoDict[key]))
        elif(key=="MspDemographics"):
            keys.append(key)
            values.append(str(infoDict[key]))
#        elif(key == "RecombinationRate"):
#            continue
#        elif(key == "MutationRate"):
#            continue
        else:
            keys.append(key)
            values.append(infoDict[key])

    table.field_names = keys
    table.add_row(values)
    print(table)
    return None


def segSitesStats(treesDirec):

    infoFilename = os.path.join(treesDirec,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))

    newLabels = []
    newMaxSegSites = 0

    #how many trees files are in this directory.
    li = os.listdir(treesDirec)
    numReps = len(li) - 1   #minus one for the 'labels.txt' file

    segSites = []

    for i in range(numReps):
        filename = str(i) + ".trees"
        filepath = os.path.join(treesDirec,filename)
        treeSequence = msp.load(filepath)
        segSites.append(treeSequence.num_sites)

    return segSites






