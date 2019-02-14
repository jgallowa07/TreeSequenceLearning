
from imports import *
from Simulator import *
from TreesDirHelpers import *
from SequenceBatchGenerator import *
from MyNetworks import *

ParentDir = "EqualForViz"
#ParentDir = "MatchParameters"
DataDir = "/data0/jgallowa/"

#-----------

trainDir = DataDir + ParentDir + "/train/"
valiDir = DataDir + ParentDir + "/vali/"
testDir = DataDir + ParentDir + "/test/"

#Parameters for simulating.
dg_params_large = {'N': 50,
    'Ne_D': 1e3,
    'Ne_E': 1e4,
    'priorLowsRho':1e-10,
    'priorHighsRho':1e-6,
    'priorLowsMu':0,
    'priorHighsMu':0,
    'ChromosomeLength':1e4,
          }

'''
dg_params_large = {'N': 50,
    'Ne_E': 1e4,
    'priorLowsRho':2.5e-11,
    'priorHighsRho':2.5e-8,
    'priorLowsMu':9e-9,
    'priorHighsMu':3e-8,
    'ChromosomeLength':1e5
       }

#instanciate three simulators all with the same params

dg_train = Simulator(**dg_params_large)
dg_vali = Simulator(**dg_params_large)
dg_test = Simulator(**dg_params_large)

CPU_Count = mp.cpu_count()

print("simulating")
dg_train.simulateAndProduceTrees(numReps=100000,direc=trainDir,simulator="msprime",nProc=CPU_Count)
dg_vali.simulateAndProduceTrees(numReps=1500,direc=valiDir,simulator="msprime",nProc=CPU_Count)
dg_test.simulateAndProduceTrees(numReps=1500,direc=testDir,simulator="msprime",nProc=CPU_Count)
print("DONE SIMULATING")

sys.exit()
'''
maxSegSites = 0
minSegSites = 10000
for ds in [trainDir,valiDir,testDir]:
    DsInfoDir = pickle.load(open(ds+"info.p","rb"))
    segSitesInDs = max(DsInfoDir["numNodes"])
    segSitesInDsMin = min(DsInfoDir["numNodes"])
    maxSegSites = max(maxSegSites,segSitesInDs)
    minSegSites = min(minSegSites,segSitesInDsMin)

print("MaxNumNodes:", maxSegSites)
print("MinNumNodes:", minSegSites)

bds_train_params = {
    'treesDirectory':trainDir,
    'targetNormalization':'zscore',
    'batchSize':16,
    'maxLen':maxSegSites,
    'frameWidth':0,
    'width':None,
    'seperateTimes':False,
    'posPadVal':0,
    'width':100
        }

bds_vali_params = copy.deepcopy(bds_train_params)
bds_vali_params['treesDirectory'] = valiDir
bds_vali_params['batchSize'] = 16

bds_test_params = copy.deepcopy(bds_train_params)
bds_test_params['treesDirectory'] = testDir
bds_test_params['batchSize'] = 1000
bds_test_params['shuffleExamples'] = False

train_sequence = SequenceBatchGenerator(**bds_train_params)
vali_sequence = SequenceBatchGenerator(**bds_vali_params)
test_sequence = SequenceBatchGenerator(**bds_test_params)


#oneBatch = train_sequence.__getitem__(0)
#print(oneBatch[0].shape)

#sys.exit()

#model = CNN1D(oneBatch[0],oneBatch[1])



exp_name = "TestSequential_b16"
resultsDir = DataDir + ParentDir + "/Results/"

resultsFile = resultsDir+exp_name+".p"
saveas = resultsDir+exp_name+".pdf"
#outputNetwork = resultsDir+exp_name+".json"
#outputWeights = resultsDir+exp_name+".h5" 

runModels(ModelFuncPointer=CNN2D_S,
        ModelName=exp_name,
        TrainDir=trainDir,
        TrainGenerator=train_sequence,
        ValidationGenerator=vali_sequence,
        TestGenerator=test_sequence,
        resultsFile=resultsFile,
        outputNetwork=None,
        numEpochs=100,
        #epochSteps=epochSteps,
        validationSteps=20,
        gpuID=0)

plotResults(resultsFile=resultsFile,saveas=saveas)


