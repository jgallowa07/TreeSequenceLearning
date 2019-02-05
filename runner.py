
from imports import *
from Simulator import *
from TreesDirHelpers import *
from SequenceBatchGenerator import *
from MyNetworks import *
from MyDemographics import *
from plotTestVali import *

ParentDir = "/EqualA"
DataDir = "/data0/jadrion/projects/recombover"

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
    'priorLowsMu':1.25e-8,
    'priorHighsMu':3.75e-8,
    'ChromosomeLength':1e5,
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

maxSegSites = 0
minSegSites = 10000
for ds in [trainDir,valiDir,testDir]:
    DsInfoDir = pickle.load(open(ds+"info.p","rb"))
    segSitesInDs = max(DsInfoDir["numNodes"])
    segSitesInDsMin = min(DsInfoDir["numNodes"])
    maxSegSites = max(maxSegSites,segSitesInDs)
    minSegSites = min(minSegSites,segSitesInDsMin)

print("MaxSegSites:", maxSegSites)
print("MinSegSites:", minSegSites)

sys.exit()

bds_train_params = {
    'treesDirectory':trainDir,
    'batchSize': 64,
    'maxLen': maxSegSites,
    'frameWidth': 5,
    'shuffleInds':True,
    'sortInds':False,
    'center':False,
    'ancVal':-1,
    'padVal':0,
    'derVal':1,
    'realLinePos':True,
    'posPadVal':0,
          }

bds_vali_params = copy.deepcopy(bds_train_params)
bds_vali_params['treesDirectory'] = valiDir
bds_vali_params['batchSize'] = 64

bds_test_params = copy.deepcopy(bds_train_params)
bds_test_params['treesDirectory'] = testDir
bds_test_params['batchSize'] = 1500
bds_test_params['shuffleExamples'] = False

train_sequence = SequenceBatchGenerator(**bds_train_params)
vali_sequence = SequenceBatchGenerator(**bds_vali_params)
test_sequence = SequenceBatchGenerator(**bds_test_params)

exp_name = "GRU_DO_STATEFUL"

resultsFile = "./Results"+ParentDir+"_"+exp_name+".p"
saveas = "./PDFs"+ParentDir+"_"+exp_name+".pdf"

runModels(ModelFuncPointer=GRU_DO_STATEFUL,
        ModelName=exp_name,
        TrainDir=trainDir,
        TrainGenerator=train_sequence,
        ValidationGenerator=vali_sequence,
        TestGenerator=test_sequence,
        resultsFile=resultsFile,
        outputNetwork=None,
        numEpochs=300,
        #epochSteps=epochSteps,
        validationSteps=20,
        gpuID=2)

plotResults(resultsFile=resultsFile,saveas=saveas)


