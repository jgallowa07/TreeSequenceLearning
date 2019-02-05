from imports import *

plt.rc('font', family='serif', serif='Times')
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.rc('axes', labelsize=6)

rf = sys.argv[1]
sa = sys.argv[2]


def mae(x,y):
    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += abs(x[i] - y[i])
    return summ/length

def mse(x,y):
    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += (x[i] - y[i])**2
    return summ/length

def unNormalize(mean,sd,data):
    data *= sd
    data += mean
    return data 

def plotResults(resultsFile,saveas,un=None):

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(2,1)
    #plt.subplots_adjust(hspace=0.5)
    
    predictions = np.array([float(Y) for Y in results["predictions"]])
    realValues = np.array([float(X) for X in results["Y_test"]])
    
    if(un!=None):
        predictions=unNormalize(un[0],un[1],predictions)
        realValues=unNormalize(un[0],un[1],realValues)

    r_2 = round((np.corrcoef(predictions,realValues)[0,1])**2,5)
    
    mae_0 = mae(realValues,predictions)
    mse_0 = mse(realValues,predictions)
    labels = "$R^{2} = $"+str(r_2)+"\n"+"$mae = $" + str(mae_0)+" | "+"$mse = $" + str(mse_0)
    
    axes[0].scatter(realValues,predictions,marker = "o", color = 'tab:purple',s=5.0,alpha=0.6)
    #axes[0].scatter(realValues,predictions)
    axes[0].plot(realValues,predictions,color='none')
    axes[0].relim()
    axes[0].autoscale_view()

    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),  # min of both axes
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),  # max of both axes
    ]
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)
    

    lossRowIndex = 1
    axes[1].plot(results["loss"],label = "mae loss",color='tab:cyan')
    axes[1].plot(results["val_loss"], label= "mae validation loss",color='tab:pink')

#    axes[1].plot(results["mean_squared_error"],label = "mse loss",color='tab:green')
#    axes[1].plot(results["val_mean_squared_error"], label= "mse validation loss",color='tab:olive')

    axes[1].legend(frameon = False,fontsize = 6)
    axes[1].set_ylabel("mse")
        
    axes[0].set_ylabel(str(len(predictions))+" msprime predictions")
    axes[0].set_xlabel(str(len(realValues))+" msprime real values")
    fig.subplots_adjust(left=.15, bottom=.16, right=.85, top=.92,hspace = 0.5,wspace=0.4)
    height = 7.00
    width = 7.00

    axes[0].grid()
    fig.set_size_inches(height, width)
    fig.savefig(saveas)

if(len(sys.argv) <= 3):
    plotResults(resultsFile = rf,saveas = sa)
else:
    plotResults(resultsFile = rf,saveas = sa,un=[float(sys.argv[3]),float(sys.argv[4])])
