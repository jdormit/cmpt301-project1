'''
Created on Aug 28, 2016

@author: km_dh
'''
import DT as dt
import KNN as knn
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import MNISTcontrol as ms
import TrainTest as tt
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import Tkinter as tk
import os.path


class Controller(object):
    pass

def dispBanner(stage):
    if stage == 'MNIST':
        print '************************************************************************'
        print '**  MNIST DIGITS EXPERIMENTS'
        print '************************************************************************'
            
        print 'First we load the data '
        dirname = os.path.dirname(__file__)
        dataset = ms.MNISTcontrol(os.path.join(dirname, "..", "MNIST"))
        trX_images, trY = dataset.load_mnist('training')
        # we need x to be 1d
        sizeX = trX_images.shape
        if len(sizeX) > 2:
            newXdim = 0
            for i in range(sizeX[1]):
                newXdim += len(trX_images[0][i])
            trX = np.reshape(trX_images, (sizeX[0], newXdim))
            #read in test data
        deX, deY = dataset.load_mnist('test')
        # we need x to be 1d
        sizeX = deX.shape
        if len(sizeX) > 2:
            newXdim = 0
            for i in range(sizeX[1]):
                newXdim += len(deX[0][i])
            deX = np.reshape(deX, (sizeX[0], newXdim))
       
        return trX_images, trX, trY, deX, deY
    print'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    print'********************************************************************'
    print'** 20 NEWSGROUPS EXPERIMENTS'
    print'********************************************************************\n'
    print 'First we load the data \n'
    cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
    news_train = fetch_20newsgroups(subset='train', categories=cats, shuffle=True, random_state=42)
    news_test = fetch_20newsgroups(subset='test', categories=cats, shuffle=True, random_state=42)
    for i in range(len(news_train.target_names)):
        print "Target number " + str(i) + " is " + news_train.target_names[i]
    vectorizer = TfidfVectorizer()
    trX = vectorizer.fit_transform(news_train.data) 
#    trX = trX.toarray() 
    trX_images = news_train.data
    trY = news_train.target
    trY = np.reshape(trY, (trY.shape[0],1))
    deX = vectorizer.fit_transform(news_test.data)
#    deX = deX.toarray()
    deY = news_test.target
    deY = np.reshape(deY, (deY.shape[0],1))
    raw_input('\nPress enter to continue...')
    return trX_images, trX, trY, deX,deY


def dispImages(title,trX_images,trY):         
    print 'and display some examples from it....'
    print '\nClose the image to continue...'
    
    if 'MNIST' in title:
        fig = plt.figure()
        for i in range(8):
            a = fig.add_subplot(2,4,i+1)
            plt.imshow(trX_images[i*200], cmap=plt.cm.get_cmap(plt.gray()))
            a.set_title('Image Label: ' + str(trY[i*200]))
            plt.savefig('../figure1.png')
        plt.show()
    else:
        print "You may have to look in your dock for a microphone-ish icon to find the document"
        print "Close each document to see the next"
        mlp.use('TkAgg')
        for i in range(4):
            ans = "Document Label: " + str(trY[200*i]) + "\n\n"
            label = tk.Label(None, text=ans + str(trX_images[200*i]), justify='left', font=('Times', '12'),fg='black')
            label.pack()
            label.focus()
            label.mainloop()
    
    raw_input('Press enter to continue...')
 
def run_test(trX, trY,res_file):
    desired_dt20 = 0.78
    desired_dt50 = 0.78
    desired_knn1 = 0.70
    desired_knn3 = 0.73
    
    print '\n\nFirst, we run DT and KNN on the training/development data to '
    print 'ensure that we are getting roughly the right accuracies.'
    print 'We use the first 80% of the data as training, and the last'
    print '20% as test.'
    
    
    decTree = dt.DT()
    res = 1

    print '\nDT (cutoff=20)...'
    sizeX = trX.shape
    end = int(np.round(sizeX[0]*0.80,decimals=0))
    testRun = tt.TrainTest(decTree, trX[:end, :], trY[:end], trX[end:, :], trY[end:], 20)
    acc = testRun.run_tt()
    res += testRun.verifyAcc(acc['acc'], desired_dt20)
    print'\nTrainTime, TestTime', acc['trainTime'], acc['testTime']
    res_file.write('\nDT (cutoff=20)')
    res_file.write('\nTrainTime, TestTime ' + str(acc['trainTime']) + ', ' + str(acc['testTime']))
 
    print '\nDT (cutoff=50)...'
    testRun = tt.TrainTest(decTree, trX[:end, :], trY[:end], trX[end:sizeX[0], :], trY[end:sizeX[0]], 50)
    acc = testRun.run_tt()
    res += testRun.verifyAcc(acc['acc'], desired_dt50)
    print'\nTrainTime, TestTime', acc['trainTime'], acc['testTime']
    res_file.write('\nDT (cutoff=50)')
    res_file.write('\nTrainTime, TestTime ' + str(acc['trainTime']) + ', ' + str(acc['testTime']))
    
    knnModel = knn.KNN()
    print '\nKNN (K=1)'
    max_size = sizeX[0] if sizeX[0] < 10001 else 10000
    end = int(np.round(max_size*0.80,decimals=0)) 
    testRun = tt.TrainTest(knnModel, trX[:end, :], trY[:end], trX[end:sizeX[0], :], trY[end:sizeX[0]], 1)
    acc = testRun.run_tt()
    res += testRun.verifyAcc(acc['acc'], desired_knn1)
    print'\nTrainTime, TestTime', acc['trainTime'], acc['testTime']
    res_file.write('\nKNN (K=1)')
    res_file.write('\nTrainTime, TestTime ' + str(acc['trainTime']) + ', ' + str(acc['testTime']))
 
    print '\nKNN (K=3)'
    testRun = tt.TrainTest(knnModel, trX[:end, :], trY[:end], trX[end:sizeX[0], :], trY[end:sizeX[0]], 3)
    acc = testRun.run_tt()
    res += testRun.verifyAcc(acc['acc'], desired_knn3)
    print'\nTrainTime, TestTime', acc['trainTime'], acc['testTime']
    res_file.write('\nKNN (K=3)')
    res_file.write('\nTrainTime, TestTime ' + str(acc['trainTime']) + ', ' + str(acc['testTime']))

    raw_input('\nPress enter to continue...')
    
    return

def run_comps(learn,thresh,X,Y,X_test,Y_test,title,xlbl,figno):
    accTr = [] 
    accTe = []
    for i in range(len(thresh)):
        print '\n', xlbl, thresh[i]
        testRun = tt.TrainTest(learn, X, Y, X_test, Y_test, thresh[i])
        test_info = testRun.run_tt()
        accTe.append(test_info['acc'])
        model_pred = learn.res('predict',test_info['model'],X)
        val = [Y == model_pred]
        val_sum = sum(sum(val))
        print 'accTr = ', float(val_sum)/(len(Y)*1.0)
        accTr.append(float(val_sum)/(len(Y)*1.0))

    plt.title(title)
    plt.plot(thresh, accTr, color='blue', marker='x', label='training accuracy')
    plt.plot(thresh, accTe, color='black', marker='o', label='test accuracy')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(xlbl)
    plt.savefig(figno)
    plt.show()
    return accTe

def compareModels(label1,label2,devX,devY,startno):
    yDT = eval(results[label1])
    matchDT = np.array([yDT == devY])
    errDT = np.invert(matchDT)
    errDT = np.reshape(errDT,(errDT.shape[1],1))
    yKNN = eval(results[label2])
    errKNN = np.invert([yKNN == devY])
    errKNN = np.reshape(errKNN,(errKNN.shape[1],1))
    errs = [[],[],[]]
    for i in range(errDT.shape[0]):
        if errDT[i] and errKNN[i]:
            errs[0].append(devX[i,:,:])
        if errDT[i] and not errKNN[i]:
            errs[1].append(devX[i,:,:])
        if not errDT[i] and errKNN[i]:
            errs[2].append(devX[i,:,:])
    errs = np.array(errs)
    titles = ['both','DT only','KNN only']
    for i in range(len(errs)):
        curr_errs = np.array(errs[i])
        if curr_errs.shape[0] > 40:
            curr_errs = curr_errs[np.random.permutation(curr_errs.shape[0]),:,:]
            curr_errs = curr_errs[0:40,:,:]
        displayErrors(curr_errs,"Figure " + str(startno+i) + 
                      ": Errors made by " + titles[i] + " models","../figure" + 
                      str(startno+i) + ".png")
    return 'done'

def displayErrors(errorList,title,figno):
    print 'errorList', errorList.shape
    if 'MNIST' in title:
        fig = plt.figure()
        plt.suptitle(title)
        for i in range(errorList.shape[0]):
            a = fig.add_subplot(5,8,i+1)
            plt.imshow(errorList[i], cmap=plt.cm.get_cmap(plt.gray()))
            a.set_title('Image ' + str(i))
        plt.savefig(figno)
        plt.show()
    else:
        print "You may have to look in your dock for a microphone-ish icon to find the document"
        print "Close each document to see the next"
        mlp.use('TkAgg')
        for i in range(4):
            ans = "Document Label: " + str(trY[200*i]) + "\n\n"
            label = tk.Label(None, text=ans + str(trX_images[200*i]), justify='left', font=('Times', '12'),fg='black')
            label.pack()
            label.focus()
            label.mainloop()
     
    raw_input('Press enter to continue...')
    return 0
 
   
if __name__ == '__main__':
        
    results = {}
    #Load the results.txt file to start where you stopped
    if os.path.exists("../results.txt"):
        res_file = open('../results.txt','r')
        content = res_file.readlines()
        res_file.close()
        print 'Found results.txt length =', len(content)
        i = 0
        while i < len(content)-1:
            if content[i] == '' or content[i].isspace():
                i+=1
                continue
            results[content[i].strip()] = content[i+1].strip()
            i+=2
        print 'results keys', results.keys()
        
    
    #Set up the infrastructure
    res_file = open('../results.txt','a+') 
    trX_images=[]
    trX=[]
    trY=[]
    deX=[]
    deY=[]
    data_types = ['MNIST','20ng']
    
    for i in range(len(data_types)):
        fin = 'finished' + data_types[i]
        if fin not in results.keys():   
            trX_images, trX, trY, deX, deY = dispBanner(data_types[i])
        
        disp = 'disp'+data_types[i]
        if disp not in results.keys():
            dispImages(data_types[i],trX_images,trY)
            raw_input('Press enter to continue...')
            res_file.write('\n' + disp + '\ndone')


        base = 'baseline'+data_types[i]
        if base not in results.keys():
            print "Lets run some baseline measures..."
            res = run_test(trX,trY,res_file)
            res_file.write('\n' + base + '\n')
            res_file.write(str(res)) 
            raw_input('Press enter to continue...')
 
        dec = 'dt'+data_types[i]
        if dec not in results.keys():
            print '\nNow we vary the cutoff for the decision tree and see how it affects accuracy...'
            thresh = [5,10,20,40,80,160]
            decTree = dt.DT()  
            res = run_comps(decTree, thresh, trX[0:4800, :], trY[0:4800], trX[4801:6000, :], 
                        trY[4801:6000],"Figure 2: DT cutoff versus accuracy (MNIST)","DT cutoff","../figure2.png")
            results[dec] = res
            res_file.write('\n' + dec + '\n') 
            res_file.write(str(res))
            raw_input('Press enter to continue...')
     
        neigh = 'knn'+data_types[i]
        if neigh not in results.keys():
            print '\nNow we vary the k for the KNN classifier and see how it affects accuracy...'
            allK = [1,8,16,32,64,128]
            knnModel = knn.KNN()
            res = run_comps(knnModel, allK, trX[0:2000, :], trY[0:2000], trX[2001:2501, :], 
                         trY[2001:2501],"Figure 3: KNN count versus accuracy (MNIST)","KNN count","../figure3.png")
            results[neigh] = res
            res_file.write('\n' + neigh + '\n')
            res_file.write(str(res))
            raw_input('Press enter to continue...')
               
        heldDT = 'hoDT_'+data_types[i]
        if heldDT not in results.keys():
            print '\nNow we make predictions on dev and test data using the best DT'
            thresh = [5,10,20,40,80,160]
            dtres = 'dt' + data_types[i]
            dtAccs = results[dtres]
            bestDT = np.argmax(dtAccs)
            decTree = dt.DT()
            model = decTree.res('train', trX[0:4800, :], trY[0:4800], trX[4801:6000, :], trY[4801:6000],thresh[bestDT])
            res = decTree.res('predict',model,trX[6001:12001])
            res_file.write('\n' + heldDT + '\ndone')
            res_file.write('\n' + heldDT + 'dev\n')
            devName = heldDT + 'dev'
            results[devName] = res
            res_file.write(str(res))
            res = decTree.res('predict',model,deX)
            res_file.write('\n' + heldDT + 'test\n')
            res_file.write(str(res))
            print 'Tests finished!'
            raw_input('Press enter to continue...')
        
        heldKNN = 'hoKNN_'+data_types[i]
        if heldKNN not in results.keys():
            print '\nNow we make predictions on dev and test data using the best KNN'
            allK = [1,8,16,32,64,128]
            knnres = 'knn' + data_types[i]
            knnAccs = results[knnres]
            bestKNN = np.argmax(knnAccs)
            knnModel = knn.KNN()
            model = knnModel.res('train', trX[0:4800, :], trY[0:4800], trX[4801:6000, :], trY[4801:6000],allK[bestKNN])
            res = knnModel.res('predict',model,trX[6001:12001])
            res_file.write('\n' + heldKNN + '\ndone')
            res_file.write('\n' + heldKNN + 'dev\n')
            devName = heldDT + 'dev'
            results[devName] = res
            res_file.write(str(res))
            res = knnModel.res('predict',model,deX)
            res_file.write('\n' + heldKNN + 'test\n')
            res_file.write(str(res))
            print 'Tests finished!'
            raw_input('Press enter to continue...')
      
        comp = 'compare'+data_types[i]
        if comp not in results.keys():
            print '\nNow we look at errors made by the two models. Remember to wait for the image to change'
            devX = trX_images[6001:12001,:,:]
            devY = trY[6001:12001]
            dtres = 'hoDT_' + data_types[i] + 'dev'
            knnres = 'hoKNN_' + data_types[i] + 'dev'
            res = compareModels(dtres,knnres,devX,devY,4*(i+1))
            res_file.write('\n' + comp + '\n')
            res_file.write(str(res))
        
        res_file.write('\nfinished' + data_types[i] + '\nDone')
    
    res_file.close()
     
    print 'Done' 

    
