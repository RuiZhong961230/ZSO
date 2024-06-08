import os
from copy import deepcopy
from enoppy.paper_based.pdo_2022 import *


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize

TrialRuns = 30
MaxFEs = DimSize * 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0
SuiteName = "CEC2014"


# initialize the Pop randomly
def Initialization(func):
    global Pop, FitPop, curFEs, DimSize
    Pop = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func.evaluate(Pop[i])


def Dis(X1, X2):
    global DimSize
    dis = 0
    for i in range(DimSize):
        dis += (X1[i] - X2[i]) ** 2
    return np.sqrt(dis)


def ZSO(func):
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize
    Xbest = Pop[np.argmin(FitPop)]
    FitBest = min(FitPop)
    Xmean = np.mean(Pop, axis=0)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    alpha = np.random.uniform(0.5, 1.5)
    beta = np.random.uniform(0.5, 1.5)
    for i in range(PopSize):
        Off[i] = Pop[i] + alpha * np.random.rand(DimSize) * (Xbest - Pop[i]) + beta * beta * np.random.normal(0, 1, DimSize)
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func.evaluate(Off[i])
        if FitOff[i] < FitPop[i]:
            FitPop[i] = FitOff[i]
            Pop[i] = deepcopy(Off[i])
            if FitOff[i] < FitBest:
                FitBest = FitOff[i]
                Xbest = deepcopy(Off[i])

    for i in range(PopSize):
        Off[i] = Pop[i] + alpha * np.random.rand(DimSize) * (Xmean - Pop[i]) + beta * beta * np.random.normal(0, 1, DimSize)
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func.evaluate(Off[i])
        if FitOff[i] < FitPop[i]:
            FitPop[i] = FitOff[i]
            Pop[i] = deepcopy(Off[i])
            if FitOff[i] < FitBest:
                FitBest = FitOff[i]
                Xbest = deepcopy(Off[i])


def RunZSO(func):
    global curFEs, curIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        Initialization(func)
        Best_list.append(min(FitPop))
        np.random.seed(2022 + 88 * i)
        while curIter <= MaxIter:
            ZSO(func)
            curIter += 1
            Best_list.append(min(FitPop))
            # print("Iter: ", curIter, "Best: ", Fgbest)
        All_Trial_Best.append(Best_list)
    np.savetxt("./ZSO_Uniform_Data/Engineer/" + str(FuncNum) + ".csv", All_Trial_Best, delimiter=",")



def main():
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, SuiteName, LB, UB
    MaxFEs = 20000
    MaxIter = int(MaxFEs / PopSize / 2)
    Probs = [WBP(), PVP(), CSP(), SRD(), TBTD(), GTD(), CBD(), IBD(), TCD(), PLD(), CBHD(), RCB()]
    Names = ["WBP", "PVP", "CSP", "SRD", "TBTD", "GTD", "CBD", "IBD", "TCD", "PLD", "CBHD", "RCB"]

    FuncNum = 0
    for i in range(len(Probs)):
        DimSize = Probs[i].n_dims
        Pop = np.zeros((PopSize, DimSize))
        LB = Probs[i].lb
        UB = Probs[i].ub
        FuncNum = Names[i]
        RunZSO(Probs[i])


if __name__ == "__main__":
    if os.path.exists('./ZSO_Uniform_Data/Engineer') == False:
        os.makedirs('./ZSO_Uniform_Data/Engineer')
    main()
