import os
from copy import deepcopy
from opfunu.cec_based.cec2022 import *


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
    alpha = np.random.normal(1, 0.5)
    beta = np.random.normal(1, 0.5)
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
    np.savetxt("./ZSO_Gauss_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")



def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, SuiteName, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize / 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(DimSize), F22022(DimSize), F32022(DimSize), F42022(DimSize), F52022(DimSize), F62022(DimSize),
               F72022(DimSize), F82022(DimSize), F92022(DimSize), F102022(DimSize), F112022(DimSize), F122022(DimSize)]

    FuncNum = 0
    for i in range(len(CEC2022)):
        FuncNum = i + 1
        RunZSO(CEC2022[i])


if __name__ == "__main__":
    if os.path.exists('./ZSO_Gauss_Data/CEC2022') == False:
        os.makedirs('./ZSO_Gauss_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
