import os
from copy import deepcopy

import numpy as np
from opfunu.cec_based.cec2014 import *


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
    np.savetxt("./ZSO_Uniform_Data/CEC2014/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, SuiteName, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize / 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2014Funcs = [F12014(Dim), F22014(Dim), F32014(Dim), F42014(Dim), F52014(Dim), F62014(Dim), F72014(Dim),
                    F82014(Dim), F92014(Dim), F102014(Dim), F112014(Dim), F122014(Dim), F132014(Dim), F142014(Dim),
                    F152014(Dim), F162014(Dim), F172014(Dim), F182014(Dim), F192014(Dim), F202014(Dim), F212014(Dim),
                    F222014(Dim), F232014(Dim), F242014(Dim), F252014(Dim), F262014(Dim), F272014(Dim), F282014(Dim),
                    F292014(Dim), F302014(Dim)]
    FuncNum = 0
    for i in range(15, len(CEC2014Funcs)):
        FuncNum = i + 1
        RunZSO(CEC2014Funcs[i])


if __name__ == "__main__":
    if os.path.exists('./ZSO_Uniform_Data/CEC2014') == False:
        os.makedirs('./ZSO_Uniform_Data/CEC2014')
    Dims = [50]
    for Dim in Dims:
        main(Dim)
