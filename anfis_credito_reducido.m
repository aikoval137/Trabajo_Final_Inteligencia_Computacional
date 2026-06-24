%% ANFIS COMO CLASIFICADOR - DATASET DE RIESGO DE CRÉDITO
%% 4 ENTRADAS (Age, CreditScore, EmploymentType, Income)
% Salida: Default (0 = cumple, 1 = incumple)
clear; clc; close all;
rng(42);

%% 1) CARGAR LOS DATASETS 
Ttrain = readtable("training_data_reducido.csv");
Ttest  = readtable("testing_data.csv");

disp("Primeras muestras de entrenamiento:");
disp(head(Ttrain(:, ["Age","CreditScore","MaritalStatus", "Education","Default"]), 5));

%% 2) EXTRAER ENTRADAS Y SALIDA (4 ENTRADAS)
% Elegimos estas 4 variables pensando en SESGO, no solo en poder
% predictivo:

varsEntrada = ["Age","CreditScore","MaritalStatus", "Education"];
varSalida   = "Default";

Xtrain = table2array(Ttrain(:, varsEntrada));
Ytrain = Ttrain.(varSalida);

Xtest = table2array(Ttest(:, varsEntrada));
Ytest = Ttest.(varSalida);

%% 3) NORMALIZAR ENTRADAS
mu    = mean(Xtrain);
sigma = std(Xtrain);

Xtrain_n = (Xtrain - mu) ./ sigma;
Xtest_n  = (Xtest  - mu) ./ sigma;

trainData = [Xtrain_n Ytrain];
testData  = [Xtest_n  Ytest];

umbral = 0.5;

%% ============================================================
%% ANFIS 1: GRID PARTITION
%% ============================================================
fprintf("=== Construyendo ANFIS con Grid Partition (16 reglas) ===\n");

optGrid = genfisOptions("GridPartition");
optGrid.NumMembershipFunctions = [2 2 2 3];  
optGrid.InputMembershipFunctionType = "gbellmf";  % base para todas

fisGrid0 = genfis(trainData(:,1:4), trainData(:,5), optGrid);

fisGrid0.Inputs(3).MembershipFunctions(1) = fismf('trimf', [-0.5 0 1.5]);
fisGrid0.Inputs(3).MembershipFunctions(2) = fismf('trimf', [0.5 1.5 2.5]);
fisGrid0.Inputs(3).MembershipFunctions(3) = fismf('trimf', [1.5 3 3.5]);


optAnfisGrid = anfisOptions("InitialFIS", fisGrid0, "EpochNumber", 50, ...
    "ValidationData", testData);
[fisGrid, trnErrorGrid, ~, fisGridBest, chkErrorGrid] = anfis(trainData, optAnfisGrid);

%% Evaluación Grid Partition
Ygrid_cont  = evalfis(fisGrid, testData(:,1:4));
Ygrid_class = double(Ygrid_cont >= umbral);
Yreal       = testData(:,5);

evaluarClasificador(Yreal, Ygrid_class, "ANFIS Grid Partition");

%% ============================================================
%% ANFIS 2: FUZZY C-MEANS
%% ============================================================
fprintf("\n=== Construyendo ANFIS con Fuzzy C-Means ===\n");

optFCM = genfisOptions("FCMClustering");
optFCM.NumClusters = 7;
fisFCM0 = genfis(trainData(:,1:4), trainData(:,5), optFCM);

optAnfisFCM = anfisOptions("InitialFIS", fisFCM0, "EpochNumber", 50, ...
    "ValidationData", testData);
[fisFCM, trnErrorFCM, ~, fisFCMBest, chkErrorFCM] = anfis(trainData, optAnfisFCM);

%% Evaluación FCM
Yfcm_cont  = evalfis(fisFCM, testData(:,1:4));
Yfcm_class = double(Yfcm_cont >= umbral);

evaluarClasificador(Yreal, Yfcm_class, "ANFIS Fuzzy C-Means");

%% ============================================================
%% EVALUACIÓN Y VISUALIZACIÓN
%% ============================================================
evaluarClasificador(Yreal, Ygrid_class, "ANFIS Grid Partition");
figure; 
confusionchart(Yreal, Ygrid_class, 'Title', 'Matriz de Confusión: ANFIS Grid');

evaluarClasificador(Yreal, Yfcm_class, "ANFIS Fuzzy C-Means");
figure; 
confusionchart(Yreal, Yfcm_class, 'Title', 'Matriz de Confusión: ANFIS FCM');

%% ============================================================
%% FUNCIÓN
%% ============================================================
function evaluarClasificador(Yreal, Ypred, nombreModelo)
    Yreal = Yreal(:);
    Ypred = Ypred(:);

    VP = sum(Yreal == 1 & Ypred == 1);
    VN = sum(Yreal == 0 & Ypred == 0);
    FP = sum(Yreal == 0 & Ypred == 1);
    FN = sum(Yreal == 1 & Ypred == 0);
    N  = VP + VN + FP + FN;

    accuracy = (VP + VN) / N;

    if (VP + FP) > 0; precision = VP / (VP + FP); else; precision = 0; end
    if (VP + FN) > 0; recall   = VP / (VP + FN); else; recall    = 0; end
    if (VN + FP) > 0; especificidad = VN / (VN + FP); else; especificidad = 0; end
    if (precision + recall) > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;
    end

    fprintf("\n========================================\n");
    fprintf(" Resultados: %s\n", nombreModelo);
    fprintf("========================================\n");
    fprintf(" Matriz de confusión:\n");
    fprintf("                  Pred. 0    Pred. 1\n");
    fprintf("   Real 0  |      %5d      %5d\n", VN, FP);
    fprintf("   Real 1  |      %5d      %5d\n", FN, VP);
    fprintf("\n");
    fprintf(" Accuracy      : %.4f\n", accuracy);
    fprintf(" Precision     : %.4f\n", precision);
    fprintf(" Recall (TPR)  : %.4f\n", recall);
    fprintf(" Especificidad : %.4f\n", especificidad);
    fprintf(" F1-score      : %.4f\n", f1);
    fprintf("========================================\n\n");
end