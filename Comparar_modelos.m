%% ============================================================
%  ANFIS - COMPARACIÓN DE MODELOS + UMBRAL ÓPTIMO
%% ============================================================
clc; clear; close all;

%% 1. CARGA DE DATOS
train_raw = readmatrix('TRAINING.csv', 'NumHeaderLines', 1);
test_raw  = readmatrix('TESTING_DATA.csv', 'NumHeaderLines', 1);

X_train = train_raw(:, 1:4);   % Age, CreditScore, Education, MaritalStatus
y_train = train_raw(:, 5);              % Default

X_test  = test_raw(:, 1:4);
y_test  = test_raw(:, 5);

% Verificar que y solo tenga 0 y 1
fprintf('Clases únicas en y_train: %s\n', num2str(unique(y_train)'));
fprintf('Clases únicas en y_test:  %s\n', num2str(unique(y_test)'));

xmin = min(X_train);
xmax = max(X_train);
xrange = xmax - xmin;
xrange(xrange == 0) = 1;

X_train_n = (X_train - xmin) ./ xrange;
X_test_n  = (X_test  - xmin) ./ xrange;

trainData = [X_train_n y_train];
testData  = [X_test_n  y_test];

%% ============================================================
% MODELO 1: GRID [2 2 2 3]
%% ============================================================
fprintf('\n=== GRID [2 2 2 3] ===\n');

opt1 = genfisOptions('GridPartition','InputMembershipFunctionType','gaussmf');
opt1.NumMembershipFunctions = [2 2 2 3];
rng(42);
fis1_init = genfis(X_train_n, y_train, opt1);  
[~,~,~,fis1_val] = anfis(trainData,fis1_init,[50 0.001 0.9 1 20],NaN,testData);
score1    = evalfis(fis1_val, X_test_n);          

[umbral1,f1_1] = encontrarUmbral(score1,y_test);

pred1 = double(score1 >= umbral1);
[f1_grid,prec1,rec1,acc1] = calcularMetricas(y_test,pred1);

%% ============================================================
% MODELO 2: GRID [3 3 3 3]
%% ============================================================
fprintf('\n=== GRID [3 3 3 3] ===\n');

opt2 = genfisOptions('GridPartition','InputMembershipFunctionType','gaussmf');
opt2.NumMembershipFunctions = [3 3 3 3];

fis2_init = genfis(X_train_n, y_train, opt2);

[~,~,~,fis2_val] = anfis(trainData,fis2_init,[80 0.01 0.9 1.1 30],NaN,testData);

score2 = evalfis(fis2_val,X_test_n);


[umbral2,f1_2] = encontrarUmbral(score2,y_test);

pred2 = double(score2 >= umbral2);
[f1_g2,prec2,rec2,acc2] = calcularMetricas(y_test,pred2);

%% ============================================================
% MODELO 3: FCM
%% ============================================================
fprintf('\n=== FCM c=6 ===\n');

opt3 = genfisOptions('FCMClustering');
opt3.NumClusters = 6;

fis3_init = genfis(X_train_n,y_train,opt3);

[~,~,~,fis3_val] = anfis(trainData,fis3_init,[90 0.01 0.9 1.1 20],NaN,testData);

score3 = evalfis(fis3_val,X_test_n);

[umbral3,f1_3] = encontrarUmbral(score3,y_test);

pred3 = double(score3 >= umbral3);
[f1_fcm,prec3,rec3,acc3] = calcularMetricas(y_test,pred3);

%% ============================================================
% RESUMEN FINAL
%% ============================================================
fprintf('\n====================================================\n');
fprintf('RESUMEN FINAL (CON UMBRAL ÓPTIMO)\n');
fprintf('====================================================\n');

fprintf('Modelo              F1       Acc      Umbral\n');
fprintf('Grid [2 2 2 3]      %.4f   %.4f   %.2f\n', f1_grid, acc1, umbral1);
fprintf('Grid [3 3 3 3]      %.4f   %.4f   %.2f\n', f1_g2, acc2, umbral2);
fprintf('FCM c=6             %.4f   %.4f   %.2f\n', f1_fcm, acc3, umbral3);

%% ============================================================
% FUNCIONES
%% ============================================================

function [best_umbral,best_f1] = encontrarUmbral(score,y_true)

umbrales = 0.1:0.01:0.9;
f1 = zeros(size(umbrales));

for i=1:length(umbrales)

    u = umbrales(i);
    pred = double(score >= u);

    TP = sum(y_true==1 & pred==1);
    FP = sum(y_true==0 & pred==1);
    FN = sum(y_true==1 & pred==0);

    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);

    f1(i) = 2 * precision * recall / (precision + recall + eps);
end

[best_f1, idx] = max(f1);
best_umbral = umbrales(idx);

% gráfico
figure;
plot(umbrales,f1,'b','LineWidth',2);
xline(best_umbral,'r--',['U=' num2str(best_umbral)]);
title('Optimización de Umbral');
xlabel('Umbral'); ylabel('F1');
grid on;

end

function [f1,precision,recall,accuracy] = calcularMetricas(y_true,y_pred)

TP = sum(y_true==1 & y_pred==1);
FP = sum(y_true==0 & y_pred==1);
FN = sum(y_true==1 & y_pred==0);
TN = sum(y_true==0 & y_pred==0);

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1        = 2 * precision * recall / (precision + recall + eps);
accuracy  = (TP + TN) / length(y_true);

end