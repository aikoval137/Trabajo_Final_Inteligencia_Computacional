%% ============================================================
% ANFIS FINAL GRID 2-2-2-3 vs FCM c=6
% Incluye Matrices de Confusión, Explicabilidad y Escenarios
%% ============================================================
clc; clear; close all;

thr_grid = 0.62; % Umbral ajustado
thr_fcm = 0.58;

%% 1. DATA Y NORMALIZACIÓN
train_raw = readmatrix('TRAINING.csv','NumHeaderLines',1);
test_raw  = readmatrix('TESTING_DATA.csv','NumHeaderLines',1);

X_train = fillmissing(train_raw(:,1:4),'constant',0);
y_train = train_raw(:,5);
X_test  = fillmissing(test_raw(:, 1:4),'constant',0);
y_test  = test_raw(:, 5);

xmin = min(X_train); xmax = max(X_train);
range = xmax - xmin; range(range==0)=1;
X_train_n = (X_train - xmin)./range;
X_test_n  = (X_test  - xmin)./range;
trainData = [X_train_n y_train]; testData = [X_test_n y_test];

%% 2. MODELOS
opt1 = genfisOptions('GridPartition','InputMembershipFunctionType','gbellmf');
opt1.NumMembershipFunctions = [2 2 2 3];
fis_grid = anfis(trainData, genfis(X_train_n, y_train, opt1), [50 0.001 0.9 1.0 20], NaN, testData);

opt2 = genfisOptions('FCMClustering');
opt2.NumClusters = 6;
fis_fcm = anfis(trainData, genfis(X_train_n, y_train, opt2), [90 0.01 0.9 1.1 20], NaN, testData);

%% 3. MATRICES DE CONFUSIÓN
pred_grid = double(evalfis(fis_grid, X_test_n) >= thr_grid);
pred_fcm  = double(evalfis(fis_fcm,  X_test_n) >= thr_fcm);

figure('Name', 'Matrices de Confusión');
subplot(1,2,1); confusionchart(y_test, pred_grid, 'Title', 'Confusion Matrix: GRID');
subplot(1,2,2); confusionchart(y_test, pred_fcm, 'Title', 'Confusion Matrix: FCM');

%% 4. COMPARATIVA DE ESCENARIOS DETALLADA
compareProfiles(fis_grid, fis_fcm, xmin, xmax, thr_grid, thr_fcm);

%% 5. MÉTRICAS

[acc_g,prec_g,rec_g,f1_g] = metricas(y_test,pred_grid);
[acc_f,prec_f,rec_f,f1_f] = metricas(y_test,pred_fcm);

fprintf('\n==============================\n');
fprintf('GRID\n');
fprintf('Accuracy  = %.4f\n',acc_g);
fprintf('Precision = %.4f\n',prec_g);
fprintf('Recall    = %.4f\n',rec_g);
fprintf('F1-score  = %.4f\n',f1_g);

fprintf('\nFCM\n');
fprintf('Accuracy  = %.4f\n',acc_f);
fprintf('Precision = %.4f\n',prec_f);
fprintf('Recall    = %.4f\n',rec_f);
fprintf('F1-score  = %.4f\n',f1_f);

%% 6. IMPORTANCIA GLOBAL

importanciaVariables(fis_grid,'GRID');
importanciaVariables(fis_fcm,'FCM');

%% 7. ANÁLISIS DE SESGO / SENSIBILIDAD
analizarSesgoContinuo(fis_grid, fis_fcm, xmin, xmax, 1, 'Age',         thr_grid, thr_fcm);
analizarSesgoContinuo(fis_grid, fis_fcm, xmin, xmax, 2, 'CreditScore', thr_grid, thr_fcm);

analizarSesgoCategoriaMarital(fis_grid,   fis_fcm, xmin, xmax, thr_grid, thr_fcm);
analizarSesgoCategoriaEducation(fis_grid, fis_fcm, xmin, xmax, thr_grid, thr_fcm);

%% 8. DISTRIBUCIÓN DE VARIABLES
variables = {'Age','CreditScore','MaritalStatus','Education'};

for i = 1:4
    figure;
    plotmf(fis_grid,'input',i);
    ax = gca;
    tick_norm = ax.XTick;
    tick_real = tick_norm .* (xmax(i) - xmin(i)) + xmin(i);
    xticklabels(round(tick_real));
    title(['Membership Functions - ' variables{i} ' (GRID)'])
end

%% 9. CLUSTERS FCM
figure('Name','Clusters FCM')
for i = 1:4
    subplot(2,2,i)
    plotmf(fis_fcm,'input',i)
    ax = gca;
    tick_norm = ax.XTick;
    tick_real = tick_norm .* (xmax(i) - xmin(i)) + xmin(i);
    xticklabels(round(tick_real));
    title(['Clusters FCM - ' variables{i}])
    grid on
end

%% ============================================================
% FUNCIONES
%% ============================================================

function compareProfiles(fis_grid, fis_fcm, xmin, xmax, thr_grid, thr_fcm)

fprintf('\n=========== ANALISIS DE REGLAS ACTIVAS GRID vs FCM ===========\n');

casos_reales = {
    'Joven (25)',        [25, 700, 1, 2];
    'Adulto (60)',       [60, 700, 1, 2];
    'Mal Score (400)',   [60, 400, 1, 2];
    'Buen Score (850)',  [60, 850, 1, 2];
    'Soltero',           [60, 700, 0, 2];
    'Divorciado',        [60, 700, 1, 2];
    'Casado',            [60, 700, 2, 2];
    'Edu Bajo',          [60, 700, 1, 0];
    'Edu Medio',         [60, 700, 1, 1];
    'Edu Alto',          [60, 700, 1, 3];
};

rules_grid_txt = showrule(fis_grid);
rules_fcm_txt  = showrule(fis_fcm);

for i = 1:size(casos_reales,1)

    nombre = casos_reales{i,1};
    x_real = casos_reales{i,2};

    % normalización
    x_norm = (x_real - xmin) ./ (xmax - xmin);

    % salidas
    y_grid = evalfis(fis_grid, x_norm);
    y_fcm  = evalfis(fis_fcm, x_norm);

    % clase
    if y_grid >= thr_grid
        class_grid = "DEFAULT";
    else
        class_grid = "NO DEFAULT";
    end

    if y_fcm >= thr_fcm
        class_fcm = "DEFAULT";
    else
        class_fcm = "NO DEFAULT";
    end

    % firing strengths
    [~,~,~,~,rf_grid] = evalfis(fis_grid, x_norm);
    [~,~,~,~,rf_fcm]  = evalfis(fis_fcm, x_norm);

    [rfg, idxg] = sort(rf_grid,'descend');
    [rfc, idxc] = sort(rf_fcm,'descend');

    fprintf('\n==================================================\n');
    fprintf('CASO: %s\n', nombre);

    fprintf('GRID -> %.4f => %s\n', y_grid, class_grid);
    fprintf('FCM  -> %.4f => %s\n', y_fcm, class_fcm);

    %% =========================
    % GRID RULES
    %% =========================
    fprintf('\n----- GRID: TOP REGLAS ACTIVAS -----\n');

    for j = 1:min(3,length(idxg))

        r = idxg(j);

        fprintf('\nRegla %d | firing = %.4f\n', r, rfg(j));

        % regla textual
        if r <= size(rules_grid_txt,1)
            disp(strtrim(rules_grid_txt(r,:)))
        end

        % coeficientes
        p = fis_grid.Outputs(1).MembershipFunctions(r).Parameters;

        fprintf('THEN: y = %.3f*A + %.3f*CS + %.3f*M + %.3f*E + %.3f\n', ...
            p(1), p(2), p(3), p(4), p(5));
    end

    %% =========================
    % FCM RULES
    %% =========================
    fprintf('\n----- FCM: TOP REGLAS ACTIVAS -----\n');

    for j = 1:min(3,length(idxc))

        r = idxc(j);

        fprintf('\nRegla %d | firing = %.4f\n', r, rfc(j));

        if r <= size(rules_fcm_txt,1)
            disp(strtrim(rules_fcm_txt(r,:)))
        end

        p = fis_fcm.Outputs(1).MembershipFunctions(r).Parameters;

        fprintf('THEN: y = %.3f*A + %.3f*CS + %.3f*M + %.3f*E + %.3f\n', ...
            p(1), p(2), p(3), p(4), p(5));
    end

end

end

function [acc,prec,rec,f1] = metricas(y,pred)

TP = sum(y==1 & pred==1);
TN = sum(y==0 & pred==0);
FP = sum(y==0 & pred==1);
FN = sum(y==1 & pred==0);

acc = (TP+TN)/(TP+TN+FP+FN);
prec = TP/(TP+FP+eps);
rec = TP/(TP+FN+eps);
f1 = 2*prec*rec/(prec+rec+eps);

end

function importanciaVariables(fis,nombre)

nRules = length(fis.Rules);

coef = zeros(nRules,4);

for r=1:nRules

    p = fis.Outputs(1).MembershipFunctions(r).Parameters;

    coef(r,:) = abs(p(1:4));

end

imp = mean(coef);

figure('Name',['Importancia ' nombre])

bar(imp)

set(gca,'XTickLabel',...
{'Age','CreditScore','MaritalStatus','Education'})

ylabel('Magnitud promedio')

title(['Importancia Global - ' nombre])

grid on

end

function analizarSesgoContinuo(fis_grid, fis_fcm, xmin, xmax, varIdx, varName, thr_grid, thr_fcm)

xx = linspace(xmin(varIdx), xmax(varIdx), 100);

grid_out = zeros(size(xx));
fcm_out  = zeros(size(xx));

base = [60 700 1 2];

for i = 1:length(xx)

    x = base;
    x(varIdx) = xx(i);

    x_norm = (x - xmin) ./ (xmax - xmin);

    grid_out(i) = evalfis(fis_grid, x_norm);
    fcm_out(i)  = evalfis(fis_fcm, x_norm);

end

figure('Name',['Variacion ' varName])

plot(xx, grid_out, 'b', 'LineWidth', 2); hold on
plot(xx, fcm_out,  'r', 'LineWidth', 2);
yline(thr_grid, 'b--', ['thr\_grid=' num2str(thr_grid)], 'LineWidth', 1.2);
yline(thr_fcm,  'r--', ['thr\_fcm='  num2str(thr_fcm)],  'LineWidth', 1.2);

xlim([xmin(varIdx) xmax(varIdx)])
xlabel(varName)
ylabel('Salida ANFIS')
title(['Variacion por ' varName])
legend('GRID','FCM')
ylim([0 1])
grid on

end


function analizarSesgoCategoriaMarital(fis_grid, fis_fcm, xmin, xmax, thr_grid, thr_fcm)

cats   = [0 1 2];
labels = {'Single','Divorced','Married'};

grid_out = zeros(size(cats));
fcm_out  = zeros(size(cats));

base = [60 700 1 2];

for i = 1:length(cats)

    x = base;
    x(3) = cats(i);

    x_norm = (x - xmin) ./ (xmax - xmin);

    grid_out(i) = evalfis(fis_grid, x_norm);
    fcm_out(i)  = evalfis(fis_fcm,  x_norm);

end

figure('Name','Variacion Estado Matrimonial')

plot(cats, grid_out, '-o', 'LineWidth', 2); hold on
plot(cats, fcm_out,  '-o', 'LineWidth', 2);
yline(thr_grid, 'b--', ['thr\_grid=' num2str(thr_grid)], 'LineWidth', 1.2);
yline(thr_fcm,  'r--', ['thr\_fcm='  num2str(thr_fcm)],  'LineWidth', 1.2);

xticks(cats)
xticklabels(labels)
ylabel('Salida ANFIS')
title('Variacion por Marital Status')
legend('GRID','FCM')
ylim([0 1])
grid on

end


function analizarSesgoCategoriaEducation(fis_grid, fis_fcm, xmin, xmax, thr_grid, thr_fcm)

cats   = [0 1 2 3];
labels = {'High School','Bachelor','Master','PhD'};

grid_out = zeros(size(cats));
fcm_out  = zeros(size(cats));

base = [60 700 1 2];

for i = 1:length(cats)

    x = base;
    x(4) = cats(i);

    x_norm = (x - xmin) ./ (xmax - xmin);

    grid_out(i) = evalfis(fis_grid, x_norm);
    fcm_out(i)  = evalfis(fis_fcm,  x_norm);

end

figure('Name','Variacion - Educacion')

plot(cats, grid_out, '-o', 'LineWidth', 2); hold on
plot(cats, fcm_out,  '-o', 'LineWidth', 2);
yline(thr_grid, 'b--', ['thr\_grid=' num2str(thr_grid)], 'LineWidth', 1.2);
yline(thr_fcm,  'r--', ['thr\_fcm='  num2str(thr_fcm)],  'LineWidth', 1.2);

xticks(cats)
xticklabels(labels)
ylabel('Salida ANFIS')
title('Variacion por Education')
legend('GRID','FCM')
ylim([0 1])
grid on

end