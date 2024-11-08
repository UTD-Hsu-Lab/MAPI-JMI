%% Distance Analysis of MAPI PC UV-vis Curves for 240824
clear, close all; clc;

%% Load in Thermal Annealed Reference Data

% Large Area
TAref = readmatrix('240824CA02_TA2.txt');
TAref = TAref(TAref(:,1) >= 600,:);
TAref = TAref(TAref(:,1) <= 850,:);

n = numel(TAref);

%% Import Photonically Cured MAPI Data

LHS25 = readmatrix('240824CA04_PC_LHS25_2.txt');
LHS25 = LHS25(LHS25(:,1) >= 600,:);
LHS25 = LHS25(LHS25(:,1) <= 850,:);

LHS03 = readmatrix('240824CA05_PC_LHS03_2.txt');
LHS03 = LHS03(LHS03(:,1) >= 600,:);
LHS03 = LHS03(LHS03(:,1) <= 850,:);

LHS04 = readmatrix('240824CA06_PC_LHS04_2.txt');
LHS04 = LHS04(LHS04(:,1) >= 600,:);
LHS04 = LHS04(LHS04(:,1) <= 850,:);

%% Calculate Distance Metrics

DLLHS25 = procrustes(LHS25,TAref);
DROTLLHS25 = procrustes_ml(LHS25, TAref,'rotation',false,'scaling',false);
FDLLHS25 = frechetDistance(TAref, LHS25);

DLLHS03 = procrustes(LHS03,TAref);
DROTLLHS03 = procrustes_ml(LHS03, TAref,'rotation',false,'scaling',false);
FDLLHS03 = frechetDistance(TAref, LHS03);

DLLHS04 = procrustes(LHS04,TAref);
DROTLLHS04 = procrustes_ml(LHS04, TAref,'rotation',false,'scaling',false);
FDLLHS04 = frechetDistance(TAref, LHS04);

%% Collating the Data into Vectors

DL = [DLLHS25;DLLHS03;DLLHS04];
DROTL = [DROTLLHS25;DROTLLHS03;DROTLLHS04];
FDL = [FDLLHS25;FDLLHS03;FDLLHS04];

D_L = [DL,DROTL,FDL];
