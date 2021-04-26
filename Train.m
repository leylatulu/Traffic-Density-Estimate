clear all
close all
clc

T=readtable('Train.csv');

x=T{:,15}';
 
save('Train','x');
  

Z = x;
NumberOfInputs = 25;
NumberOfNeurons = 500; 
LengthOfTimeSeries = 6000;
PredictionHorizon = 48;
lambda = 0.001;


Zmin = min(Z); Zmax = max(Z);
x = (x-[ones(size(x,1),1)*Zmin])./([ones(size(x,1),1)*Zmax]-[ones(size(x,1),1)*Zmin]);


INPUT = []; OUTPUT = []; k = 0; loop = 1;
while loop
    k=k+1;
    INPUT = [INPUT ; x(k+0:k+NumberOfInputs-1)];
    OUTPUT = [OUTPUT ; x(k+NumberOfInputs)];
    if k+NumberOfInputs>=LengthOfTimeSeries; loop=0; end
end


NumberOfInputs = size(INPUT,2); R = NumberOfInputs;
NumberOfData = length(INPUT);
TrainingIndex =1:2:NumberOfData;
ValidationIndex = 2:2:NumberOfData;
TrainingINPUT = INPUT(TrainingIndex,:);
TrainingOUTPUT = OUTPUT(TrainingIndex,:);
ValidationINPUT = INPUT(ValidationIndex,:);
ValidationOUTPUT = OUTPUT(ValidationIndex,:);
NumberOfTrainingData = size(TrainingINPUT,1);
NumberOfValidationData = size(ValidationINPUT,1);
Nmax=700; MaxNumOfAllowableInc = 10;
eps1 = 1e-6;
Smax = (NumberOfTrainingData-1)/3;


for S = 70;
    Wg = rand(S,R) - 0.5;
    bh = rand(S,1) - 0.5;
    Wc = rand(1,S) - 0.5;
    bc = rand(1,1) - 0.5;
    [x] = Matrix2Vector(Wg,bh,Wc,bc);
    [J] = MISOANNjacobian(TrainingINPUT,Wg,bh,Wc,bc,lambda);
    [yhat] = MISOANNio(TrainingINPUT,Wg,bh,Wc,bc);
    eTRAINING = [[TrainingOUTPUT-yhat]/sqrt(NumberOfTrainingData); lambda*x];
    fTRAINING = eTRAINING'*eTRAINING;
    yhat = MISOANNio(ValidationINPUT,Wg,bh,Wc,bc);
    eVALIDATION = [[ValidationOUTPUT-yhat]/sqrt(NumberOfValidationData); lambda*x];
    fVALIDATION = eVALIDATION'*eVALIDATION;
    gk = [2*J'*eTRAINING];
    I = eye(length(x)); mu = 1; mumax=1e+20;
    MainLoop = 1; fVALIDATIONbest = fVALIDATION; k = 0; FTRA = 0; FVAL = 0;
    fprintf('k:%4.0f\t ftra:%4.8f\t fval*:%4.8f\t ||g||:%4.6f\n',([k log10(fTRAINING) log10(fVALIDATION) norm(gk)])) 
    while MainLoop
        k = k+1;
        yhat = MISOANNio(TrainingINPUT,Wg,bh,Wc,bc);
        eTRAINING = [[TrainingOUTPUT-yhat]/sqrt(NumberOfTrainingData); lambda*x];
        fTRAINING = eTRAINING'*eTRAINING;
        [J] = MISOANNjacobian(TrainingINPUT,Wg,bh,Wc,bc,lambda);
        InnerLoop = 1;
        while InnerLoop
            pk = -inv(J'*J+mu*I)*J'*eTRAINING;
            z = x + pk;
            [Wgz,bhz,Wcz,bcz]=Vector2Matrix(z,S,R);
            yhatz = MISOANNio(TrainingINPUT,Wgz,bhz,Wcz,bcz);
            ez = [[TrainingOUTPUT-yhatz]/sqrt(NumberOfTrainingData); lambda*z];
            fz = ez'*ez;
            if fz<fTRAINING
                x = z; Wg = Wgz; bh = bhz; Wc = Wcz; bc = bcz; fTRAINING = fz;
                mu = mu/10; InnerLoop = 0;
            else
                mu = mu*10;
            end
            if mu>mumax; InnerLoop = 0; MainLoop = 0; end
        end
        gk = [2*J'*eTRAINING];
        yhat = MISOANNio(ValidationINPUT,Wg,bh,Wc,bc);
        eVALIDATION = [[ValidationOUTPUT - yhat]/sqrt(NumberOfValidationData); lambda*x];
        fVALIDATION = eVALIDATION'*eVALIDATION;
        FVAL(k) = log10(fVALIDATION);
        if fVALIDATION<fVALIDATIONbest
            fVALIDATIONbest = fVALIDATION;
            xBEST = x; WgBEST = Wg; bhBEST = bh; WcBEST = Wc; bcBEST = bc; kBEST = k;
            FvalIncCounter = 0;
        else
            FvalIncCounter=FvalIncCounter + 1;
        end
        FTRA(k) = log10(fTRAINING);
        if [k>Nmax] | [norm(gk)<eps1] | [FvalIncCounter>=MaxNumOfAllowableInc]; MainLoop = 0; end
        fprintf('k:%4.0f\t ftra:%4.8f\t fval*:%4.8f\t ||g||:%4.6f\n',([k log10(fTRAINING) log10(fVALIDATION) norm(gk)]))
    end
    fprintf('S:%4.0f\t ftra:%4.8f\n',([S log10(fVALIDATIONbest)]))
end


subplot(311)
plot(FTRA,'r'); hold on; grid
plot(FVAL,'b');
h = line([kBEST kBEST],[min([FTRA,FVAL]) max([FTRA,FVAL])]); set(h,'LineWidth',1.0); set(h,'LineStyle','--'); set(h,'Color',[0 1 0]);
xlabel('k');
ylabel('F_(tra)(k),  F_(val)(k)');
title('Training and Validation errors');
h = gca; set(h,'FontName','Cambria'); set(h,'FontSize',10)


subplot(312)
TestIndex = LengthOfTimeSeries+1:LengthOfTimeSeries+PredictionHorizon;
plot(TestIndex,Z(TestIndex),'r'); hold on
plot(TestIndex,Z(TestIndex),'ro');
PREDICTIONS = []; input=[Z(LengthOfTimeSeries-R+1:LengthOfTimeSeries)];


input = [input-Zmin]/[Zmax-Zmin];
for k=1:PredictionHorizon
    [yhat] = MISOANNio(input,WgBEST,bhBEST,WcBEST,bcBEST);
    PREDICTIONS = [PREDICTIONS; yhat];
    input = [input(2:end),yhat];
end
PREDICTIONS=Zmin+(Zmax-Zmin)*PREDICTIONS;
plot(TestIndex,PREDICTIONS,'b')
plot(TestIndex,PREDICTIONS,'bx')
xlabel('\it t_[i]');
ylabel('\it y_[i]  yhat_[i]');
title('Predictions');
h = gca; set(h,'FontName','Cambria'); set(h,'FontSize',10)


subplot(313)
h = stem((xBEST),'filled'); set(h,'Color',[1 0 0]); set(h,'MarkerSize',2); set(h,'LineWidth',2);
xlabel('\it i');
ylabel('\it x_[i]');
title('Parameters');
h = gca; set(h,'FontName','Cambria');  set(h,'FontSize',10)


set(gcf,'color',[1 1 1])
set(gcf,'Position',[236 209 1230 420])
