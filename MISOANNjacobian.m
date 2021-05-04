function [J] = MISOANNjacobian(TrainingINPUT,Wg,bh,Wc,bc,lambda);
NumberOfTrainingData = size(TrainingINPUT,1);
S = size(Wg,1); R = size(Wg,2);
for i=1:NumberOfTrainingData
    for j=1:S*R
        k = mod(j-1,S) + 1;
        m = fix((j-1)/S) + 1;
        J(i,j) = -[Wc(1,k)*TrainingINPUT(i,m)*[1-tanh(Wg(k,:)*TrainingINPUT(i,:)'+bh(k))^2]];
    end
    for j=S*R+1:S*R+S
        J(i,j) = -[Wc(1,j-S*R)*[1-tanh(Wg(j-S*R,:)*TrainingINPUT(i,:)'+bh(j-S*R))^2]];
    end
    for j=S*(R+1)+1:S*(R+2)
        J(i,j) = -[tanh(Wg(j-(R+1)*S,:)*TrainingINPUT(i,:)'+bh(j-(R+1)*S))];
    end
    for j=S*(R+2)+1:S*(R+2)+1
        J(i,j) = -[1];
    end
end
J = J/sqrt(NumberOfTrainingData);
J = [J; lambda*eye(S*(R+2)+1)];