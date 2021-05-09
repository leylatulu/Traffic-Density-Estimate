function [yhat]=MISOANNio(INPUT,Wg,bh,Wc,bc)
NumberOfData = size(INPUT,1);
S = size(Wg,1);
for i=1:NumberOfData
    yhat(i,1) = Wc*tanh(Wg*INPUT(i,:)' + bh) + bc;
end 