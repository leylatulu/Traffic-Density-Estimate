function [x]=Matrix2Vector(Wg,bh,Wc,bc)
x = [];
for i=1:size(Wg,2)
    x = [x; Wg(:,i)];
end
x = [x; bh; Wc'; bc];