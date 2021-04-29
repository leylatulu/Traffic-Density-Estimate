function [Wg,bh,Wc,bc]=Vector2Matrix(x,S,R)
Wg = []; bh = []; Wc = []; bc = [];
for i=1:R
    Wg = [Wg, x((i-1)*S+1:i*S)];
end
bh = x(R*S+1:R*S+S);
Wc = x(S*(R+1)+1:S*(R+2))';
bc = x(S*(R+2)+1);