function [ cmap ] = VorticityColormap( N )
%VORTICITYCOLORMAP Summary of this function goes here
%   Detailed explanation goes here
cmap=colormap(jet(N));
close
for i=N/8+1:7*N/16
    cmap(i,:)=[(i-N/8)/(5*N/16),(i-N/8)/(5*N/16),1];    % blue fading to white
end

    cmap(7*N/16+1:9*N/16,:)=1;  % white

for i=9*N/16+1:7*N/8
    cmap(i,:)=[1,1-(i-9*N/16)/(5*N/16),1-(i-9*N/16)/(5*N/16)];    % white turning red
end

end

