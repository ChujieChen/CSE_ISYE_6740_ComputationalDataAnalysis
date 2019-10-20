function [Acc]=AccMeasure(T,idx)

% This is the accuracy calculation function which you can use for evaluation. 
% T is the true label and idx would be from your function
% You are welcomed to use your own implementation for evaluation. And other evaluation metrics are encouraged.	
% 

k=max(T);
n=length(T);
for i=1:k
    temp=find(T==i);
    a{i}=temp;
end

b1=[];
t1=zeros(1,k);
for i=1:k
    tt1=find(idx==i);
    for j=1:k
       t1(j)=sum(ismember(tt1,a{j}));
    end
    b1=[b1;t1]; 
end
    Members=zeros(1,k); 
    
P = perms((1:k));
    Acc1=0;
for pi=1:size(P,1)
    for ki=1:k
        Members(ki)=b1(P(pi,ki),ki);
    end
    if sum(Members)>Acc1
        match=P(pi,:);
        Acc1=sum(Members);
    end
end

Acc=Acc1/n*100; 
